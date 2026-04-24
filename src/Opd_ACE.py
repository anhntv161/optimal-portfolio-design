"""
ACE Solver wrapper for OPD (Orthogonal {v,b,r}-Design) using PyCSP3 + ACE.

Workflow:
  1. Parse v, b, r from input file.
  2. Run Opd.py (PyCSP3) to generate input.xml.
  3. Solve with ACE Solver (Java).
  4. Parse ACE output to extract lambda, time, and status.
  5. Write results to Excel (same format as other OPD solvers).

Usage:
    python Opd_ACE.py --input input/small
    python Opd_ACE.py --input input/small/small_1.txt
    python Opd_ACE.py --input input/small --timeout 300 --variant aux
    python Opd_ACE.py --input input/small --ace-jar /path/to/ACE.jar

Requirements:
    pip install openpyxl
    Java must be installed and on PATH.
    ACE-*.jar must be available (specify via --ace-jar or ACE_JAR env variable).
"""

import argparse
import math
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_input_file(filepath):
    """Parse v, b, r from input file (same format as other OPD solvers)."""
    params = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        v_match = re.search(r'v\s*=\s*(\d+)', content)
        b_match = re.search(r'b\s*=\s*(\d+)', content)
        r_match = re.search(r'r\s*=\s*(\d+)', content)

        if v_match: params['v'] = int(v_match.group(1))
        if b_match: params['b'] = int(b_match.group(1))
        if r_match: params['r'] = int(r_match.group(1))

        return params
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
        return None


def compute_lower_bound(v, b, r):
    """Compute theoretical lower bound on lambda: λ ≥ ⌈r(rv - b) / (b(v-1))⌉."""
    if v <= 1:
        return 0
    numerator = r * (r * v - b)
    denominator = b * (v - 1)
    if denominator == 0:
        return 0
    return max(0, math.ceil(numerator / denominator))


# ---------------------------------------------------------------------------
# ACE output parsing
# ---------------------------------------------------------------------------

def parse_ace_output(ace_output):
    """
    Parse ACE Solver stdout/stderr to extract key results.

    Returns dict with keys:
        status       : 'OPTIMAL' | 'FEASIBLE' | 'UNSAT' | 'TIMEOUT' | 'ERROR'
        lambda_val   : int or None   (best objective value found)
        wck_first    : float or None (wall-clock time to first solution)
        wck_last     : float or None (wall-clock time to last/best solution)
        real_time    : float or None (total wall-clock time reported by ACE)
        found_solutions : int        (number of solutions found)
        incomplete   : bool          (True if exploration was incomplete)
    """
    result = {
        'status': 'UNKNOWN',
        'lambda_val': None,
        'wck_first': None,
        'wck_last': None,
        'real_time': None,
        'found_solutions': 0,
        'incomplete': False,
        'num_variables': None,
        'num_constraints': None,
    }

    # --- status line ---
    # "s SATISFIABLE", "s UNSATISFIABLE", "s OPTIMUM FOUND"
    status_match = re.search(r'^s\s+(.+)$', ace_output, re.MULTILINE)
    if status_match:
        status_str = status_match.group(1).strip().upper()
        if 'UNSATISFIABLE' in status_str:
            result['status'] = 'UNSAT'
        elif 'OPTIMUM' in status_str:
            result['status'] = 'OPTIMAL'
        elif 'SATISFIABLE' in status_str:
            result['status'] = 'FEASIBLE'

    # --- objective bound ("d BOUND  <value>") ---
    bound_match = re.search(r'^d BOUND\s+(\d+)', ace_output, re.MULTILINE)
    if bound_match:
        result['lambda_val'] = int(bound_match.group(1))

    # Fallback: last "o <value>" line (incumbent updates)
    o_matches = re.findall(r'^o\s+(\d+)', ace_output, re.MULTILINE)
    if o_matches and result['lambda_val'] is None:
        result['lambda_val'] = int(o_matches[-1])

    # --- timing ---
    wck_first_match = re.search(r'^d WCK FIRST\s+([\d.]+)', ace_output, re.MULTILINE)
    if wck_first_match:
        result['wck_first'] = float(wck_first_match.group(1))

    wck_last_match = re.search(r'^d WCK LAST\s+([\d.]+)', ace_output, re.MULTILINE)
    if wck_last_match:
        result['wck_last'] = float(wck_last_match.group(1))

    real_time_match = re.search(r'^c real time\s*:\s*([\d.]+)', ace_output, re.MULTILINE)
    if real_time_match:
        result['real_time'] = float(real_time_match.group(1))

    # --- number of solutions ---
    sols_match = re.search(r'^d FOUND SOLUTIONS\s+(\d+)', ace_output, re.MULTILINE)
    if sols_match:
        result['found_solutions'] = int(sols_match.group(1))

    # --- number of variables and constraints ---
    vars_match = re.search(r'^d VARIABLES\s+(\d+)', ace_output, re.MULTILINE)
    if vars_match:
        result['num_variables'] = int(vars_match.group(1))

    cons_match = re.search(r'^d CONSTRAINTS\s+(\d+)', ace_output, re.MULTILINE)
    if cons_match:
        result['num_constraints'] = int(cons_match.group(1))

    # --- incomplete flag ---
    if re.search(r'd INCOMPLETE EXPLORATION', ace_output, re.MULTILINE):
        result['incomplete'] = True

    # --- derive status when not explicitly given ---
    if result['status'] == 'UNKNOWN':
        if result['lambda_val'] is not None:
            result['status'] = 'FEASIBLE'
        elif result['incomplete']:
            result['status'] = 'TIMEOUT'
        else:
            result['status'] = 'UNKNOWN'

    # Timeout with incumbent → FEASIBLE (not TIMEOUT)
    if result['status'] == 'TIMEOUT' and result['lambda_val'] is not None:
        result['status'] = 'FEASIBLE'

    return result


# ---------------------------------------------------------------------------
# Core solve function for a single file
# ---------------------------------------------------------------------------

def solve_from_file(filepath, ace_jar, timeout=3600, variant="", quiet=False):
    """
    Solve one OPD instance using PyCSP3 (Opd.py) + ACE Solver.

    Steps:
      1. Parse v, b, r from filepath.
      2. Run `python Opd.py -data=[v,b,r] -variant=<variant> -output=<xml>` to
         generate the XCSP3 XML file.
      3. Run `java -jar <ace_jar> <xml> -t=<timeout>s -npc` and capture output.
      4. Parse the output and return a result dict.

    Returns a dict with keys matching COLUMNS (for Excel export).
    """
    start_wall = time.time()
    filename = os.path.basename(filepath)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\n{'='*60}")
    print(f"Processing file: {filename}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Parse parameters
    # ------------------------------------------------------------------
    params = parse_input_file(filepath)
    if not params or not all(k in params for k in ('v', 'b', 'r')):
        print("Error: Could not extract v, b, r from file.")
        return {
            'File': filename, 'v': None, 'b': None, 'r': None,
            'Lower Bound': None, 'Optimal Lambda': None,
            'Variant': variant or 'direct',
            'Num Variables': None, 'Num Constraints': None,
            'Time (s)': 0, 'ACE Real Time (s)': None,
            'WCK First (s)': None, 'WCK Last (s)': None,
            'Solutions Found': 0, 'Status': 'Parse Error'
        }

    v, b, r = params['v'], params['b'], params['r']
    lb = compute_lower_bound(v, b, r)
    print(f"Parameters: v={v}, b={b}, r={r}  (lower_bound={lb})")

    # ------------------------------------------------------------------
    # 2. Generate XML via Opd.py
    # ------------------------------------------------------------------
    data_str = f"[{v},{b},{r}]"
    xml_path = os.path.join(script_dir, f"_ace_tmp_{filename}.xml")

    opd_cmd = [
        sys.executable,
        os.path.join(script_dir, "Opd.py"),
        f"-data={data_str}",
        f"-output={xml_path}",
    ]
    if variant:
        opd_cmd.append(f"-variant={variant}")

    print(f"\n[1/2] Generating XML with Opd.py...")
    if not quiet:
        print(f"  Command: {' '.join(opd_cmd)}")

    try:
        gen_result = subprocess.run(
            opd_cmd,
            capture_output=True, text=True,
            cwd=script_dir, timeout=60
        )
        if gen_result.returncode != 0 or not os.path.exists(xml_path):
            err_msg = gen_result.stderr.strip() or gen_result.stdout.strip()
            print(f"Error: Opd.py failed to generate XML.\n{err_msg}")
            return {
                'File': filename, 'v': v, 'b': b, 'r': r,
                'Lower Bound': lb, 'Optimal Lambda': None,
                'Variant': variant or 'direct',
                'Num Variables': None, 'Num Constraints': None,
                'Time (s)': round(time.time() - start_wall, 3),
                'ACE Real Time (s)': None,
                'WCK First (s)': None, 'WCK Last (s)': None,
                'Solutions Found': 0, 'Status': 'XML Gen Error'
            }
        if not quiet:
            print(f"  XML generated: {xml_path}")
    except subprocess.TimeoutExpired:
        print("Error: Opd.py timed out during XML generation.")
        return {
            'File': filename, 'v': v, 'b': b, 'r': r,
            'Lower Bound': lb, 'Optimal Lambda': None,
            'Variant': variant or 'direct',
            'Num Variables': None, 'Num Constraints': None,
            'Time (s)': round(time.time() - start_wall, 3),
            'ACE Real Time (s)': None,
            'WCK First (s)': None, 'WCK Last (s)': None,
            'Solutions Found': 0, 'Status': 'XML Gen Timeout'
        }

    # Parse variable and constraint counts directly from the XML model
    xml_num_vars, xml_num_cons = parse_xml_stats(xml_path)
    print(f"  Model stats:   variables={xml_num_vars}, constraints={xml_num_cons}")

    # ------------------------------------------------------------------
    # 3. Run ACE Solver
    # ------------------------------------------------------------------
    elapsed_after_gen = time.time() - start_wall
    remaining_timeout = max(1, timeout - int(elapsed_after_gen))
    timeout_str = f"{remaining_timeout}s"

    # Try to find java executable
    java_bin = "java"
    is_windows_java = False
    try:
        subprocess.run(["java", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(["java.exe", "-version"], capture_output=True, check=True)
            java_bin = "java.exe"
            is_windows_java = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # Convert paths for Windows Java on WSL if necessary
    ace_xml_path = xml_path
    ace_jar_path = ace_jar
    if is_windows_java:
        try:
            # Try to convert paths using wslpath
            wsl_xml = subprocess.check_output(["wslpath", "-w", xml_path], text=True).strip()
            wsl_jar = subprocess.check_output(["wslpath", "-w", ace_jar], text=True).strip()
            ace_xml_path = wsl_xml
            ace_jar_path = wsl_jar
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    ace_cmd = [
        java_bin, "-jar", ace_jar_path,
        ace_xml_path,
        f"-t={timeout_str}",
        "-npc"
    ]

    print(f"\n[2/2] Running ACE Solver (timeout={timeout_str})...")
    if not quiet:
        print(f"  Command: {' '.join(ace_cmd)}")

    try:
        ace_start = time.time()
        ace_result = subprocess.run(
            ace_cmd,
            capture_output=True, text=True,
            cwd=script_dir, timeout=remaining_timeout + 30
        )
        ace_elapsed = time.time() - ace_start
        ace_output = ace_result.stdout + ace_result.stderr
    except subprocess.TimeoutExpired:
        ace_elapsed = time.time() - ace_start
        print(f"ACE Solver process timed out after {ace_elapsed:.1f}s")
        _cleanup_xml(xml_path)
        total_time = time.time() - start_wall
        return {
            'File': filename, 'v': v, 'b': b, 'r': r,
            'Lower Bound': lb, 'Optimal Lambda': None,
            'Variant': variant or 'direct',
            'Num Variables': xml_num_vars, 'Num Constraints': xml_num_cons,
            'Time (s)': round(total_time, 3),
            'ACE Real Time (s)': round(ace_elapsed, 3),
            'WCK First (s)': None, 'WCK Last (s)': None,
            'Solutions Found': 0, 'Status': 'Timeout'
        }
    except FileNotFoundError:
        print(f"Error: 'java' not found. Make sure Java is installed and on PATH.")
        _cleanup_xml(xml_path)
        return {
            'File': filename, 'v': v, 'b': b, 'r': r,
            'Lower Bound': lb, 'Optimal Lambda': None,
            'Variant': variant or 'direct',
            'Num Variables': xml_num_vars, 'Num Constraints': xml_num_cons,
            'Time (s)': round(time.time() - start_wall, 3),
            'ACE Real Time (s)': None,
            'WCK First (s)': None, 'WCK Last (s)': None,
            'Solutions Found': 0, 'Status': 'Java Not Found'
        }

    _cleanup_xml(xml_path)

    # ------------------------------------------------------------------
    # 4. Parse ACE output
    # ------------------------------------------------------------------
    parsed = parse_ace_output(ace_output)

    total_time = time.time() - start_wall

    # Derive human-readable status
    ace_status = parsed['status']
    if ace_status == 'OPTIMAL':
        display_status = 'Optimal'
    elif ace_status == 'FEASIBLE':
        display_status = 'Feasible (Timeout)' if parsed['incomplete'] else 'Feasible'
    elif ace_status == 'UNSAT':
        display_status = 'UNSAT'
    else:
        display_status = 'Timeout' if parsed['lambda_val'] is None else 'Feasible (Timeout)'

    lam = parsed['lambda_val']

    # Summary
    print(f"\nRESULT for {filename}:")
    print(f"  Status:         {display_status}")
    print(f"  Optimal lambda: {lam}")
    print(f"  Lower bound:    {lb}")
    if lam is not None:
        print(f"  Gap from LB:    {lam - lb}")
    print(f"  Num variables:  {xml_num_vars}")
    print(f"  Num constraints:{xml_num_cons}")
    print(f"  ACE real time:  {parsed['real_time']}s")
    print(f"  Total time:     {total_time:.3f}s")
    print(f"  Solutions found: {parsed['found_solutions']}")

    return {
        'File':             filename,
        'v':                v,
        'b':                b,
        'r':                r,
        'Lower Bound':      lb,
        'Optimal Lambda':   lam,
        'Variant':          variant or 'direct',
        'Num Variables':    xml_num_vars,
        'Num Constraints':  xml_num_cons,
        'Time (s)':         round(total_time, 3),
        'ACE Real Time (s)': parsed['real_time'],
        'WCK First (s)':    parsed['wck_first'],
        'WCK Last (s)':     parsed['wck_last'],
        'Solutions Found':  parsed['found_solutions'],
        'Status':           display_status,
    }


def _cleanup_xml(xml_path):
    """Remove temporary XML file."""
    try:
        if os.path.exists(xml_path):
            os.remove(xml_path)
    except OSError:
        pass


def parse_xml_stats(xml_path):
    """
    Parse a XCSP3 XML file to count the number of variables and constraints.

    Variables:
      - Each <var> element counts as 1.
      - Each <array size="[d1][d2]..."> element counts as d1*d2*... variables.
    Constraints:
      - Each direct child of <constraints> counts as 1, EXCEPT <group> elements
        whose inner <args> children each count as 1 separate constraint.

    Returns (num_variables, num_constraints) or (None, None) on error.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # --- variables ---
        num_vars = 0
        variables_elem = root.find('variables')
        if variables_elem is not None:
            for elem in variables_elem:
                if elem.tag == 'var':
                    num_vars += 1
                elif elem.tag == 'array':
                    size_attr = elem.get('size', '')
                    dims = re.findall(r'\[(\d+)\]', size_attr)
                    if dims:
                        count = 1
                        for d in dims:
                            count *= int(d)
                        num_vars += count
                    else:
                        num_vars += 1  # fallback

        # --- constraints ---
        num_cons = 0
        constraints_elem = root.find('constraints')
        if constraints_elem is not None:
            for elem in constraints_elem:
                if elem.tag == 'group':
                    # Each <args> inside a group is one constraint instantiation
                    args_count = len(elem.findall('args'))
                    num_cons += args_count if args_count > 0 else 1
                else:
                    num_cons += 1

        return num_vars, num_cons
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Default ACE JAR path resolution
# ---------------------------------------------------------------------------

def find_ace_jar():
    """
    Attempt to locate the ACE JAR automatically.
    Search order:
      1. ACE_JAR environment variable
      2. Common pycsp3 package locations (without importing pycsp3)
      3. Current working directory (*.jar glob)
    Returns path string if found, else None.
    """
    # 1. Environment variable
    env_jar = os.environ.get('ACE_JAR')
    if env_jar and os.path.isfile(env_jar):
        return env_jar

    # 2. pycsp3 bundled solver (using importlib to avoid triggering modeling engine)
    try:
        import importlib.util
        spec = importlib.util.find_spec('pycsp3')
        if spec and spec.origin:
            pycsp3_dir = os.path.dirname(spec.origin)
            candidate = os.path.join(pycsp3_dir, 'solvers', 'ace')
            if os.path.isdir(candidate):
                jars = sorted(f for f in os.listdir(candidate) if f.endswith('.jar'))
                if jars:
                    return os.path.join(candidate, jars[-1])
    except Exception:
        pass

    # 3. CWD
    import glob
    jars = sorted(glob.glob(os.path.join(os.getcwd(), '*.jar')))
    if jars:
        return jars[0]

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ACE Solver wrapper for OPD (PyCSP3 + ACE)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Opd_ACE.py --input input/small
  python Opd_ACE.py --input input/small/small_1.txt
  python Opd_ACE.py --input input/small --timeout 300 --variant aux
  python Opd_ACE.py --input input/small --ace-jar /opt/ACE-2.5.jar --quiet
        """
    )

    parser.add_argument('--input',   type=str, required=True,
                        help='Input file or directory path')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Solver time limit per instance in seconds (default: 3600)')
    parser.add_argument('--variant', choices=['', 'aux'], default='',
                        help="PyCSP3 model variant: '' (direct) or 'aux' (auxiliary) (default: '')")
    parser.add_argument('--ace-jar', type=str, default=None,
                        help='Path to ACE-*.jar file. Overrides ACE_JAR env variable.')
    parser.add_argument('--quiet',   action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()
    input_path = args.input

    # Allow short paths relative to cwd
    if not os.path.exists(input_path):
        candidate = os.path.join('input', input_path)
        if os.path.exists(candidate):
            input_path = candidate

    # Resolve ACE JAR
    ace_jar = args.ace_jar or find_ace_jar()
    if ace_jar is None:
        print(
            "Error: ACE JAR not found.\n"
            "  Set the ACE_JAR environment variable or use --ace-jar <path>.\n"
            "  Example:\n"
            "    export ACE_JAR=/path/to/ACE-2.5.jar\n"
            "    python Opd_ACE.py --input input/small"
        )
        sys.exit(1)

    print(f"Using ACE JAR: {ace_jar}")

    COLUMNS = [
        'File', 'v', 'b', 'r',
        'Lower Bound', 'Optimal Lambda',
        'Variant',
        'Num Variables', 'Num Constraints',
        'Time (s)', 'ACE Real Time (s)',
        'WCK First (s)', 'WCK Last (s)',
        'Solutions Found', 'Status'
    ]

    if os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        files = sorted(f for f in os.listdir(input_path) if f.endswith('.txt'))

        if not files:
            print("No .txt files found in directory.")
            sys.exit(0)

        print(f"Found {len(files)} input file(s).")

        try:
            import openpyxl
            has_openpyxl = True
        except ImportError:
            print("Warning: openpyxl not found. Excel export will be disabled.")
            has_openpyxl = False

        folder_name  = os.path.basename(os.path.normpath(input_path))
        script_name  = os.path.splitext(os.path.basename(__file__))[0]
        output_file  = f"result_{folder_name}_Opd_ACE.xlsx"

        for f in files:
            filepath = os.path.join(input_path, f)
            res = solve_from_file(filepath, ace_jar, args.timeout, args.variant, args.quiet)

            if res and has_openpyxl:
                if not os.path.exists(output_file):
                    wb = openpyxl.Workbook()
                    ws = wb.active
                    ws.append(COLUMNS)
                else:
                    wb = openpyxl.load_workbook(output_file)
                    ws = wb.active
                ws.append([res.get(c) for c in COLUMNS])
                wb.save(output_file)
                print(f"\n{'-'*60}")
                print(f"Result for {f} appended to {output_file}")
                print(f"{'-'*60}")
            elif res:
                print(f"{res['File']}: lambda={res['Optimal Lambda']}, "
                      f"time={res['Time (s)']}s, status={res['Status']}")

    elif os.path.isfile(input_path):
        solve_from_file(input_path, ace_jar, args.timeout, args.variant, args.quiet)

    else:
        print(f"Error: Input path '{args.input}' not found.")
        sys.exit(1)
