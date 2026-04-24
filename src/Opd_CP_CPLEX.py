"""
CP Model for OPD (Orthogonal {v,b,r}-Design) using IBM CPLEX CP Optimizer.

Equivalent to Opd.py (PyCSP3) but solved via docplex.cp (DOcplex CP Optimizer).

Usage:
    python Opd_CP_CPLEX.py --input input/small
    python Opd_CP_CPLEX.py --input input/small/small_5.txt
    python Opd_CP_CPLEX.py --input input/small --timeout 300 --variant aux

Requirements:
    pip install docplex
    IBM CPLEX Optimization Studio must be installed and on the system PATH.
"""

import argparse
import time
import math
import os
import re
import sys
import multiprocessing
from itertools import combinations

from docplex.cp.model import CpoModel


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_input_file(filepath):
    """Parse v, b, r from input file (same format as Opd_pure_sat_ver1.py)."""
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


# ---------------------------------------------------------------------------
# CP Solver class
# ---------------------------------------------------------------------------

class OpdCPCPLEX:
    """CP-based OPD solver using IBM CPLEX CP Optimizer (docplex.cp)."""

    def __init__(self, v, b, r, variant="", sym_options=None, verbose=True):
        self.v = v
        self.b = b
        self.r = r
        self.variant = variant
        self.sym_options = sym_options or []
        self.verbose = verbose

        if self.verbose:
            print(f"OPD Instance: v={v}, b={b}, r={r}")
            print(f"Solver: IBM CPLEX CP Optimizer (docplex.cp)")
            print(f"Variant: '{variant}' ({'auxiliary variables' if variant == 'aux' else 'direct'})")

    def compute_lower_bound(self):
        """Compute theoretical lower bound on lambda."""
        if self.v <= 1:
            return 0
        numerator = self.r * (self.r * self.v - self.b)
        denominator = self.b * (self.v - 1)
        if denominator == 0:
            return 0
        return max(0, math.ceil(numerator / denominator))

    def build_and_solve(self, timeout=300, stats_queue=None):
        """Build CP model and solve it.

        Symmetry options (self.sym_options list):
          'r'       : Fix row 0 to have 1 in the first r columns and 0 in the rest.
          'lex_row' : Lexicographic ordering between consecutive rows.
          'lex_col' : Lexicographic ordering between consecutive columns.

        Returns (status, matrix, solve_time, lambda_val, applied_sym).
        """
        start_build = time.time()
        mdl = CpoModel(name="OPD")
        v, b, r = self.v, self.b, self.r

        use_fix_r   = 'r'       in self.sym_options
        use_lex_row = 'lex_row' in self.sym_options
        use_lex_col = 'lex_col' in self.sym_options

        # Decision variables: x[i][j] in {0, 1}
        x = [
            [mdl.integer_var(min=0, max=1, name=f"x_{i}_{j}") for j in range(b)]
            for i in range(v)
        ]

        # Constraint: each row sums to r
        for i in range(v):
            mdl.add(mdl.sum(x[i]) == r)

        # ----------------------------------------------------------------
        # Symmetry breaking
        # ----------------------------------------------------------------
        sym_methods = []

        # 'r': Fix row 0 → first r columns = 1, remaining = 0
        if use_fix_r:
            sym_methods.append('r')
            for j in range(r):
                mdl.add(x[0][j] == 1)
            for j in range(r, b):
                mdl.add(x[0][j] == 0)

        # 'lex_row': consecutive rows in lexicographic order
        #   Without fix-r  → ascending : row[i] <=_lex row[i+1]
        #   With    fix-r  → descending: row[i+1] <=_lex row[i]
        #     (because row 0 is fixed as the lexicographically largest row)
        if use_lex_row:
            sym_methods.append('lex_row')
            for i in range(v - 1):
                if use_fix_r:
                    mdl.add(mdl.lexicographic(x[i + 1], x[i]))  # x[i+1] <=_lex x[i]
                else:
                    mdl.add(mdl.lexicographic(x[i], x[i + 1]))  # x[i] <=_lex x[i+1]

        # 'lex_col': consecutive columns in lexicographic order
        cols = [[x[i][j] for i in range(v)] for j in range(b)]
        if use_lex_col:
            sym_methods.append('lex_col')
            for j in range(b - 1):
                if use_fix_r:
                    mdl.add(mdl.lexicographic(cols[j + 1], cols[j]))  # col[j+1] <=_lex col[j]
                else:
                    mdl.add(mdl.lexicographic(cols[j], cols[j + 1]))  # col[j] <=_lex col[j+1]

        applied_sym = "+".join(sym_methods) if sym_methods else "none"
        self.applied_sym = applied_sym

        if self.verbose:
            lb = self.compute_lower_bound()
            print(f"Lower bound on lambda: {lb}")
            print(f"Symmetry breaking: {applied_sym}")

        # ----------------------------------------------------------------
        # Objective: minimise maximum pairwise dot-product (lambda)
        # ----------------------------------------------------------------
        pairs = list(combinations(range(v), 2))

        if self.variant == "aux":
            # Auxiliary variables: s[i][j][k] = x[i][k] * x[j][k]
            s = {}
            for i, j in pairs:
                for k in range(b):
                    s[(i, j, k)] = mdl.integer_var(min=0, max=1, name=f"s_{i}_{j}_{k}")
                    mdl.add(s[(i, j, k)] == x[i][k] * x[j][k])
            dot_products = [
                mdl.sum(s[(i, j, k)] for k in range(b))
                for i, j in pairs
            ]
        else:
            dot_products = [
                mdl.sum(x[i][k] * x[j][k] for k in range(b))
                for i, j in pairs
            ]

        lambda_var = mdl.integer_var(min=0, max=r, name="lambda")
        for dp in dot_products:
            mdl.add(lambda_var >= dp)
        mdl.minimize(lambda_var)

        # Get model statistics (variables and constraints)
        n_vars = len(mdl.get_all_variables())
        n_constrs = len(mdl.get_all_expressions())
        self.n_vars = n_vars
        self.n_constrs = n_constrs

        if self.verbose:
            print(f"Model statistics: {n_vars} variables, {n_constrs} constraints")
            print(f"Solving...")

        # Send stats to main process NOW (before solve), so timeout can still read them
        if stats_queue is not None:
            stats_queue.put({'stats_only': True, 'n_vars': n_vars,
                             'n_constrs': n_constrs, 'applied_sym': applied_sym})

        # Solve
        encoding_end = time.time()
        remaining_timeout = max(0, timeout - (encoding_end - start_build))
        sol = mdl.solve(TimeLimit=remaining_timeout, LogVerbosity="Terse" if self.verbose else "Quiet")
        solve_time = time.time() - start_build

        status = sol.get_solve_status() if sol else "Unknown"

        if sol and status == "Optimal":
            lam = int(sol.get_value(lambda_var))
            matrix = [
                [int(sol.get_value(x[i][j])) for j in range(b)]
                for i in range(v)
            ]
            return 'OPTIMAL', matrix, solve_time, lam, applied_sym, n_vars, n_constrs
        elif sol and status == "Feasible":
            lam = int(sol.get_value(lambda_var))
            matrix = [
                [int(sol.get_value(x[i][j])) for j in range(b)]
                for i in range(v)
            ]
            return 'FEASIBLE', matrix, solve_time, lam, applied_sym, n_vars, n_constrs
        elif sol and "Infeasible" in status:
            return 'UNSAT', None, solve_time, None, applied_sym, n_vars, n_constrs
        else:
            return 'TIMEOUT', None, solve_time, None, applied_sym, n_vars, n_constrs

    def verify_solution(self, matrix, target_lambda):
        """Verify that solution satisfies all constraints."""
        for i in range(self.v):
            row_sum = sum(matrix[i])
            if row_sum != self.r:
                if self.verbose:
                    print(f"Row {i} sum = {row_sum}, expected {self.r}")
                return False, -1

        max_overlap = 0
        for i1, i2 in combinations(range(self.v), 2):
            overlap = sum(matrix[i1][j] * matrix[i2][j] for j in range(self.b))
            max_overlap = max(max_overlap, overlap)
            if overlap > target_lambda:
                if self.verbose:
                    print(f"Overlap({i1},{i2}) = {overlap} > {target_lambda}")
                return False, overlap

        return True, max_overlap

    def print_matrix(self, matrix):
        """Print solution matrix in readable format."""
        print("\nSolution matrix:")
        for i, row in enumerate(matrix):
            row_str = ''.join(str(cell) for cell in row)
            print(f"  Row {i:2d}: {row_str} (sum={sum(row)})")


# ---------------------------------------------------------------------------
# File-level solve function
# ---------------------------------------------------------------------------

def _solve_worker(v, b, r, variant, sym_options, quiet, queue, timeout):
    """Worker function for running the solver in a separate process."""
    try:
        t_start = time.time()
        solver = OpdCPCPLEX(v, b, r, variant=variant, sym_options=sym_options, verbose=not quiet)
        remaining_timeout = max(0, timeout - (time.time() - t_start))
        # Pass queue so stats are sent immediately after model build (before solve)
        status, matrix, solve_time, lam, applied_sym, n_vars, n_constrs = solver.build_and_solve(
            timeout=remaining_timeout, stats_queue=queue
        )
        total_time = time.time() - t_start
        queue.put({
            'status': status,
            'matrix': matrix,
            'lam': lam,
            'applied_sym': applied_sym,
            'n_vars': n_vars,
            'n_constrs': n_constrs,
            'total_time': total_time,
        })
    except Exception as e:
        queue.put({'error': str(e)})


def solve_from_file(filepath, timeout=300, variant="", sym_options=None, quiet=False):
    """Read params from file and solve using multiprocessing for strict timeout."""
    start_time = time.time()  # Start measuring immediately to include parsing time
    sym_options = sym_options or []
    filename = os.path.basename(filepath)
    print(f"\n{'='*60}")
    print(f"Processing file: {filename}")
    print(f"{'='*60}")

    params = parse_input_file(filepath)
    if not params or not all(k in params for k in ('v', 'b', 'r')):
        print("Error: Could not extract v, b, r from file.")
        return {
            'File': filename, 'v': None, 'b': None, 'r': None,
            'Lower Bound': None, 'Optimal Lambda': None,
            'Variables': 0, 'Clauses': 0,
            'Sym Method': 'none', 'Time (s)': 0, 'Status': 'Parse Error'
        }

    v, b, r = params['v'], params['b'], params['r']
    print(f"Parameters: v={v}, b={b}, r={r}")

    # Compute lower bound in main process
    tmp = OpdCPCPLEX.__new__(OpdCPCPLEX)
    tmp.v, tmp.b, tmp.r = v, b, r
    lb = tmp.compute_lower_bound()

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_solve_worker,
        args=(v, b, r, variant, sym_options, quiet, queue, timeout)
    )
    p.start()

    try:
        # Give CPLEX a 10-second grace period to exit cleanly after its internal TimeLimit
        p.join(timeout=timeout + 10)
        elapsed = time.time() - start_time

        if p.is_alive():
            print(f"\nTIMEOUT strictly enforced after {timeout} seconds.")
            p.terminate()
            p.join()
            # Drain queue: prefer full result, fall back to stats_only entry
            full_result = None
            stats_only  = None
            while not queue.empty():
                r_item = queue.get()
                if r_item.get('stats_only'):
                    stats_only = r_item
                elif 'lam' in r_item:
                    full_result = r_item
            result       = full_result or stats_only
            _n_vars    = result.get('n_vars', 0)    if result else 0
            _n_constrs = result.get('n_constrs', 0) if result else 0
            _lam       = full_result['lam'] if full_result else None
            _sym       = result.get('applied_sym', '+'.join(sym_options) if sym_options else 'none') if result else ('+'.join(sym_options) if sym_options else 'none')
            if _lam is not None:
                print(f"Partial Result before TIMEOUT: Optimal lambda = {_lam}")
            print(f"Variables: {_n_vars}, Constraints: {_n_constrs}")
            return {
                'File': filename, 'v': v, 'b': b, 'r': r,
                'Lower Bound': lb,
                'Optimal Lambda': _lam,
                'Variables': _n_vars,
                'Clauses': _n_constrs,
                'Sym Method': _sym,
                'Time (s)': round(elapsed, 3),
                'Status': 'Timeout'
            }
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Đã tiếp nhận yêu cầu dừng từ người dùng. Đang hủy tiến trình...")
        p.terminate()
        p.join()
        sys.exit(1)

    elapsed = time.time() - start_time
    result = None
    stats_only = None
    while not queue.empty():
        res = queue.get()
        if res.get('stats_only'):
            stats_only = res
        elif 'error' in res:
            result = res
            break
        else:
            result = res

    if result is None:
        # Worker crashed before finishing — stats_only may still have model stats
        _n_vars    = stats_only.get('n_vars', 0)    if stats_only else 0
        _n_constrs = stats_only.get('n_constrs', 0) if stats_only else 0
        _sym       = stats_only.get('applied_sym', '+'.join(sym_options) if sym_options else 'none') if stats_only else ('+'.join(sym_options) if sym_options else 'none')
        print(f"Process crashed or returned no result.")
        print(f"Variables: {_n_vars}, Constraints: {_n_constrs}")
        return {
            'File': filename, 'v': v, 'b': b, 'r': r,
            'Lower Bound': lb, 'Optimal Lambda': None,
            'Variables': _n_vars, 'Clauses': _n_constrs,
            'Sym Method': _sym,
            'Time (s)': round(elapsed, 3), 'Status': 'Crashed'
        }

    if 'error' in result:
        # stats_only may have been sent before the exception was raised
        _n_vars    = stats_only.get('n_vars', 0)    if stats_only else 0
        _n_constrs = stats_only.get('n_constrs', 0) if stats_only else 0
        _sym       = stats_only.get('applied_sym', 'none') if stats_only else 'none'
        print(f"Solver error: {result['error']}")
        print(f"Variables: {_n_vars}, Constraints: {_n_constrs}")
        return {
            'File': filename, 'v': v, 'b': b, 'r': r,
            'Lower Bound': lb, 'Optimal Lambda': None,
            'Variables': _n_vars, 'Clauses': _n_constrs,
            'Sym Method': _sym,
            'Time (s)': round(elapsed, 3),
            'Status': f"Error: {result['error']}"
        }

    status   = result['status']
    matrix   = result['matrix']
    lam      = result['lam']
    applied_sym = result['applied_sym']
    n_vars   = result.get('n_vars', 0)
    n_constrs = result.get('n_constrs', 0)
    total_time = elapsed

    if status == 'OPTIMAL':
        solver_for_verify = OpdCPCPLEX.__new__(OpdCPCPLEX)
        solver_for_verify.v, solver_for_verify.b, solver_for_verify.r = v, b, r
        solver_for_verify.verbose = not quiet
        valid, _ = solver_for_verify.verify_solution(matrix, lam)
        if not quiet:
            solver_for_verify.print_matrix(matrix)
        print(f"\nRESULT for {filename}:")
        print(f"  Optimal lambda: {lam}")
        print(f"  Lower bound:    {lb}")
        print(f"  Gap from LB:    {lam - lb}")
        print(f"  Variables:      {n_vars}")
        print(f"  Constraints:    {n_constrs}")
        print(f"  Sym Method:     {applied_sym}")
        print(f"  Total time:     {total_time:.3f}s")
        return {
            'File': filename, 'v': v, 'b': b, 'r': r,
            'Lower Bound': lb,
            'Optimal Lambda': lam,
            'Variables': n_vars,
            'Clauses': n_constrs,
            'Sym Method': applied_sym,
            'Time (s)': round(total_time, 3),
            'Status': 'Solved' if valid else 'Invalid'
        }
    elif status == 'FEASIBLE':
        solver_for_verify = OpdCPCPLEX.__new__(OpdCPCPLEX)
        solver_for_verify.v, solver_for_verify.b, solver_for_verify.r = v, b, r
        solver_for_verify.verbose = not quiet
        valid, _ = solver_for_verify.verify_solution(matrix, lam)
        if not quiet:
            solver_for_verify.print_matrix(matrix)
        print(f"\nRESULT for {filename} (TIMEOUT WITH FEASIBLE SOLUTION):")
        print(f"  Best lambda:    {lam}")
        print(f"  Lower bound:    {lb}")
        print(f"  Gap from LB:    {lam - lb}")
        print(f"  Variables:      {n_vars}")
        print(f"  Constraints:    {n_constrs}")
        print(f"  Sym Method:     {applied_sym}")
        print(f"  Total time:     {total_time:.3f}s")
        return {
            'File': filename, 'v': v, 'b': b, 'r': r,
            'Lower Bound': lb,
            'Optimal Lambda': lam,
            'Variables': n_vars,
            'Clauses': n_constrs,
            'Sym Method': applied_sym,
            'Time (s)': round(total_time, 3),
            'Status': 'Feasible (Timeout)' if valid else 'Invalid'
        }
    elif status == 'UNSAT':
        print(f"\nRESULT for {filename}: UNSAT. Time: {total_time:.3f}s")
        print(f"  Variables:   {n_vars}")
        print(f"  Constraints: {n_constrs}")
        return {
            'File': filename, 'v': v, 'b': b, 'r': r,
            'Lower Bound': lb, 'Optimal Lambda': None,
            'Variables': n_vars, 'Clauses': n_constrs,
            'Sym Method': applied_sym,
            'Time (s)': round(total_time, 3), 'Status': 'UNSAT'
        }
    else:
        print(f"\nRESULT for {filename}: Timeout. Time: {total_time:.3f}s")
        print(f"  Optimal lambda: {lam}")
        print(f"  Variables:      {n_vars}")
        print(f"  Constraints:    {n_constrs}")
        return {
            'File': filename, 'v': v, 'b': b, 'r': r,
            'Lower Bound': lb, 'Optimal Lambda': lam,
            'Variables': n_vars, 'Clauses': n_constrs,
            'Sym Method': applied_sym,
            'Time (s)': round(total_time, 3), 'Status': 'Timeout'
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CP-based OPD solver using IBM CPLEX CP Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Opd_CP_CPLEX.py --input input/small
  python Opd_CP_CPLEX.py --input input/small/small_5.txt
  python Opd_CP_CPLEX.py --input input/small --timeout 300 --variant aux
  python Opd_CP_CPLEX.py --input input/small --sym r lex_row lex_col
  python Opd_CP_CPLEX.py --input input/small --sym r --quiet
        """
    )

    parser.add_argument('--input',   type=str, required=True,
                        help='Input file or directory path')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Solver time limit in seconds (default: 3600)')
    parser.add_argument('--variant', choices=["", "aux"], default="",
                        help="Model variant: '' (direct) or 'aux' (auxiliary) (default: '')")
    parser.add_argument('--sym', nargs='+', default=[],
                        choices=['r', 'lex_row', 'lex_col'],
                        help='Symmetry breaking methods: r (fix row 0), lex_row, lex_col')
    parser.add_argument('--quiet',   action='store_true',
                        help='Suppress verbose solver output')

    args = parser.parse_args()
    input_path = args.input

    # Allow short paths relative to cwd
    if not os.path.exists(input_path):
        candidate = os.path.join('input', input_path)
        if os.path.exists(candidate):
            input_path = candidate

    COLUMNS = ['File', 'v', 'b', 'r', 'Lower Bound', 'Optimal Lambda',
               'Variables', 'Clauses', 'Sym Method', 'Time (s)', 'Status']

    if os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        files = sorted(f for f in os.listdir(input_path) if f.endswith('.txt'))

        if not files:
            print("No .txt files found in directory.")
        else:
            print(f"Found {len(files)} input file(s).")
            try:
                import openpyxl
                has_openpyxl = True
            except ImportError:
                print("Warning: openpyxl not found. Excel export will be disabled.")
                has_openpyxl = False

            folder_name = os.path.basename(os.path.normpath(input_path))
            script_name = os.path.splitext(os.path.basename(__file__))[0]
            output_file = f"result_{folder_name}_{script_name}.xlsx"

            for f in files:
                filepath = os.path.join(input_path, f)
                res = solve_from_file(filepath, args.timeout, args.variant, args.sym, args.quiet)
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
        solve_from_file(input_path, args.timeout, args.variant, args.sym, args.quiet)
    else:
        print(f"Error: Input path '{args.input}' not found.")
