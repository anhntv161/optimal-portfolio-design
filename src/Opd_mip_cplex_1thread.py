"""
MIP-based Portfolio Optimization solver using IBM CPLEX (docplex.mp).
Single-threaded version: cplex.parameters.threads = 1 (no parallelism)

The Portfolio Optimization problem:
- Given: v (sub-pools), b (credits), r (sub-pool size)
- Variables: x[i][j] ∈ {0,1} for i in [0..v-1], j in [0..b-1]
- Constraints:
  1. Each row (sub-pool) sums to r
  2. Minimize max overlap between all pairs of distinct rows

Usage:
    python Opd_mip_cplex_1thread.py --input input/small
    python Opd_mip_cplex_1thread.py --input input/small/small_5.txt
    python Opd_mip_cplex_1thread.py --input input/small --timeout 300 --sym lex_row lex_col

Requirements:
    pip install docplex
    IBM CPLEX Optimization Studio must be installed and on the system PATH.
"""

import argparse
import signal
import time
import math
import os
import re
import numpy as np

from docplex.mp.model import Model


# ---------------------------------------------------------------------------
# Hard wall-clock timeout (Linux/macOS via SIGALRM)
# ---------------------------------------------------------------------------

class _HardTimeout(Exception):
    """Raised by SIGALRM when the wall-clock deadline is exceeded."""
    pass


def _alarm_handler(signum, frame):
    raise _HardTimeout("Wall-clock timeout exceeded")


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


# ---------------------------------------------------------------------------
# MIP Solver class (CPLEX via docplex.mp) – single-threaded
# ---------------------------------------------------------------------------

class PortfolioMIPCPLEX:
    """MIP-based solver for Portfolio Optimization problem using IBM CPLEX (single-threaded)."""

    def __init__(self, v, b, r, sym_options=None, verbose=True):
        self.v = v
        self.b = b
        self.r = r
        self.sym_options = sym_options or []
        self.verbose = verbose

        self.lambda_lb = self._compute_lower_bound()

        if self.verbose:
            print(f"Portfolio Optimization Instance: v={v}, b={b}, r={r}")
            print(f"Solver: IBM CPLEX MIP (docplex.mp, single-threaded)")
            print(f"Lower bound on lambda: {self.lambda_lb}")

    def _compute_lower_bound(self):
        """
        Compute theoretical lower bound on lambda:
        λ ≥ ⌈r(rv - b) / (b(v-1))⌉ ∧ λ ≥ 0
        """
        if self.v <= 1:
            return 0
        numerator = self.r * (self.r * self.v - self.b)
        denominator = self.b * (self.v - 1)
        if denominator == 0:
            return 0
        return max(0, math.ceil(numerator / denominator))

    def build_model(self):
        """Build the MIP model using docplex.mp."""

        self.mdl = Model(name="portfolio_optimization")

        # Suppress CPLEX output if not verbose
        if not self.verbose:
            self.mdl.set_log_output(None)
            self.mdl.context.solver.log_output = False
            self.mdl.context.cplex_parameters.mip.display = 0
            self.mdl.context.cplex_parameters.simplex.display = 0

        v, b, r = self.v, self.b, self.r

        # Decision variables: x[i,j] ∈ {0,1}
        self.x = {
            (i, j): self.mdl.binary_var(name=f"x_{i}_{j}")
            for i in range(v) for j in range(b)
        }

        # Lambda variable: maximum overlap (integer)
        self.lambda_var = self.mdl.integer_var(
            lb=self.lambda_lb, ub=r, name="lambda"
        )

        # Constraint: each sub-pool has exactly r credits
        for i in range(v):
            self.mdl.add_constraint(
                self.mdl.sum(self.x[i, j] for j in range(b)) == r,
                ctname=f"subpool_size_{i}"
            )

        # Linearized pairwise overlap constraints:
        # introduce y[i1,i2,j] = x[i1,j] AND x[i2,j]
        self.y = {}
        for i1 in range(v):
            for i2 in range(i1 + 1, v):
                for j in range(b):
                    yv = self.mdl.binary_var(name=f"y_{i1}_{i2}_{j}")
                    self.y[i1, i2, j] = yv
                    # y <= x[i1,j]
                    self.mdl.add_constraint(yv <= self.x[i1, j])
                    # y <= x[i2,j]
                    self.mdl.add_constraint(yv <= self.x[i2, j])
                    # y >= x[i1,j] + x[i2,j] - 1
                    self.mdl.add_constraint(yv >= self.x[i1, j] + self.x[i2, j] - 1)

                # Overlap constraint: sum(y) <= lambda
                overlap_expr = self.mdl.sum(self.y[i1, i2, j] for j in range(b))
                self.mdl.add_constraint(
                    overlap_expr <= self.lambda_var,
                    ctname=f"overlap_{i1}_{i2}"
                )

        # Symmetry breaking
        self._add_symmetry_breaking()

        # Objective: minimize lambda
        self.mdl.minimize(self.lambda_var)

        # Statistics
        self.num_vars = self.mdl.number_of_variables
        self.num_constrs = self.mdl.number_of_constraints

        if self.verbose:
            print(f"\nModel statistics:")
            print(f"  Variables:   {self.num_vars}")
            print(f"  Constraints: {self.num_constrs}")

    # ------------------------------------------------------------------
    # Symmetry breaking helpers
    # ------------------------------------------------------------------

    def _add_lex_le(self, a_vars, b_vars, n, prefix):
        """
        Add linearized lex-order constraint: a <=_lex b
        (binary vectors of length n via prefix-equality propagation).

        p[0] = 1 (Python constant — empty prefix is always equal)
        eq_k  = 1 iff a[k] == b[k]
        p[k+1] = p[k] AND eq_k
        p[k] => a[k] <= b[k]
        """
        p = {0: 1}  # Python int constant

        for k in range(n):
            p_k_is_const = isinstance(p[k], int)

            if k < n - 1:
                eq_k = self.mdl.binary_var(name=f"{prefix}_eq_{k}")
                # eq_k = 1 iff a[k] == b[k]
                self.mdl.add_constraint(a_vars[k] - b_vars[k] <= 1 - eq_k,
                                        ctname=f"{prefix}_eq1_{k}")
                self.mdl.add_constraint(b_vars[k] - a_vars[k] <= 1 - eq_k,
                                        ctname=f"{prefix}_eq2_{k}")
                # Force eq_k=1 when both 1 or both 0
                self.mdl.add_constraint(eq_k >= a_vars[k] + b_vars[k] - 1,
                                        ctname=f"{prefix}_eq3_{k}")
                self.mdl.add_constraint(eq_k >= 1 - a_vars[k] - b_vars[k],
                                        ctname=f"{prefix}_eq4_{k}")

                # p[k+1] = p[k] AND eq_k
                if p_k_is_const:
                    # p[k]=1 constant => p[k+1] = eq_k
                    p[k + 1] = eq_k
                else:
                    pk1 = self.mdl.binary_var(name=f"{prefix}_p_{k+1}")
                    self.mdl.add_constraint(pk1 <= p[k],              ctname=f"{prefix}_p1_{k+1}")
                    self.mdl.add_constraint(pk1 <= eq_k,              ctname=f"{prefix}_p2_{k+1}")
                    self.mdl.add_constraint(pk1 >= p[k] + eq_k - 1,  ctname=f"{prefix}_p3_{k+1}")
                    p[k + 1] = pk1

            # Core lex constraint: p[k] => a[k] <= b[k]
            if p_k_is_const:
                # unconditional
                self.mdl.add_constraint(a_vars[k] <= b_vars[k], ctname=f"{prefix}_lex_{k}")
            else:
                # a[k] - b[k] <= 1 - p[k]
                self.mdl.add_constraint(a_vars[k] - b_vars[k] <= 1 - p[k],
                                        ctname=f"{prefix}_lex_{k}")

    def _add_symmetry_breaking(self):
        """
        Add symmetry breaking constraints based on sym_options:
          - r:       Fix row 0 to have 1 in first r columns
          - lex_row: Lexicographic ordering of consecutive rows
          - lex_col: Lexicographic ordering of consecutive columns
        """
        use_lex_row = 'lex_row' in self.sym_options
        use_lex_col = 'lex_col' in self.sym_options
        fixed_r     = 'r'       in self.sym_options

        sym_methods = []

        if fixed_r:
            sym_methods.append('r')
            for j in range(self.r):
                self.mdl.add_constraint(self.x[0, j] == 1, ctname=f"fix_row0_one_{j}")
            for j in range(self.r, self.b):
                self.mdl.add_constraint(self.x[0, j] == 0, ctname=f"fix_row0_zero_{j}")

        if use_lex_row:
            sym_methods.append('lex_row')
            for i in range(self.v - 1):
                a_row = [self.x[i,     j] for j in range(self.b)]
                b_row = [self.x[i + 1, j] for j in range(self.b)]
                if fixed_r:
                    self._add_lex_le(b_row, a_row, self.b, f"row_lex_{i}")
                else:
                    self._add_lex_le(a_row, b_row, self.b, f"row_lex_{i}")

        if use_lex_col:
            sym_methods.append('lex_col')
            for j in range(self.b - 1):
                a_col = [self.x[i, j]     for i in range(self.v)]
                b_col = [self.x[i, j + 1] for i in range(self.v)]
                if fixed_r:
                    self._add_lex_le(b_col, a_col, self.v, f"col_lex_{j}")
                else:
                    self._add_lex_le(a_col, b_col, self.v, f"col_lex_{j}")

        self.applied_sym = "+".join(sym_methods) if sym_methods else "none"

        if self.verbose:
            print(f"Symmetry breaking: {self.applied_sym}")

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, time_limit=3600):
        """Solve the MIP model (single-threaded: threads = 1). Returns result dict."""

        self.mdl.set_time_limit(time_limit)
        # Require proven optimality (MIP gap = 0)
        self.mdl.parameters.mip.tolerances.mipgap = 0.0
        # Run CPLEX heuristics more frequently to find incumbents faster
        self.mdl.parameters.mip.strategy.heuristicfreq = 10

        # ── Single-threaded mode ──────────────────────────────────────
        self.mdl.parameters.threads = 1   # Use only 1 worker thread
        # ─────────────────────────────────────────────────────────────

        if self.verbose:
            print(f"\nSolving MIP model (threads=1, single-threaded)...")

        start_time = time.time()
        sol = self.mdl.solve()
        solve_time = time.time() - start_time

        return self._extract_solution(sol, solve_time)

    def _extract_solution(self, sol, solve_time):
        """Extract and return solution information."""

        def _base_dict():
            return {
                'time':        solve_time,
                'n_vars':      self.num_vars,
                'n_constrs':   self.num_constrs,
                'applied_sym': self.applied_sym,
                'gap':         None,
                'nodes':       None,
            }

        if sol is None:
            # Could be timeout or infeasible
            solve_details = self.mdl.get_solve_details()
            status_str = solve_details.status if solve_details else "Unknown"
            if self.verbose:
                print(f"\nNo solution found. Status: {status_str}")
            result = _base_dict()
            result.update({'status': 'TIMEOUT', 'lambda': None, 'matrix': None, 'valid': False})
            return result

        solve_details = self.mdl.get_solve_details()
        status_str    = solve_details.status if solve_details else "Unknown"
        mip_gap       = solve_details.mip_relative_gap if solve_details else None
        n_nodes       = int(solve_details.nb_nodes_processed) if solve_details else 0

        # Extract matrix
        matrix = np.zeros((self.v, self.b), dtype=int)
        for i in range(self.v):
            for j in range(self.b):
                matrix[i, j] = int(round(sol.get_value(self.x[i, j])))

        optimal_lambda = int(round(sol.get_value(self.lambda_var)))

        valid, actual_lambda = self._verify_solution(matrix, optimal_lambda)

        is_optimal = (mip_gap is not None and mip_gap < 1e-9)
        out_status  = 'OPTIMAL' if is_optimal else 'FEASIBLE'

        if self.verbose:
            print(f"\nSOLVED in {solve_time:.3f}s")
            print(f"Status: {status_str}")
            print(f"Optimal lambda: {optimal_lambda}")
            if mip_gap is not None:
                print(f"MIP Gap: {mip_gap:.6f}")
            print(f"Nodes explored: {n_nodes}")

        result = _base_dict()
        result.update({
            'status': out_status,
            'lambda': optimal_lambda,
            'matrix': matrix,
            'valid':  valid,
            'gap':    mip_gap,
            'nodes':  n_nodes,
        })
        return result

    # ------------------------------------------------------------------
    # Verification & display
    # ------------------------------------------------------------------

    def _verify_solution(self, matrix, target_lambda):
        """Verify that solution satisfies all constraints."""
        for i in range(self.v):
            row_sum = int(np.sum(matrix[i]))
            if row_sum != self.r:
                if self.verbose:
                    print(f"WARNING: Row {i} sum = {row_sum}, expected {self.r}")
                return False, -1

        max_overlap = 0
        for i1 in range(self.v):
            for i2 in range(i1 + 1, self.v):
                overlap = int(np.sum(matrix[i1] * matrix[i2]))
                max_overlap = max(max_overlap, overlap)
                if overlap > target_lambda:
                    if self.verbose:
                        print(f"WARNING: Overlap({i1},{i2}) = {overlap} > {target_lambda}")
                    return False, overlap

        return True, max_overlap

    def print_matrix(self, matrix):
        """Print solution matrix in readable format."""
        print("\nSolution matrix:")
        for i in range(self.v):
            row_str = ''.join(str(int(c)) for c in matrix[i])
            row_sum = int(np.sum(matrix[i]))
            print(f"  Row {i:2d}: {row_str} (sum={row_sum})")


# ---------------------------------------------------------------------------
# File-level solve function
# ---------------------------------------------------------------------------

def solve_from_file(filepath, sym_options=None, timeout=3600, quiet=False):
    """Read params from file and solve. Returns dict of results.

    A hard SIGALRM deadline is set for exactly `timeout` wall-clock seconds.
    This guarantees that the total time (encoding + solving) never exceeds
    `timeout`, regardless of CPLEX's internal time counter behaviour.
    """
    start_time = time.time()

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
            'Variables': 0, 'Clauses': 0, 'Sym Method': 'none',
            'Time (s)': 0, 'MIP Gap': None, 'Nodes': None,
            'Status': 'Parse Error'
        }

    v, b, r = params['v'], params['b'], params['r']
    print(f"Parameters: v={v}, b={b}, r={r}")

    # ------------------------------------------------------------------
    # Install hard wall-clock deadline via SIGALRM (Linux/macOS only).
    # This fires after exactly `timeout` seconds from now, interrupting
    # CPLEX even if its internal time counter disagrees.
    # ------------------------------------------------------------------
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)  # set hard deadline

    try:
        solver = PortfolioMIPCPLEX(v, b, r, sym_options=sym_options, verbose=not quiet)
        solver.build_model()

        encoding_time     = time.time() - start_time
        remaining_timeout = max(1, timeout - encoding_time)  # remaining wall-clock seconds
        result = solver.solve(time_limit=remaining_timeout)
        total_time = time.time() - start_time

        result_data = {
            'File':           filename,
            'v':              v,
            'b':              b,
            'r':              r,
            'Lower Bound':    solver.lambda_lb,
            'Optimal Lambda': result['lambda'],
            'Variables':      result.get('n_vars', 0),
            'Clauses':        result.get('n_constrs', 0),
            'Sym Method':     result.get('applied_sym', 'none'),
            'Time (s)':       round(total_time, 3),
            'MIP Gap':        round(result['gap'], 6) if result['gap'] is not None else None,
            'Nodes':          result.get('nodes'),
            'Status':         result['status'],
        }

        if result['status'] in ('OPTIMAL', 'FEASIBLE', 'TIMEOUT_WITH_SOLUTION') and result['matrix'] is not None:
            print(f"\nRESULT for {filename}:")
            print(f"Optimal lambda: {result['lambda']}")
            print(f"Lower bound:    {solver.lambda_lb}")
            print(f"Gap from lower bound: {result['lambda'] - solver.lambda_lb}")
            print(f"Total time: {total_time:.3f}s")
            print(f"Nodes explored: {result.get('nodes')}")
            if not quiet:
                solver.print_matrix(result['matrix'])
        else:
            print(f"\nNo solution found for {filename}")
            print(f"Status: {result['status']}")

        return result_data

    except _HardTimeout:
        # SIGALRM fired: total wall-clock time has reached the limit.
        total_time = time.time() - start_time
        print(f"\nHARD TIMEOUT: {total_time:.1f}s elapsed (limit={timeout}s)")
        print(f"No solution returned for {filename}")
        n_vars    = getattr(solver, 'num_vars',    0) if 'solver' in dir() else 0
        n_constrs = getattr(solver, 'num_constrs', 0) if 'solver' in dir() else 0
        lb        = getattr(solver, 'lambda_lb',   None) if 'solver' in dir() else None
        sym       = getattr(solver, 'applied_sym', 'none') if 'solver' in dir() else 'none'
        return {
            'File':           filename,
            'v':              v,
            'b':              b,
            'r':              r,
            'Lower Bound':    lb,
            'Optimal Lambda': None,
            'Variables':      n_vars,
            'Clauses':        n_constrs,
            'Sym Method':     sym,
            'Time (s)':       round(total_time, 3),
            'MIP Gap':        None,
            'Nodes':          None,
            'Status':         'TIMEOUT',
        }

    except Exception as e:
        import traceback
        print(f"Solver error: {e}")
        traceback.print_exc()
        return {
            'File':           filename,
            'v':              v,
            'b':              b,
            'r':              r,
            'Lower Bound':    None,
            'Optimal Lambda': None,
            'Variables':      0,
            'Clauses':        0,
            'Sym Method':     'none',
            'Time (s)':       round(time.time() - start_time, 3),
            'MIP Gap':        None,
            'Nodes':          None,
            'Status':         f'Error: {str(e)}'
        }

    finally:
        # Always cancel the alarm and restore the previous SIGALRM handler.
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='MIP-based Portfolio Optimization solver using IBM CPLEX (single-threaded, threads=1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Opd_mip_cplex_1thread.py --input input/small
  python Opd_mip_cplex_1thread.py --input input/small/small_5.txt --timeout 300
  python Opd_mip_cplex_1thread.py --input input/small --sym lex_row lex_col
  python Opd_mip_cplex_1thread.py --input input/small --sym r lex_row lex_col --quiet
        """
    )

    parser.add_argument('--input',   type=str, required=True,
                        help='Input file or directory path')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout per instance in seconds (default: 3600)')
    parser.add_argument('--sym', nargs='+', default=[],
                        choices=['r', 'lex_row', 'lex_col'],
                        help='Symmetry breaking methods: r, lex_row, lex_col')
    parser.add_argument('--quiet',   action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()
    input_path = args.input

    # Allow short paths relative to cwd
    if not os.path.exists(input_path):
        candidate = os.path.join('input', input_path)
        if os.path.exists(candidate):
            input_path = candidate

    COLUMNS = [
        'File', 'v', 'b', 'r',
        'Lower Bound', 'Optimal Lambda',
        'Variables', 'Clauses', 'Sym Method',
        'Time (s)', 'MIP Gap', 'Nodes', 'Status'
    ]

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
                print("Warning: openpyxl not found. Excel export disabled.")
                has_openpyxl = False

            folder_name = os.path.basename(os.path.normpath(input_path))
            script_name = os.path.splitext(os.path.basename(__file__))[0]
            output_file = f"result_{folder_name}_{script_name}.xlsx"

            for f in files:
                filepath = os.path.join(input_path, f)
                res = solve_from_file(filepath, args.sym, args.timeout, args.quiet)
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
        solve_from_file(input_path, args.sym, args.timeout, args.quiet)

    else:
        print(f"Error: Input path '{args.input}' not found.")
