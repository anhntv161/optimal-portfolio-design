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

    def __init__(self, v, b, r, variant="", verbose=True):
        self.v = v
        self.b = b
        self.r = r
        self.variant = variant
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

    def build_and_solve(self, timeout=300):
        """Build CP model and solve it. Returns (status, matrix, solve_time, lambda_val)."""
        mdl = CpoModel(name="OPD")
        v, b, r = self.v, self.b, self.r

        # Decision variables: x[i][j] in {0, 1}
        x = [
            [mdl.integer_var(min=0, max=1, name=f"x_{i}_{j}") for j in range(b)]
            for i in range(v)
        ]

        # Constraint 1: each row sums to r
        for i in range(v):
            mdl.add(mdl.sum(x[i]) == r)

        # Symmetry breaking: lexicographic ordering of consecutive rows
        for i in range(v - 1):
            mdl.add(mdl.lexicographic(x[i], x[i + 1]))

        # Symmetry breaking: lexicographic ordering of consecutive columns
        cols = [[x[i][j] for i in range(v)] for j in range(b)]
        for j in range(b - 1):
            mdl.add(mdl.lexicographic(cols[j], cols[j + 1]))

        # Objective: minimise maximum dot-product (lambda)
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

        if self.verbose:
            lb = self.compute_lower_bound()
            print(f"Lower bound on lambda: {lb}")
            print(f"Solving...")

        start = time.time()
        sol = mdl.solve(TimeLimit=timeout, LogVerbosity="Terse")
        solve_time = time.time() - start

        status = sol.get_solve_status() if sol else "Unknown"

        if sol and status in ("Optimal", "Feasible"):
            lam = int(sol.get_value(lambda_var))
            matrix = [
                [int(sol.get_value(x[i][j])) for j in range(b)]
                for i in range(v)
            ]
            return 'SAT', matrix, solve_time, lam
        elif sol and "Infeasible" in status:
            return 'UNSAT', None, solve_time, None
        else:
            return 'TIMEOUT', None, solve_time, None

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

def solve_from_file(filepath, timeout=300, variant="", quiet=False):
    """Read params from file and solve. Returns dict of results."""
    filename = os.path.basename(filepath)
    print(f"\n{'='*60}")
    print(f"Processing file: {filename}")
    print(f"{'='*60}")

    params = parse_input_file(filepath)
    if not params or not all(k in params for k in ('v', 'b', 'r')):
        print("Error: Could not extract v, b, r from file.")
        return {
            'File': filename, 'v': None, 'b': None, 'r': None,
            'Optimal Lambda': None, 'Time (s)': 0, 'Status': 'Parse Error'
        }

    v, b, r = params['v'], params['b'], params['r']
    print(f"Parameters: v={v}, b={b}, r={r}")

    try:
        solver = OpdCPCPLEX(v, b, r, variant=variant, verbose=not quiet)
        status, matrix, solve_time, lam = solver.build_and_solve(timeout=timeout)

        if status == 'SAT':
            valid, actual_lam = solver.verify_solution(matrix, lam)
            if not quiet:
                solver.print_matrix(matrix)
            print(f"\nRESULT for {filename}:")
            print(f"Optimal lambda: {lam}")
            print(f"Total time: {solve_time:.3f}s")
            print(f"Verified: {valid}")
            return {
                'File': filename, 'v': v, 'b': b, 'r': r,
                'Optimal Lambda': lam,
                'Time (s)': round(solve_time, 3),
                'Status': 'Solved' if valid else 'Invalid'
            }
        elif status == 'UNSAT':
            print(f"\nNo solution found for {filename} (UNSAT). Time: {solve_time:.3f}s")
            return {
                'File': filename, 'v': v, 'b': b, 'r': r,
                'Optimal Lambda': None, 'Time (s)': round(solve_time, 3), 'Status': 'UNSAT'
            }
        else:
            print(f"\nTimeout for {filename}. Time: {solve_time:.3f}s")
            return {
                'File': filename, 'v': v, 'b': b, 'r': r,
                'Optimal Lambda': None, 'Time (s)': round(solve_time, 3), 'Status': 'Timeout'
            }

    except Exception as e:
        print(f"Solver error: {e}")
        return {
            'File': filename, 'v': v, 'b': b, 'r': r,
            'Optimal Lambda': None, 'Time (s)': 0, 'Status': f'Error: {str(e)}'
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
        """
    )

    parser.add_argument('--input',   type=str, required=True,
                        help='Input file or directory path')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Solver time limit in seconds (default: 3600)')
    parser.add_argument('--variant', choices=["", "aux"], default="",
                        help="Model variant: '' (direct) or 'aux' (auxiliary) (default: '')")
    parser.add_argument('--quiet',   action='store_true',
                        help='Suppress verbose solver output')

    args = parser.parse_args()
    input_path = args.input

    # Allow short paths relative to cwd
    if not os.path.exists(input_path):
        candidate = os.path.join('input', input_path)
        if os.path.exists(candidate):
            input_path = candidate

    if os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        files = sorted(f for f in os.listdir(input_path) if f.endswith('.txt'))

        if not files:
            print("No .txt files found in directory.")
        else:
            print(f"Found {len(files)} input file(s).")
            results = []

            try:
                import pandas as pd
            except ImportError:
                print("Warning: pandas not found. Excel export will be disabled.")
                pd = None

            for f in files:
                filepath = os.path.join(input_path, f)
                res = solve_from_file(filepath, args.timeout, args.variant, args.quiet)
                if res:
                    results.append(res)

            if pd and results:
                folder_name = os.path.basename(os.path.normpath(input_path))
                output_file = f"result_{folder_name}_cp_cplex.xlsx"
                df = pd.DataFrame(results)
                df.to_excel(output_file, index=False)
                print(f"\n{'-'*60}")
                print(f"Results exported to {output_file}")
                print(f"{'-'*60}")

    elif os.path.isfile(input_path):
        solve_from_file(input_path, args.timeout, args.variant, args.quiet)
    else:
        print(f"Error: Input path '{args.input}' not found.")
