"""
ILP-based OPD solver using OR-Tools

The OPD problem:
- Given: v (rows), b (columns), r (row weight)
- Variables: x[i][j] ∈ {0,1} for i in [0..v-1], j in [0..b-1]
- Constraints:
  1. Each row sums to r
  2. Minimize max dot product between all pairs of distinct rows
"""

import argparse
import time
import itertools
import math


class ORToolsSolver:
    """OR-Tools ILP-based solver for OPD problem"""
    
    def __init__(self, v, b, r, verbose=True):
        try:
            from ortools.linear_solver import pywraplp
            self.pywraplp = pywraplp
        except ImportError:
            raise ImportError("OR-Tools not installed. Install with: pip install ortools")
        
        self.v = v
        self.b = b
        self.r = r
        self.verbose = verbose
        
        if self.verbose:
            print(f"OPD Instance: v={v}, b={b}, r={r}")
            print(f"Solver: OR-Tools ILP")
    
    def compute_lower_bound(self):
        """Compute theoretical lower bound on lambda"""
        if self.v <= 1:
            return 0
        numerator = self.r * (self.r * self.v - self.b)
        denominator = self.b * (self.v - 1)
        if denominator == 0:
            return 0
        return max(0, math.ceil(numerator / denominator))
    
    def verify_solution(self, matrix, target_max_overlap):
        """Verify that solution satisfies constraints"""
        for i in range(self.v):
            row_sum = sum(matrix[i])
            if row_sum != self.r:
                if self.verbose:
                    print(f"Row {i} sum = {row_sum}, expected {self.r}")
                return False, -1
        
        max_overlap = 0
        for i1, i2 in itertools.combinations(range(self.v), 2):
            overlap = sum(matrix[i1][j] * matrix[i2][j] for j in range(self.b))
            max_overlap = max(max_overlap, overlap)
            if overlap > target_max_overlap:
                if self.verbose:
                    print(f"Overlap({i1},{i2}) = {overlap} > {target_max_overlap}")
                return False, overlap
        
        return True, max_overlap
    
    def print_matrix(self, matrix):
        """Print matrix in readable format"""
        print("\nSolution matrix:")
        for i, row in enumerate(matrix):
            row_str = ''.join(str(x) for x in row)
            print(f"  Row {i:2d}: {row_str} (sum={sum(row)})")
    
    def solve_with_max_overlap(self, max_overlap, timeout=60):
        """Solve with OR-Tools ILP"""
        solver = self.pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            solver = self.pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            raise RuntimeError("ILP solver not available")
        
        start_time = time.time()
        solver.SetTimeLimit(timeout * 1000)
        
        x = {}
        for i in range(self.v):
            for j in range(self.b):
                x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')
        
        for i in range(self.v):
            solver.Add(solver.Sum([x[i, j] for j in range(self.b)]) == self.r)
        
        for i1, i2 in itertools.combinations(range(self.v), 2):
            overlap_vars = []
            for j in range(self.b):
                y = solver.IntVar(0, 1, f'y_{i1}_{i2}_{j}')
                solver.Add(y <= x[i1, j])
                solver.Add(y <= x[i2, j])
                solver.Add(y >= x[i1, j] + x[i2, j] - 1)
                overlap_vars.append(y)
            
            solver.Add(solver.Sum(overlap_vars) <= max_overlap)
        
        for j in range(self.r):
            solver.Add(x[0, j] == 1)
        for j in range(self.r, self.b):
            solver.Add(x[0, j] == 0)
        
        if self.verbose:
            print(f"Solving with lambda = {max_overlap}...")
        
        solver.Minimize(solver.NumVar(0, 0, 'dummy'))
        
        status = solver.Solve()
        solve_time = time.time() - start_time
        
        if status == self.pywraplp.Solver.OPTIMAL or status == self.pywraplp.Solver.FEASIBLE:
            matrix = []
            for i in range(self.v):
                row = [int(round(x[i, j].solution_value())) for j in range(self.b)]
                matrix.append(row)
            
            valid, actual_overlap = self.verify_solution(matrix, max_overlap)
            
            if valid:
                if self.verbose:
                    print(f"SAT (lambda={actual_overlap}) in {solve_time:.3f}s")
                return 'Optimal', matrix, solve_time
            else:
                return 'Invalid', matrix, solve_time
        else:
            if self.verbose:
                print(f"UNSAT in {solve_time:.3f}s")
            return 'Infeasible', None, solve_time
    
    def solve_direct_optimization(self, timeout=300):
        """Solve by directly minimizing max overlap"""
        solver = self.pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            solver = self.pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            raise RuntimeError("ILP solver not available")
        
        start_time = time.time()
        solver.SetTimeLimit(timeout * 1000)
        
        x = {}
        for i in range(self.v):
            for j in range(self.b):
                x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')
        
        for i in range(self.v):
            solver.Add(solver.Sum([x[i, j] for j in range(self.b)]) == self.r)
        
        max_overlap = solver.IntVar(0, self.r, 'max_overlap')
        
        for i1, i2 in itertools.combinations(range(self.v), 2):
            overlap_vars = []
            for j in range(self.b):
                y = solver.IntVar(0, 1, f'y_{i1}_{i2}_{j}')
                solver.Add(y <= x[i1, j])
                solver.Add(y <= x[i2, j])
                solver.Add(y >= x[i1, j] + x[i2, j] - 1)
                overlap_vars.append(y)
            
            solver.Add(solver.Sum(overlap_vars) <= max_overlap)
        
        for j in range(self.r):
            solver.Add(x[0, j] == 1)
        for j in range(self.r, self.b):
            solver.Add(x[0, j] == 0)
        
        solver.Minimize(max_overlap)
        
        if self.verbose:
            print("\nSolving with direct optimization...")
        
        status = solver.Solve()
        solve_time = time.time() - start_time
        
        if status == self.pywraplp.Solver.OPTIMAL or status == self.pywraplp.Solver.FEASIBLE:
            optimal_lambda = int(round(max_overlap.solution_value()))
            
            matrix = []
            for i in range(self.v):
                row = [int(round(x[i, j].solution_value())) for j in range(self.b)]
                matrix.append(row)
            
            if self.verbose:
                print(f"OPTIMAL - Found solution with max overlap = {optimal_lambda}")
                print(f"Solve time: {solve_time:.3f}s")
            
            return optimal_lambda, matrix, solve_time
        else:
            if self.verbose:
                print(f"No solution found in {solve_time:.3f}s")
            return None, None, solve_time
    
    
    def binary_search_optimize(self, timeout=60):
        """Find optimal max overlap using binary search"""
        lb = self.compute_lower_bound()
        ub = self.r
        
        if self.verbose:
            print(f"\nBinary search for optimal lambda in [{lb}, {ub}]")
        
        best_lambda = ub
        best_matrix = None
        total_time = 0
        solve_history = []
        
        while lb <= ub:
            mid = (lb + ub) // 2
            
            status, matrix, solve_time = self.solve_with_max_overlap(mid, timeout)
            total_time += solve_time
            solve_history.append({
                'lambda': mid,
                'status': status,
                'time': solve_time
            })
            
            if status == 'Optimal':
                best_lambda = mid
                best_matrix = matrix
                ub = mid - 1
            else:
                lb = mid + 1
        
        if self.verbose:
            print(f"\nBinary search complete!")
            print(f"Optimal lambda = {best_lambda}")
            print(f"Total time: {total_time:.3f}s")
            print(f"Solver calls: {len(solve_history)}")
        
        return best_lambda, best_matrix, total_time


def parse_input_file(filepath):
    """Parse v, b, r from input file"""
    import re
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

def solve_from_file(filepath, timeout=120, method='binary_search', quiet=False):
    """Read params from file and solve. Returns dict of results."""
    import os
    filename = os.path.basename(filepath)
    print(f"\n{'='*60}")
    print(f"Processing file: {filename}")
    print(f"{'='*60}")
    
    params = parse_input_file(filepath)
    if not params or 'v' not in params or 'b' not in params or 'r' not in params:
        print("Error: Could not extract v, b, r from file.")
        return {
            'File': filename, 'v': None, 'b': None, 'r': None,
            'Optimal Lambda': None, 'Time (s)': 0, 'Status': 'Parse Error'
        }
        
    v, b, r = params['v'], params['b'], params['r']
    print(f"Parameters: v={v}, b={b}, r={r}")
    
    try:
        solver = ORToolsSolver(v, b, r, verbose=not quiet)
        
        if method == 'binary_search':
            optimal_lambda, matrix, total_time = solver.binary_search_optimize(timeout=timeout)
        else:
            optimal_lambda, matrix, total_time = solver.solve_direct_optimization(timeout=timeout)
        
        result_data = {
            'File': filename,
            'v': v, 'b': b, 'r': r,
            'Optimal Lambda': optimal_lambda,
            'Time (s)': round(total_time, 3),
            'Status': 'Solved' if matrix else 'No Solution'
        }
        
        if matrix:
            print(f"\nRESULT for {filename}:")
            print(f"Optimal lambda: {optimal_lambda}")
            print(f"Total time: {total_time:.3f}s")
        else:
            print(f"\nNo solution found for {filename}")
            
        return result_data
            
    except Exception as e:
        print(f"Solver error: {e}")
        return {
            'File': filename, 'v': v, 'b': b, 'r': r,
            'Optimal Lambda': None, 'Time (s)': 0, 'Status': f'Error: {str(e)}'
        }


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(
        description='OR-Tools based OPD solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Opd_ilp.py --input input/small/small_1.txt
  python Opd_ilp.py --input input/small
  python Opd_ilp.py --v 8 --b 28 --r 14 --timeout 120
        """
    )
    
    parser.add_argument('--input', type=str, help='Input file or directory path')
    parser.add_argument('--v', type=int, default=9, help='Number of rows (if input not used)')
    parser.add_argument('--b', type=int, default=36, help='Number of columns (if input not used)')
    parser.add_argument('--r', type=int, default=12, help='Row weight (if input not used)')
    parser.add_argument('--lambda', type=int, default=None, dest='target_lambda',
                        help='Target max overlap (if not set, will optimize)')
    parser.add_argument('--timeout', type=int, default=900, 
                        help='Timeout per solver call (seconds)')
    parser.add_argument('--method', type=str, default='binary_search',
                        choices=['binary_search', 'direct'],
                        help='Optimization method: binary_search or direct (default: binary_search)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    if args.input:
        input_path = args.input
        
        if not os.path.exists(input_path):
            potential_path = os.path.join('input', input_path)
            if os.path.exists(potential_path):
                input_path = potential_path
        
        if os.path.isdir(input_path):
            print(f"Processing directory: {input_path}")
            files = [f for f in os.listdir(input_path) if f.endswith('.txt')]
            
            if not files:
                print("No .txt files found in directory.")
            else:
                print(f"Found {len(files)} input files.")
                results = []
                
                try:
                    import pandas as pd
                except ImportError:
                    print("Warning: pandas not found. Excel export will be disabled.")
                    print("Install with: pip install pandas openpyxl")
                    pd = None
                
                for f in files:
                    filepath = os.path.join(input_path, f)
                    res = solve_from_file(filepath, args.timeout, args.method, args.quiet)
                    if res:
                        results.append(res)
                
                if pd and results:
                    folder_name = os.path.basename(os.path.normpath(input_path))
                    output_file = f"result_{folder_name}_ilp.xlsx"
                    
                    df = pd.DataFrame(results)
                    df.to_excel(output_file, index=False)
                    print(f"\n{'-'*60}")
                    print(f"Results exported to {output_file}")
                    print(f"{'-'*60}")

        elif os.path.isfile(input_path):
            solve_from_file(input_path, args.timeout, args.method, args.quiet)
        else:
            print(f"Error: Input path '{args.input}' not found.")
            
    else:
        print("=" * 60)
        print("OR-Tools based OPD Solver")
        print("=" * 60)
        
        try:
            solver = ORToolsSolver(args.v, args.b, args.r, verbose=not args.quiet)
        except ImportError as e:
            print(f"\nError: {e}")
            print("\nInstall OR-Tools with: pip install ortools")
            exit(1)
        
        if args.target_lambda is not None:
            print(f"\nTarget: Find solution with lambda <= {args.target_lambda}")
            status, matrix, solve_time = solver.solve_with_max_overlap(
                args.target_lambda, args.timeout
            )
            
            if status == 'Optimal' and matrix is not None:
                solver.print_matrix(matrix)
                valid, actual = solver.verify_solution(matrix, args.target_lambda)
                print(f"\nVerification: valid={valid}, actual_max_overlap={actual}")
        else:
            print(f"\nObjective: Minimize maximum overlap (lambda)")
            
            if args.method == 'binary_search':
                optimal_lambda, matrix, total_time = solver.binary_search_optimize(
                    timeout=args.timeout
                )
            else:
                optimal_lambda, matrix, total_time = solver.solve_direct_optimization(
                    timeout=args.timeout
                )
            
            if matrix is not None:
                solver.print_matrix(matrix)
                
                valid, actual = solver.verify_solution(matrix, optimal_lambda)
                print(f"\n{'='*60}")
                print(f"FINAL RESULT")
                print(f"{'='*60}")
                print(f"Optimal lambda: {optimal_lambda}")
                print(f"Total time: {total_time:.3f}s")
                print(f"Verification: valid={valid}, actual_max_overlap={actual}")
                print(f"{'='*60}")
            else:
                print("\nNo solution found")
