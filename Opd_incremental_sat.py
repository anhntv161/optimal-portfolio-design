"""
SAT-based OPD solver using Cadical153 (Incremental)

The OPD problem:
- Given: v (rows), b (columns), r (row weight)
- Variables: x[i][j] ∈ {0,1} for i in [0..v-1], j in [0..b-1]
- Constraints:
  1. Each row sums to r
  2. Minimize max dot product between all pairs of distinct rows
"""

from pysat.solvers import Cadical153
import argparse
import time
import itertools
import threading
import math
import os
import re
from pysat.card import CardEnc, EncType


class OpdSAT:
    """SAT-based solver for Orthogonal Latin Defect problem"""
    
    def __init__(self, v, b, r, verbose=True):
        self.v = v
        self.b = b
        self.r = r
        self.verbose = verbose
        
        # Create variable indices for x[i][j]
        self.var_id = 1
        self.x = []
        for i in range(v):
            row = []
            for j in range(b):
                row.append(self.var_id)
                self.var_id += 1
            self.x.append(row)
        
        if self.verbose:
            print(f"OPD Instance: v={v}, b={b}, r={r}")
            print(f"Solver: Cadical153 (PySAT)")
            print(f"Total variables: {self.var_id - 1}")

    def get_var(self, i, j):
        """Map x[i][j] to variable ID (1-based)"""
        return (i * self.b) + j + 1
    
    def compute_lower_bound(self):
        """Compute theoretical lower bound on lambda"""
        if self.v <= 1:
            return 0
        numerator = self.r * (self.r * self.v - self.b)
        denominator = self.b * (self.v - 1)
        if denominator == 0:
            return 0
        return max(0, math.ceil(numerator / denominator))
    
    def add_exactly_r(self, solver, vars):
        """Add constraint: exactly r variables must be true"""
        cnf = CardEnc.equals(lits=vars, bound=self.r, top_id=self.var_id, encoding=EncType.totalizer)
        solver.append_formula(cnf)
        self.var_id = cnf.nv
    
    def add_at_most_k(self, solver, vars, k):
        """Add constraint: at most k variables can be true"""
        if k >= len(vars):
            return
        
        cnf = CardEnc.atmost(lits=vars, bound=k, top_id=self.var_id, encoding=EncType.totalizer)
        solver.append_formula(cnf)
        self.var_id = cnf.nv
    
    def add_row_constraints(self, solver):
        """Add constraint: each row sums to r"""
        for i in range(self.v):
            self.add_exactly_r(solver, self.x[i])
    
    def create_overlap_structure(self, solver):
        """Create overlap AND gates and return overlap vars for each pair"""
        overlap_vars_per_pair = []
        
        for i1, i2 in itertools.combinations(range(self.v), 2):
            overlap_vars = []
            for j in range(self.b):
                y = self.var_id
                self.var_id += 1
                overlap_vars.append(y)
                
                # Encode y <=> x[i1][j] AND x[i2][j]
                solver.add_clause([-y, self.x[i1][j]])
                solver.add_clause([-y, self.x[i2][j]])
                solver.add_clause([-self.x[i1][j], -self.x[i2][j], y])
            
            overlap_vars_per_pair.append(overlap_vars)
        
        return overlap_vars_per_pair
    
    def add_symmetry_breaking(self, solver):
        """Add symmetry breaking constraints"""
        for j in range(self.r):
            solver.add_clause([self.x[0][j]])
        
        for j in range(self.r, self.b):
            solver.add_clause([-self.x[0][j]])

        for i in range(1, self.v - 1):
            self.add_lex_constraints(solver, i, i+1)
            
        if self.verbose:
            print("Added symmetry breaking")

    def add_lex_constraints(self, solver, row_a, row_b):
        """
        Row a >= Row b
        """
        u = self.get_var(row_a, 0)
        v = self.get_var(row_b, 0)
        solver.add_clause([-v, u])
    
    def extract_matrix(self, model):
        """Extract solution matrix from SAT model"""
        model_set = set(model)
        matrix = []
        for i in range(self.v):
            row = []
            for j in range(self.b):
                row.append(1 if self.x[i][j] in model_set else 0)
            matrix.append(row)
        return matrix
    
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
    
    def incremental_binary_search_optimize(self, timeout=120):
        """Find optimal max overlap using incremental SAT with binary search"""
        lb = self.compute_lower_bound()
        ub = self.r
        
        if self.verbose:
            print(f"\n[INCREMENTAL] Binary search for optimal lambda in [{lb}, {ub}]")
        
        solver = Cadical153()
        start_build = time.time()
        
        self.add_row_constraints(solver)
        overlap_vars_per_pair = self.create_overlap_structure(solver)
        self.add_symmetry_breaking(solver)
        
        build_time = time.time() - start_build
        if self.verbose:
            print(f"Built base solver in {build_time:.3f}s")
            print(f"  Overlap pairs: {len(overlap_vars_per_pair)}")
        
        best_lambda = ub
        best_matrix = None
        total_solve_time = 0
        iteration = 0
        
        saved_var_id = self.var_id
        
        while lb <= ub:
            iteration += 1
            mid = (lb + ub) // 2
            
            self.var_id = saved_var_id
            
            for overlap_vars in overlap_vars_per_pair:
                self.add_at_most_k(solver, overlap_vars, mid)
            
            saved_var_id = self.var_id
            
            if self.verbose:
                print(f"  [{iteration}] Testing lambda = {mid}...", end=" ")
            
            start_time = time.time()
            result = [None]
            
            def solve_thread():
                try:
                    result[0] = solver.solve()
                except:
                    result[0] = False
            
            thread = threading.Thread(target=solve_thread)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout)
            
            solve_time = time.time() - start_time
            total_solve_time += solve_time
            
            if thread.is_alive():
                if self.verbose:
                    print(f"TIMEOUT after {solve_time:.3f}s")
                solver.delete()
                return best_lambda, best_matrix, build_time + total_solve_time
            
            if result[0]:
                model = solver.get_model()
                matrix = self.extract_matrix(model)
                
                valid, actual_overlap = self.verify_solution(matrix, mid)
                
                if valid:
                    if self.verbose:
                        print(f"SAT (lambda={actual_overlap}) in {solve_time:.3f}s")
                    best_lambda = mid
                    best_matrix = matrix
                    ub = mid - 1
                else:
                    if self.verbose:
                        print(f"Invalid solution!")
                    lb = mid + 1
            else:
                if self.verbose:
                    print(f"UNSAT in {solve_time:.3f}s")
                lb = mid + 1
        
        solver.delete()
        total_time = build_time + total_solve_time
        
        if self.verbose:
            print(f"\nIncremental binary search complete!")
            print(f"  Optimal lambda: {best_lambda}")
            print(f"  Build time: {build_time:.3f}s")
            print(f"  Solve time: {total_solve_time:.3f}s")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Iterations: {iteration}")
        
        return best_lambda, best_matrix, total_time


def parse_input_file(filepath):
    """Parse v, b, r from input file"""
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

def solve_from_file(filepath, timeout=120, quiet=False):
    """Read params from file and solve. Returns dict of results."""
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
        solver = OpdSAT(v, b, r, verbose=not quiet)
        optimal_lambda, matrix, total_time = solver.incremental_binary_search_optimize(timeout=timeout)
        
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
    parser = argparse.ArgumentParser(
        description='SAT-based OPD solver (Cleaned)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Opd_incremental_sat.py --input input/small/small_1.txt
  python Opd_incremental_sat.py --input input/small
        """
    )
    
    parser.add_argument('--input', type=str, required=True, help='Input file or directory path')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout per SAT call (seconds)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
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
                # Removed openpyxl check since pandas is mainly used, but keeping it simple
            except ImportError:
                print("Warning: pandas not found. Excel export will be disabled.")
                pd = None
            
            for f in files:
                filepath = os.path.join(input_path, f)
                res = solve_from_file(filepath, args.timeout, args.quiet)
                if res:
                    results.append(res)
            
            if pd and results:
                folder_name = os.path.basename(os.path.normpath(input_path))
                output_file = f"result_{folder_name}_incremental.xlsx"
                
                df = pd.DataFrame(results)
                df.to_excel(output_file, index=False)
                print(f"\n{'-'*60}")
                print(f"Results exported to {output_file}")
                print(f"{'-'*60}")
                
    elif os.path.isfile(input_path):
        solve_from_file(input_path, args.timeout, args.quiet)
    else:
        print(f"Error: Input path '{args.input}' not found.")
