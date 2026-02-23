"""
SAT-based OPD solver using Cadical195 with Preliminary Phase

The OPD problem:
- Given: v (rows), b (columns), r (row weight)
- Variables: x[i][j] ∈ {0,1} for i in [0..v-1], j in [0..b-1]
- Constraints:
  1. Each row sums to r
  2. Minimize max dot product between all pairs of distinct rows

Key Improvement:
- Preliminary phase to find tighter upper bound
- Narrows search space before main binary search
"""

from pysat.solvers import Cadical195
import argparse
import time
import itertools
import threading
import math
import os
import re
from pysat.card import CardEnc, EncType


class OpdSAT:
    """SAT-based solver for Orthogonal Latin Defect problem with preliminary phase"""
    
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
            print(f"Solver: Cadical195 (PySAT)")
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
    
    def add_overlap_constraints(self, solver, max_overlap):
        """Add constraint: dot product between any two rows <= max_overlap"""
        for i1, i2 in itertools.combinations(range(self.v), 2):
            overlap_vars = []
            for j in range(self.b):
                # y = x[i1][j] AND x[i2][j]
                y = self.var_id
                self.var_id += 1
                overlap_vars.append(y)
                
                # Encode y <=> x[i1][j] AND x[i2][j]
                solver.add_clause([-y, self.x[i1][j]])
                solver.add_clause([-y, self.x[i2][j]])
                solver.add_clause([-self.x[i1][j], -self.x[i2][j], y])
            
            self.add_at_most_k(solver, overlap_vars, max_overlap)
    
    def create_overlap_structure(self, solver):
        """Create overlap AND gates and return overlap vars for each pair"""
        overlap_vars_per_pair = []
        
        for i1, i2 in itertools.combinations(range(self.v), 2):
            overlap_vars = []
            for j in range(self.b):
                y = self.var_id
                self.var_id += 1
                overlap_vars.append(y)
                
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
    
    def solve_with_max_overlap(self, max_overlap, timeout=120):
        """Solve OPD with given max overlap constraint"""
        solver = Cadical195()
        start_time = time.time()
        
        self.add_row_constraints(solver)
        self.add_overlap_constraints(solver, max_overlap)
        self.add_symmetry_breaking(solver)
        
        if self.verbose:
            print(f"Solving with lambda = {max_overlap}...")
        
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
        
        if thread.is_alive():
            if self.verbose:
                print(f"TIMEOUT after {solve_time:.3f}s")
            solver.delete()
            return 'TIMEOUT', None, solve_time
        
        if result[0]:
            model = solver.get_model()
            matrix = self.extract_matrix(model)
            
            valid, actual_overlap = self.verify_solution(matrix, max_overlap)
            
            if valid:
                if self.verbose:
                    print(f"SAT (lambda={actual_overlap}) in {solve_time:.3f}s")
                solver.delete()
                return 'SAT', matrix, solve_time
            else:
                if self.verbose:
                    print(f"Solution verification failed!")
                solver.delete()
                return 'INVALID', matrix, solve_time
        else:
            if self.verbose:
                print(f"UNSAT in {solve_time:.3f}s")
            solver.delete()
            return 'UNSAT', None, solve_time
    
    def preliminary_phase(self, lb, ub, max_iterations=10, timeout=30):
        """
        Preliminary phase to find a tighter upper bound.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"PRELIMINARY PHASE: Finding tighter upper bound")
            print(f"{'='*60}")
        
        best_ub = ub
        best_matrix = None
        total_time = 0
        
        # Strategy: Start from upper bound and work down
        # Try values: ub, ub-1, ub-2, ... with limited iterations
        step_size = max(1, (ub - lb) // max_iterations)
        
        for iteration in range(max_iterations):
            # Calculate test value: gradually decrease from ub
            test_value = ub - (iteration * step_size)
            
            # Don't go below lower bound
            if test_value < lb:
                test_value = lb
            
            # If we already have a better upper bound, don't test higher values
            if test_value >= best_ub:
                test_value = best_ub - 1
            
            if test_value < lb:
                break
            
            if self.verbose:
                print(f"  [Preliminary {iteration+1}/{max_iterations}] Testing lambda = {test_value}...", end=" ")
            
            status, matrix, solve_time = self.solve_with_max_overlap(test_value, timeout=timeout)
            total_time += solve_time
            
            if status == 'SAT':
                best_ub = test_value
                best_matrix = matrix
                if self.verbose:
                    print(f"✓ Found solution! New UB = {best_ub}")
            elif status == 'UNSAT':
                if self.verbose:
                    print(f"✗ UNSAT")
                # If UNSAT, we know the answer is higher
                break
            else:
                if self.verbose:
                    print(f"⏱ Timeout/Invalid")
        
        return best_ub, best_matrix, total_time
    
    def incremental_optimize_linear_descent(self, timeout=120, 
                                            preliminary_iterations=None,
                                            preliminary_timeout=30):
        """
        Incremental SAT Solver using Linear Descent strategy.
        """
        lb = self.compute_lower_bound()
        ub = self.r
        
        if self.verbose:
            print(f"\n{'='*60}\nPRELIMINARY PHASE\n{'='*60}")
            
        # Calculate dynamic iterations if not set
        if preliminary_iterations is None:
            search_space_size = ub - lb
            # Changed logic: 1/4 of the search space
            preliminary_iterations = max(1, search_space_size // 4)
            if self.verbose:
                print(f"Dynamic preliminary iterations: {preliminary_iterations} (1/4 of range [{lb}, {ub}])")
        
        current_best_lambda, best_matrix, time_pre = self.preliminary_phase(
            lb, ub, max_iterations=preliminary_iterations, timeout=preliminary_timeout
        )
        
        ub = current_best_lambda
        
        if self.verbose:
            print(f"\n{'='*60}\nINCREMENTAL MAIN PHASE (Linear Descent)\n{'='*60}")
            print(f"Starting Linear Descent from lambda = {ub}")
            print(f"Lower Bound: {lb}")

        solver = Cadical195()
        start_build = time.time()
        
        self.add_row_constraints(solver)
        overlap_vars_per_pair = self.create_overlap_structure(solver)
        self.add_symmetry_breaking(solver)
        
        build_time = time.time() - start_build
        total_solve_time = 0
        
        current_target = ub
        
        while current_target >= lb:
            if self.verbose:
                print(f"Checking lambda <= {current_target}...", end=" ")
            
            for overlap_vars in overlap_vars_per_pair:
                self.add_at_most_k(solver, overlap_vars, current_target)
            
            start_time = time.time()
            result = [None]
            
            def solve_thread():
                try: result[0] = solver.solve()
                except: result[0] = False
            
            t = threading.Thread(target=solve_thread)
            t.daemon = True
            t.start()
            t.join(timeout=timeout)
            
            solve_time = time.time() - start_time
            total_solve_time += solve_time
            
            if t.is_alive():
                if self.verbose: print(f"TIMEOUT ({solve_time:.2f}s)")
                solver.terminate()
                break
                
            if result[0]: # SAT
                model = solver.get_model()
                matrix = self.extract_matrix(model)
                valid, actual_overlap = self.verify_solution(matrix, current_target)
                
                if valid:
                    if self.verbose: print(f"SAT (Found max_overlap={actual_overlap}) in {solve_time:.2f}s")
                    best_matrix = matrix
                    current_best_lambda = actual_overlap
                    current_target = actual_overlap - 1
                else:
                    if self.verbose: print("Invalid solution (Logic error)")
                    break
            else: # UNSAT
                if self.verbose: print(f"UNSAT in {solve_time:.2f}s. Optimality Proven.")
                break
                
        solver.delete()
        
        total_time = time_pre + build_time + total_solve_time
        return current_best_lambda, best_matrix, total_time


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

def solve_from_file(filepath, timeout=120, preliminary_iterations=None, 
                    preliminary_timeout=30, quiet=False):
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
        optimal_lambda, matrix, total_time = solver.incremental_optimize_linear_descent(
            timeout=timeout,
            preliminary_iterations=preliminary_iterations,
            preliminary_timeout=preliminary_timeout
        )
        
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
        description='SAT-based OPD solver with preliminary phase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Opd_incremental_sat_preliminary.py --input input/small/small_1.txt
  python Opd_incremental_sat_preliminary.py --input input/small
        """
    )
    
    parser.add_argument('--input', type=str, required=True, help='Input file or directory path')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout per SAT call in main search (seconds)')
    parser.add_argument('--preliminary-iterations', type=int, default=None, 
                        help='Number of iterations in preliminary phase (default: 1/4 of search space)')
    parser.add_argument('--preliminary-timeout', type=int, default=30,
                        help='Timeout per SAT call in preliminary phase (seconds)')
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
            except ImportError:
                print("Warning: pandas not found. Excel export will be disabled.")
                pd = None
            
            for f in files:
                filepath = os.path.join(input_path, f)
                res = solve_from_file(
                    filepath, 
                    args.timeout, 
                    args.preliminary_iterations,
                    args.preliminary_timeout,
                    args.quiet
                )
                if res:
                    results.append(res)
            
            if pd and results:
                folder_name = os.path.basename(os.path.normpath(input_path))
                output_file = f"result_{folder_name}_preliminary.xlsx"
                
                df = pd.DataFrame(results)
                df.to_excel(output_file, index=False)
                print(f"\n{'-'*60}")
                print(f"Results exported to {output_file}")
                print(f"{'-'*60}")
                
    elif os.path.isfile(input_path):
        solve_from_file(
            input_path, 
            args.timeout, 
            args.preliminary_iterations,
            args.preliminary_timeout,
            args.quiet
        )
    else:
        print(f"Error: Input path '{args.input}' not found.")
