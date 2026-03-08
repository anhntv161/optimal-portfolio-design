"""
SAT-based OPD solver using Cadical195

The OPD problem:
- Given: v (rows), b (columns), r (row weight)
- Variables: x[i][j] ∈ {0,1} for i in [0..v-1], j in [0..b-1]
- Constraints:
  1. Each row sums to r
  2. Minimize max dot product between all pairs of distinct rows
"""

from pysat.solvers import Glucose4
import argparse
import time
import itertools
import threading
import multiprocessing
import sys
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
            print(f"Solver: Glucose4 (PySAT)")
            print(f"Total variables: {self.var_id - 1}")
    
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
        cnf = CardEnc.equals(lits=vars, bound=self.r, top_id=self.var_id, encoding=EncType.sortnetwrk)
        solver.append_formula(cnf)
        self.var_id = cnf.nv
    
    def add_at_most_k(self, solver, vars, k):
        """Add constraint: at most k variables can be true"""
        if k >= len(vars):
            return
        
        cnf = CardEnc.atmost(lits=vars, bound=k, top_id=self.var_id, encoding=EncType.sortnetwrk)
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
                y = self.var_id
                self.var_id += 1
                overlap_vars.append(y)
                
                solver.add_clause([-y, self.x[i1][j]])
                solver.add_clause([-y, self.x[i2][j]])
                solver.add_clause([-self.x[i1][j], -self.x[i2][j], y])
            
            self.add_at_most_k(solver, overlap_vars, max_overlap)
    
    # def _add_lex_constraint(self, solver, a, b_vec, n):
    #     """Add SAT encoding of a <=_lex b_vec (binary vectors of length n).
        
    #     Uses auxiliary prefix-equality variables p[k]:
    #       p[k] = True  iff  a[0..k] == b_vec[0..k]
    #     Constraint (a[0] <= b[0]) is always enforced.
    #     For each subsequent position k+1: if p[k], then a[k+1] <= b_vec[k+1].
    #     """
    #     # Position 0: a[0] <= b[0]  (no prefix needed)
    #     solver.add_clause([-a[0], b_vec[0]])

    #     prev_p = None  # p[-1] = True (empty prefix, always equal)

    #     for k in range(n - 1):
    #         p_k = self.var_id
    #         self.var_id += 1

    #         if prev_p is None:
    #             # p[0] = (a[0] == b_vec[0])
    #             solver.add_clause([-p_k, -a[0],    b_vec[0]])   # p[0] => a[0]->b[0]
    #             solver.add_clause([-p_k,  a[0],   -b_vec[0]])   # p[0] => b[0]->a[0]
    #             solver.add_clause([-a[0],   -b_vec[0],  p_k])   # a=b=1 => p[0]
    #             solver.add_clause([ a[0],    b_vec[0],  p_k])   # a=b=0 => p[0]
    #         else:
    #             # p[k] = prev_p AND (a[k] == b_vec[k])
    #             solver.add_clause([-p_k, prev_p])                        # p[k] => prev_p
    #             solver.add_clause([-p_k, -a[k],    b_vec[k]])            # p[k] => a[k]->b[k]
    #             solver.add_clause([-p_k,  a[k],   -b_vec[k]])            # p[k] => b[k]->a[k]
    #             solver.add_clause([-prev_p, -a[k], -b_vec[k],  p_k])    # prev AND a=b=1 => p[k]
    #             solver.add_clause([-prev_p,  a[k],  b_vec[k],  p_k])    # prev AND a=b=0 => p[k]

    #         # Main lex constraint at position k+1: if p[k], then a[k+1] <= b_vec[k+1]
    #         solver.add_clause([-p_k, -a[k + 1], b_vec[k + 1]])

    #         prev_p = p_k

    # def add_symmetry_breaking(self, solver):
    #     """Add symmetry breaking: lex ordering on rows and columns."""
    #     # Consecutive rows are lexicographically non-decreasing
    #     for i in range(self.v - 1):
    #         self._add_lex_constraint(solver, self.x[i], self.x[i + 1], self.b)

    #     # Consecutive columns are lexicographically non-decreasing
    #     for j in range(self.b - 1):
    #         col_j     = [self.x[i][j]     for i in range(self.v)]
    #         col_j1    = [self.x[i][j + 1] for i in range(self.v)]
    #         self._add_lex_constraint(solver, col_j, col_j1, self.v)

    #     if self.verbose:
    #         print("Added symmetry breaking (lex rows + lex columns)")

    def add_symmetry_breaking(self, solver):
        """Add symmetry breaking constraints"""
        for j in range(self.r):
            solver.add_clause([self.x[0][j]])
        
        for j in range(self.r, self.b):
            solver.add_clause([-self.x[0][j]])
            
        if self.verbose:
            print("Added symmetry breaking")
    
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
        solver = Glucose4()
        start_time = time.time()
        
        self.add_row_constraints(solver)
        self.add_overlap_constraints(solver, max_overlap)
        self.add_symmetry_breaking(solver)
        
        if self.verbose:
            print(f"Solving with lambda = {max_overlap}...")
        
        timed_out = [False]
        
        def interrupt_solver():
            timed_out[0] = True
            solver.interrupt()
        
        timer = threading.Timer(timeout, interrupt_solver)
        timer.start()
        
        try:
            sat = solver.solve_limited(expect_interrupt=True)
        finally:
            timer.cancel()
        
        solve_time = time.time() - start_time
        
        if timed_out[0] or sat is None:
            if self.verbose:
                print(f"TIMEOUT after {solve_time:.3f}s")
            solver.delete()
            return 'TIMEOUT', None, solve_time
        
        if sat:
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
    
    def binary_search_optimize(self, timeout=120, queue=None, running_flag=None):
        """Find optimal max overlap using binary search"""
        lb = self.compute_lower_bound()
        ub = self.r
        
        if self.verbose:
            print(f"\nBinary search for optimal lambda in [{lb}, {ub}]")
        
        best_lambda = ub
        best_matrix = None
        total_time = 0
        
        while lb <= ub:
            if running_flag is not None and not running_flag.value:
                break
                
            mid = (lb + ub) // 2
            
            status, matrix, solve_time = self.solve_with_max_overlap(mid, timeout)
            total_time += solve_time
            
            if status == 'SAT':
                best_lambda = mid
                best_matrix = matrix
                ub = mid - 1
                if queue is not None:
                    queue.put(best_lambda)
            else:
                lb = mid + 1
                
        if running_flag is not None and not running_flag.value:
            if queue is not None:
                queue.put(-best_lambda - 1)
        
        if self.verbose:
            print(f"\nBinary search complete!")
            print(f"Optimal lambda = {best_lambda}")
            print(f"Total time: {total_time:.3f}s")
        
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

def solve(filepath, timeout_per_call, quiet, queue, running_flag):
    """Worker function for multiprocessing"""
    params = parse_input_file(filepath)
    if not params or 'v' not in params or 'b' not in params or 'r' not in params:
        return
        
    v, b, r = params['v'], params['b'], params['r']
    try:
        solver = OpdSAT(v, b, r, verbose=not quiet)
        queue.put(r) # Initial worst case lambda
        solver.binary_search_optimize(timeout=timeout_per_call, queue=queue, running_flag=running_flag)
    except Exception as e:
        pass

def solve_from_file(filepath, timeout=120, overall_timeout=600, quiet=False):
    """Read params from file and solve with overall timeout. Returns dict of results."""
    filename = os.path.basename(filepath)
    if not quiet:
        print(f"\n{'='*60}")
        print(f"Processing file: {filename}")
        print(f"{'='*60}")
    
    params = parse_input_file(filepath)
    if not params or 'v' not in params or 'b' not in params or 'r' not in params:
        if not quiet: print("Error: Could not extract v, b, r from file.")
        return {
            'File': filename, 'v': None, 'b': None, 'r': None,
            'Optimal Lambda': None, 'Time (s)': 0, 'Status': 'Parse Error'
        }
        
    v, b, r = params['v'], params['b'], params['r']
    if not quiet: print(f"Parameters: v={v}, b={b}, r={r}")
    
    start = time.time()
    queue = multiprocessing.Queue()
    running_flag = multiprocessing.Value('b', True)
    
    p = multiprocessing.Process(target=solve, args=(filepath, timeout, quiet, queue, running_flag))
    p.start()
    
    try:
        p.join(timeout=overall_timeout)
        
        if p.is_alive():
            if not quiet: print(f"Quá thời gian {overall_timeout} giây cho toàn bộ quá trình!")
            running_flag.value = False
            p.terminate()
            p.join()
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Đã tiếp nhận yêu cầu dừng từ người dùng. Đang hủy tiến trình...")
        running_flag.value = False
        p.terminate()
        p.join()
        sys.exit(1)
        
    last_val = None
    while not queue.empty():
        last_val = queue.get()
        
    end = time.time()
    elapsed = round((end-start), 2)
    
    status = 'No Solution'
    optimal_lambda = None
    
    if last_val is not None:
        status = 'Solved'
        optimal_lambda = last_val
        if last_val < 0:
            status = 'Timeout'
            optimal_lambda = abs(last_val) - 1

    result_data = {
        'File': filename,
        'v': v, 'b': b, 'r': r,
        'Optimal Lambda': optimal_lambda,
        'Time (s)': elapsed,
        'Status': status
    }
    
    if not quiet:
        if optimal_lambda is not None:
            print(f"\nRESULT for {filename}:")
            print(f"Optimal lambda: {optimal_lambda}")
            print(f"Total time: {elapsed:.3f}s")
            print(f"Status: {status}")
        else:
            print(f"\nNo solution found for {filename}")
            
    return result_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SAT-based OPD solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
 
  python Opd_pure_sat_ver1.py --input input/small
        """
    )
    
    parser.add_argument('--input', type=str, required=True, help='Input file or directory path')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout per SAT call (seconds)')
    parser.add_argument('--overall_timeout', type=int, default=600, help='Overall timeout for the file (seconds)')
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
            try:
                import openpyxl
                has_openpyxl = True
            except ImportError:
                print("Warning: openpyxl not found. Excel export will be disabled.")
                has_openpyxl = False
            
            folder_name = os.path.basename(os.path.normpath(input_path))
            script_name = os.path.splitext(os.path.basename(__file__))[0]
            output_file = f"result_{folder_name}_{script_name}.xlsx"
            COLUMNS = ['File', 'v', 'b', 'r', 'Optimal Lambda', 'Time (s)', 'Status']
            
            for f in files:
                filepath = os.path.join(input_path, f)
                res = solve_from_file(filepath, args.timeout, args.overall_timeout, args.quiet)
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
                
    elif os.path.isfile(input_path):
        solve_from_file(input_path, args.timeout, args.overall_timeout, args.quiet)
    else:
        print(f"Error: Input path '{args.input}' not found.")
