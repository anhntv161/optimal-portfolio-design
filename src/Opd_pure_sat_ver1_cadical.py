"""
SAT-based OPD solver using Cadical195

The OPD problem:
- Given: v (rows), b (columns), r (row weight)
- Variables: x[i][j] ∈ {0,1} for i in [0..v-1], j in [0..b-1]
- Constraints:
  1. Each row sums to r
  2. Minimize max dot product between all pairs of distinct rows
"""

from pysat.solvers import Solver
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

ENCODING_MAP = {
    'sortnetwrk': EncType.sortnetwrk,
    'cardnetwrk': EncType.cardnetwrk,
    'seqcounter': EncType.seqcounter,
    'totalizer': EncType.totalizer,
    'mtotalizer': EncType.mtotalizer,
    'pairwise': EncType.pairwise
}


class OpdSAT:
    """SAT-based solver for Orthogonal Latin Defect problem"""
    
    def __init__(self, v, b, r, solver_name='cadical195', encoding_name='sortnetwrk', sym_options=None, verbose=True):
        self.v = v
        self.b = b
        self.r = r
        self.solver_name = solver_name
        self.encoding_name = encoding_name
        self.encoding_type = ENCODING_MAP.get(encoding_name, EncType.sortnetwrk)
        self.sym_options = sym_options or []
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
            print(f"Solver: {self.solver_name} | Encoding: {self.encoding_name}")
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
        cnf = CardEnc.equals(lits=vars, bound=self.r, top_id=self.var_id, encoding=self.encoding_type)
        solver.append_formula(cnf)
        self.var_id = cnf.nv
    
    def add_at_most_k(self, solver, vars, k):
        """Add constraint: at most k variables can be true"""
        if k >= len(vars):
            return
        
        cnf = CardEnc.atmost(lits=vars, bound=k, top_id=self.var_id, encoding=self.encoding_type)
        solver.append_formula(cnf)
        self.var_id = cnf.nv
    
    def add_row_constraints(self, solver, push_callback=None):
        """Add constraint: each row sums to r"""
        for i in range(self.v):
            self.add_exactly_r(solver, self.x[i])
            if push_callback: push_callback()
    
    def add_overlap_constraints(self, solver, max_overlap, push_callback=None):
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
            if push_callback: push_callback()
    def _add_lex_constraint(self, solver, a, b_vec, n):
        """Add SAT encoding of a <=_lex b_vec (binary vectors of length n).
        
        Uses auxiliary prefix-equality variables p[k]:
          p[k] = True  iff  a[0..k] == b_vec[0..k]
        Constraint (a[0] <= b[0]) is always enforced.
        For each subsequent position k+1: if p[k], then a[k+1] <= b_vec[k+1].
        """
        # Position 0: a[0] <= b[0]  (no prefix needed)
        solver.add_clause([-a[0], b_vec[0]])

        prev_p = None  # p[-1] = True (empty prefix, always equal)

        for k in range(n - 1):
            p_k = self.var_id
            self.var_id += 1

            if prev_p is None:
                # p[0] = (a[0] == b_vec[0])
                solver.add_clause([-p_k, -a[0],    b_vec[0]])   # p[0] => a[0]->b[0]
                solver.add_clause([-p_k,  a[0],   -b_vec[0]])   # p[0] => b[0]->a[0]
                solver.add_clause([-a[0],   -b_vec[0],  p_k])   # a=b=1 => p[0]
                solver.add_clause([ a[0],    b_vec[0],  p_k])   # a=b=0 => p[0]
            else:
                # p[k] = prev_p AND (a[k] == b_vec[k])
                solver.add_clause([-p_k, prev_p])                        # p[k] => prev_p
                solver.add_clause([-p_k, -a[k],    b_vec[k]])            # p[k] => a[k]->b[k]
                solver.add_clause([-p_k,  a[k],   -b_vec[k]])            # p[k] => b[k]->a[k]
                solver.add_clause([-prev_p, -a[k], -b_vec[k],  p_k])    # prev AND a=b=1 => p[k]
                solver.add_clause([-prev_p,  a[k],  b_vec[k],  p_k])    # prev AND a=b=0 => p[k]

            # Main lex constraint at position k+1: if p[k], then a[k+1] <= b_vec[k+1]
            solver.add_clause([-p_k, -a[k + 1], b_vec[k + 1]])

            prev_p = p_k

    def add_symmetry_breaking(self, solver):
        """Add symmetry breaking constraints based on sym_options"""
        fixed_r = 'r' in self.sym_options
        use_lex_row = 'lex_row' in self.sym_options
        use_lex_col = 'lex_col' in self.sym_options

        sym_methods = []

        if fixed_r:
            for j in range(self.r):
                solver.add_clause([self.x[0][j]])
            for j in range(self.r, self.b):
                solver.add_clause([-self.x[0][j]])
            sym_methods.append("fixed_r")

        if use_lex_row:
            sym_methods.append("lex_row")
            for i in range(self.v - 1):
                if fixed_r:
                    # If r is fixed, row 0 is maximum, so use descending lex
                    self._add_lex_constraint(solver, self.x[i + 1], self.x[i], self.b)
                else:
                    # Otherwise use ascending lex
                    self._add_lex_constraint(solver, self.x[i], self.x[i + 1], self.b)

        if use_lex_col:
            sym_methods.append("lex_col")
            for j in range(self.b - 1):
                col_j     = [self.x[i][j]     for i in range(self.v)]
                col_j1    = [self.x[i][j + 1] for i in range(self.v)]
                if fixed_r:
                    self._add_lex_constraint(solver, col_j1, col_j, self.v)
                else:
                    self._add_lex_constraint(solver, col_j, col_j1, self.v)

        self.applied_sym = "+".join(sym_methods) if sym_methods else "none"

        if self.verbose:
            print(f"Added symmetry breaking: {self.applied_sym}")
    
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
    
    def solve_with_max_overlap(self, max_overlap, timeout=120, queue=None):
        """Solve OPD with given max overlap constraint"""
        solver = Solver(name=self.solver_name)
        start_time = time.time()
        
        last_push_time = [time.time()]
        def push_stats(force=False):
            if queue is not None:
                current_time = time.time()
                if force or current_time - last_push_time[0] > 1.0:
                    try:
                        queue.put({
                            'n_vars': solver.nof_vars(),
                            'n_clauses': solver.nof_clauses(),
                            'applied_sym': getattr(self, 'applied_sym', 'none')
                        })
                        last_push_time[0] = current_time
                    except Exception:
                        pass
        
        self.add_row_constraints(solver, push_stats)
        self.add_overlap_constraints(solver, max_overlap, push_stats)
        self.add_symmetry_breaking(solver)
        push_stats(force=True)
        
        if self.verbose:
            print(f"Solving with lambda = {max_overlap}...")
        
        timed_out = [False]
        
        def interrupt_solver():
            timed_out[0] = True
            solver.interrupt()
        
        timer = threading.Timer(timeout, interrupt_solver)
        timer.start()
        
        n_vars = solver.nof_vars()
        n_clauses = solver.nof_clauses()

        try:
            sat = solver.solve_limited(expect_interrupt=True)
        finally:
            timer.cancel()
        
        solve_time = time.time() - start_time
        
        if timed_out[0] or sat is None:
            if self.verbose:
                print(f"TIMEOUT after {solve_time:.3f}s")
            solver.delete()
            return 'TIMEOUT', None, solve_time, n_vars, n_clauses
        
        if sat:
            model = solver.get_model()
            matrix = self.extract_matrix(model)
            
            valid, actual_overlap = self.verify_solution(matrix, max_overlap)
            
            if valid:
                if self.verbose:
                    print(f"SAT (lambda={actual_overlap}) in {solve_time:.3f}s")
                solver.delete()
                return 'SAT', matrix, solve_time, n_vars, n_clauses
            else:
                if self.verbose:
                    print(f"Solution verification failed!")
                solver.delete()
                return 'INVALID', matrix, solve_time, n_vars, n_clauses
        else:
            if self.verbose:
                print(f"UNSAT in {solve_time:.3f}s")
            solver.delete()
            return 'UNSAT', None, solve_time, n_vars, n_clauses
    
    def binary_search_optimize(self, timeout=120, queue=None):
        """Find optimal max overlap using binary search with global timeout"""
        lb = self.compute_lower_bound()
        ub = self.r
        
        if self.verbose:
            print(f"\nBinary search for optimal lambda in [{lb}, {ub}] with global timeout: {timeout}s")
        
        best_lambda = ub
        best_matrix = None
        best_vars = 0
        best_clauses = 0
        total_time = 0
        global_start_time = time.time()
        
        while lb <= ub:
            # Calculate remaining time for global timeout constraints
            time_spent = time.time() - global_start_time
            remaining_time = timeout - time_spent
            
            if remaining_time <= 0:
                if self.verbose:
                    print("\nStopping binary search: Global TIMEOUT reached.")
                break
                
            mid = (lb + ub) // 2
            
            # Pass the smaller of the remaining time or the specified timeout 
            # (though here timeout IS the global timeout)
            status, matrix, solve_time, n_vars, n_clauses = self.solve_with_max_overlap(mid, remaining_time, queue)
            total_time += solve_time
            best_vars = n_vars
            best_clauses = n_clauses
            
            if status == 'SAT':
                best_lambda = mid
                best_matrix = matrix
                ub = mid - 1
                
                if queue is not None:
                    queue.put({
                        'optimal_lambda': best_lambda,
                        'matrix': best_matrix,
                        'total_time': total_time,
                        'n_vars': best_vars,
                        'n_clauses': best_clauses,
                        'applied_sym': getattr(self, 'applied_sym', 'none'),
                        'status': 'Intermediate SAT'
                    })
            elif status == 'TIMEOUT':
                if self.verbose:
                    print("\nStopping binary search due to TIMEOUT (solver time limit reached).")
                break
            else:
                lb = mid + 1
        
        if self.verbose:
            print(f"\nBinary search complete!")
            print(f"Optimal lambda = {best_lambda}")
            print(f"Total time: {total_time:.3f}s")
        
        return best_lambda, best_matrix, total_time, best_vars, best_clauses, getattr(self, 'applied_sym', 'none')


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

def _solve_worker(v, b, r, solver_name, encoding_name, sym_options, quiet, queue, timeout):
    """Worker function for running the solver in a separate process"""
    try:
        t_start = time.time()
        solver = OpdSAT(v, b, r, solver_name=solver_name, encoding_name=encoding_name, sym_options=sym_options, verbose=not quiet)
        
        # Adjust timeout to account for model building
        remaining_timeout = max(0, timeout - (time.time() - t_start))
        optimal_lambda, matrix, _, n_vars, n_clauses, applied_sym = solver.binary_search_optimize(timeout=remaining_timeout, queue=queue)
        total_time = time.time() - t_start
        
        queue.put({
            'optimal_lambda': optimal_lambda,
            'matrix': matrix,
            'total_time': total_time,
            'n_vars': n_vars,
            'n_clauses': n_clauses,
            'applied_sym': applied_sym,
            'status': 'Solved' if matrix else 'No Solution'
        })
    except Exception as e:
        queue.put({
            'error': str(e)
        })

def solve_from_file(filepath, solver_name='cadical195', encoding_name='sortnetwrk', sym_options=None, timeout=600, quiet=False):
    """Read params from file and solve using multiprocessing for strict timeout. Returns dict of results."""
    start_time = time.time()  # Start measuring immediately to include parsing time
    
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
            'Solver': solver_name, 'Encoding': encoding_name,
            'Optimal Lambda': None, 'Variables': 0, 'Clauses': 0, 'Sym Method': 'none',
            'Time (s)': 0, 'Status': 'Parse Error'
        }
        
    v, b, r = params['v'], params['b'], params['r']
    if not quiet: print(f"Parameters: v={v}, b={b}, r={r}")
    
    queue = multiprocessing.Queue()
    
    p = multiprocessing.Process(target=_solve_worker, args=(v, b, r, solver_name, encoding_name, sym_options, quiet, queue, timeout))
    p.start()
    
    try:
        p.join(timeout=timeout)
        
        elapsed = time.time() - start_time
        
        if p.is_alive():
            if not quiet: print(f"\nTIMEOUT strictly enforced after {timeout} seconds.")
            p.terminate()
            p.join()
            
            # Retrieve the last valid partial result before termination
            optimal_lambda = None
            best_vars = 0
            best_clauses = 0
            applied_sym = "+".join(sym_options) if sym_options else 'none'
            total_time = round(elapsed, 3)
            while not queue.empty():
                result = queue.get()
                if 'n_vars' in result:
                    best_vars = max(best_vars, result['n_vars'])
                if 'n_clauses' in result:
                    best_clauses = max(best_clauses, result['n_clauses'])
                if 'applied_sym' in result:
                    applied_sym = result.get('applied_sym', applied_sym)
                if 'optimal_lambda' in result and result['optimal_lambda'] is not None:
                    optimal_lambda = result['optimal_lambda']
            
            if not quiet and optimal_lambda is not None:
                print(f"Partial Result before TIMEOUT: Optimal lambda = {optimal_lambda}")
                    
            return {
                'File': filename, 'v': v, 'b': b, 'r': r,
                'Solver': solver_name, 'Encoding': encoding_name,
                'Optimal Lambda': optimal_lambda, 'Variables': best_vars, 'Clauses': best_clauses,
                'Sym Method': applied_sym, 'Time (s)': total_time, 'Status': 'Timeout'
            }
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Đã tiếp nhận yêu cầu dừng từ người dùng. Đang hủy tiến trình...")
        p.terminate()
        p.join()
        sys.exit(1)
        
    elapsed = time.time() - start_time
    result = None
    best_vars = 0
    best_clauses = 0
    while not queue.empty():
        res = queue.get()
        if 'error' in res:
            result = res
            break
        if 'n_vars' in res:
            best_vars = max(best_vars, res['n_vars'])
        if 'n_clauses' in res:
            best_clauses = max(best_clauses, res['n_clauses'])
        if 'status' in res:
            result = res

    if result is not None:
        if 'error' in result:
             if not quiet: print(f"Solver error: {result['error']}")
             return {
                 'File': filename, 'v': v, 'b': b, 'r': r,
                 'Solver': solver_name, 'Encoding': encoding_name,
                 'Optimal Lambda': None, 'Variables': 0, 'Clauses': 0, 'Sym Method': 'none',
                 'Time (s)': 0, 'Status': f"Error: {result['error']}"
             }
             
        optimal_lambda = result.get('optimal_lambda')
        matrix = result.get('matrix')
        total_time = elapsed  # Use elapsed time (includes parsing and overhead)
        
        result_data = {
            'File': filename,
            'v': v, 'b': b, 'r': r,
            'Solver': solver_name,
            'Encoding': encoding_name,
            'Optimal Lambda': optimal_lambda,
            'Variables': best_vars,
            'Clauses': best_clauses,
            'Sym Method': result.get('applied_sym', 'none'),
            'Time (s)': round(total_time, 3), # Full execution time including file parsing
            'Status': result['status']
        }
        
        if matrix and not quiet:
            print(f"\nRESULT for {filename}:")
            print(f"Optimal lambda: {optimal_lambda}")
            print(f"Total time: {total_time:.3f}s")
            print(f"Solver: {solver_name}, Encoding: {encoding_name}")
            print(f"Variables: {result.get('n_vars', 0)}, Clauses: {result.get('n_clauses', 0)}, Sym Method: {result.get('applied_sym', 'none')}")
        elif not quiet:
            print(f"\nNo solution found for {filename}")
            
        return result_data
    else:
         if not quiet: print(f"Process crashed or returned no result.")
         return {
             'File': filename, 'v': v, 'b': b, 'r': r,
             'Solver': solver_name, 'Encoding': encoding_name,
             'Optimal Lambda': None, 'Variables': 0, 'Clauses': 0, 'Sym Method': 'none',
             'Time (s)': round(elapsed, 3), 'Status': 'Crashed'
         }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SAT-based OPD solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
 
  python Opd_pure_sat_ver1_cadical.py --input input/small
  python Opd_pure_sat_ver1_cadical.py --input input/small/instance.txt --timeout 300
  python Opd_pure_sat_ver1_cadical.py --input input/small --sym lex_row lex_col r
  python Opd_pure_sat_ver1_cadical.py --input input/small --solver glucose4 --encoding seqcounter
        """
    )
    
    parser.add_argument('--input', type=str, required=True, help='Input file or directory path')
    parser.add_argument('--timeout', type=int, default=3600, help='Overall timeout for solving the file (seconds)')
    parser.add_argument('--solver', type=str, default='cadical195', choices=['cadical195', 'glucose3', 'glucose4', 'minisat22', 'lingeling'], help='SAT solver to use')
    parser.add_argument('--encoding', type=str, default='sortnetwrk', choices=['sortnetwrk', 'cardnetwrk', 'seqcounter', 'totalizer', 'mtotalizer', 'pairwise'], help='Cardinality encoding method')
    parser.add_argument('--sym', nargs='+', default=[], help='Symmetry breaking methods (e.g., lex_row lex_col r)')
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
            COLUMNS = ['File', 'v', 'b', 'r', 'Solver', 'Encoding', 'Optimal Lambda', 'Variables', 'Clauses', 'Sym Method', 'Time (s)', 'Status']
            
            for f in files:
                filepath = os.path.join(input_path, f)
                res = solve_from_file(filepath, args.solver, args.encoding, args.sym, args.timeout, args.quiet)
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
        solve_from_file(input_path, args.solver, args.encoding, args.sym, args.timeout, args.quiet)
    else:
        print(f"Error: Input path '{args.input}' not found.")
