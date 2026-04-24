"""
MIP-based Portfolio Optimization solver using Gurobi

The Portfolio Optimization problem:
- Given: v (sub-pools), b (credits), r (sub-pool size)
- Variables: x[i][j] ∈ {0,1} for i in [0..v-1], j in [0..b-1]
- Constraints:
  1. Each row (sub-pool) sums to r
  2. Minimize max overlap between all pairs of distinct rows
"""

import gurobipy as gp
from gurobipy import GRB
import argparse
import time
import math
import os
import re
import numpy as np


class PortfolioMIP:
    """MIP-based solver for Portfolio Optimization problem"""
    
    def __init__(self, v, b, r, sym_options=None, verbose=True):
        self.v = v
        self.b = b
        self.r = r
        self.sym_options = sym_options or []
        self.verbose = verbose
        
        # Calculate lower bound on lambda
        self.lambda_lb = self._compute_lower_bound()
        
        if self.verbose:
            print(f"Portfolio Optimization Instance: v={v}, b={b}, r={r}")
            print(f"Solver: Gurobi MIP")
            print(f"Lower bound on lambda: {self.lambda_lb}")
    
    def _compute_lower_bound(self):
        """
        Compute theoretical lower bound on lambda using constraint (9):
        λ ≥ ⌈r(rv - b) / (b(v-1))⌉ ∧ λ ≥ 0
        """
        if self.v <= 1:
            return 0
        numerator = self.r * (self.r * self.v - self.b)
        denominator = self.b * (self.v - 1)
        if denominator == 0:
            return 0
        lb = numerator / denominator
        return max(0, math.ceil(lb))
    
    def build_model(self, use_symmetry_breaking=True, linearize=False):
        """Build the MIP model"""
        
        self.model = gp.Model("portfolio_optimization")
        
        # Suppress Gurobi output if not verbose
        if not self.verbose:
            self.model.setParam('OutputFlag', 0)
        
        # Decision variables: x[i,j] = 1 if credit j in sub-pool i
        self.x = self.model.addVars(
            self.v, self.b,
            vtype=GRB.BINARY,
            name="x"
        )
        
        # Lambda variable: maximum overlap
        self.lambda_var = self.model.addVar(
            vtype=GRB.INTEGER,
            lb=self.lambda_lb,
            ub=self.r,
            name="lambda"
        )
        
        # Constraint (6): Each sub-pool has exactly r credits
        for i in range(self.v):
            self.model.addConstr(
                gp.quicksum(self.x[i, j] for j in range(self.b)) == self.r,
                f"subpool_size_{i}"
            )
        
        # Constraint (7): Pairwise overlap <= lambda
        if linearize:
            # Linearized version: introduce auxiliary variables y[i1,i2,j] = x[i1,j] AND x[i2,j]
            self.y = self.model.addVars(
                [(i1, i2, j) for i1 in range(self.v) for i2 in range(i1 + 1, self.v) for j in range(self.b)],
                vtype=GRB.BINARY,
                name="y"
            )
            
            for i1 in range(self.v):
                for i2 in range(i1 + 1, self.v):
                    for j in range(self.b):
                        # y[i1,i2,j] <= x[i1,j]
                        self.model.addConstr(self.y[i1, i2, j] <= self.x[i1, j])
                        # y[i1,i2,j] <= x[i2,j]
                        self.model.addConstr(self.y[i1, i2, j] <= self.x[i2, j])
                        # y[i1,i2,j] >= x[i1,j] + x[i2,j] - 1
                        self.model.addConstr(self.y[i1, i2, j] >= self.x[i1, j] + self.x[i2, j] - 1)
                    
                    # Overlap constraint
                    overlap = gp.quicksum(self.y[i1, i2, j] for j in range(self.b))
                    self.model.addConstr(
                        overlap <= self.lambda_var,
                        f"overlap_{i1}_{i2}"
                    )
        else:
            # Quadratic version (requires Gurobi to handle quadratic constraints)
            for i1 in range(self.v):
                for i2 in range(i1 + 1, self.v):
                    overlap = gp.quicksum(
                        self.x[i1, j] * self.x[i2, j]
                        for j in range(self.b)
                    )
                    self.model.addConstr(
                        overlap <= self.lambda_var,
                        f"overlap_{i1}_{i2}"
                    )
        
        # Symmetry breaking constraints
        if use_symmetry_breaking:
            self._add_symmetry_breaking()
        
        # Objective (8): Minimize lambda
        self.model.setObjective(self.lambda_var, GRB.MINIMIZE)
        
        # Update model to get statistics
        self.model.update()
        
        # Check model size for free license limits
        self.num_vars = self.model.NumVars
        self.num_constrs = self.model.NumConstrs + (self.model.NumQConstrs if hasattr(self.model, 'NumQConstrs') else 0)
        
        if self.verbose:
            print(f"\nModel statistics:")
            print(f"  Variables: {self.num_vars}")
            print(f"  Total constraints: {self.num_constrs}")
            
            # Warn if approaching free license limits (2000 vars, 2000 constrs)
            if self.num_vars > 1800 or self.num_constrs > 1800:
                print(f"  ⚠️  WARNING: Model size approaching free license limits!")
                print(f"  ⚠️  Free Gurobi license: max 2000 variables, 2000 constraints")
    
    def _add_lex_le(self, a, b_vec, n, prefix):
        """
        Add MIP encoding of a <=_lex b_vec (binary vectors of length n).

        Encoding via prefix-equality propagation:
          p[0] = 1  (Python int constant — empty prefix is always equal)
          eq_k  = 1 iff a[k] == b_vec[k]   (auxiliary Gurobi binary var)
          p[k+1] = p[k] AND eq_k            (Gurobi binary var for k >= 1)
          p[k] => a[k] <= b_vec[k]          (core lex constraint at each position)

        IMPORTANT: p[0] is a Python int (1), while p[k] for k>=1 are Gurobi Vars.
        Use isinstance(p[k], int) — NOT `p[k] == 1` — to distinguish them,
        because comparing a Gurobi Var with == returns a TempConstr object
        (not a bool), which raises "Constraint has no bool value".
        """
        # p[0] = 1: Python int constant (empty prefix always equal)
        p = {0: 1}

        for k in range(n):
            # Check whether p[k] is the Python constant 1 or a Gurobi variable.
            # MUST use isinstance — never `p[k] == 1` when p[k] may be a Gurobi Var.
            p_k_is_const = isinstance(p[k], int)

            # Build eq_k only when needed for p[k+1] propagation (k < n-1)
            if k < n - 1:
                eq_k = self.model.addVar(vtype=GRB.BINARY, name=f"{prefix}_eq_{k}")
                # eq_k = 1 iff a[k] == b_vec[k]
                self.model.addConstr(a[k] - b_vec[k] <= 1 - eq_k, f"{prefix}_eq1_{k}")
                self.model.addConstr(b_vec[k] - a[k] <= 1 - eq_k, f"{prefix}_eq2_{k}")
                # Force eq_k = 1 when a[k] == b_vec[k] (both 1 or both 0)
                self.model.addConstr(eq_k >= a[k] + b_vec[k] - 1,  f"{prefix}_eq3_{k}")  # both 1
                self.model.addConstr(eq_k >= 1 - a[k] - b_vec[k],  f"{prefix}_eq4_{k}")  # both 0

                # p[k+1] = p[k] AND eq_k
                if p_k_is_const:
                    # p[k] = 1 (constant) => p[k+1] = eq_k directly (no new var needed)
                    p[k + 1] = eq_k
                else:
                    pk1 = self.model.addVar(vtype=GRB.BINARY, name=f"{prefix}_p_{k+1}")
                    self.model.addConstr(pk1 <= p[k],             f"{prefix}_p1_{k+1}")
                    self.model.addConstr(pk1 <= eq_k,             f"{prefix}_p2_{k+1}")
                    self.model.addConstr(pk1 >= p[k] + eq_k - 1, f"{prefix}_p3_{k+1}")
                    p[k + 1] = pk1

            # Core lex constraint: p[k] => a[k] <= b_vec[k]
            # If prefix is equal up to position k, then a[k] must be <= b_vec[k].
            if p_k_is_const:
                # p[k] = 1 (constant) => unconditionally enforce a[k] <= b_vec[k]
                self.model.addConstr(a[k] <= b_vec[k], f"{prefix}_lex_{k}")
            else:
                # p[k] is a Gurobi Var: linearize  p[k]=1 => a[k] <= b_vec[k]
                # as:  a[k] - b_vec[k] <= 1 - p[k]
                self.model.addConstr(a[k] - b_vec[k] <= 1 - p[k], f"{prefix}_lex_{k}")

    def _add_symmetry_breaking(self):
        """
        Add symmetry breaking constraints based on sym_options:
          - r: Fix row 0 to have 1 in the first r elements
          - lex_row: Lexicographic ordering between consecutive rows
          - lex_col: Lexicographic ordering between consecutive columns
        """
        use_lex_row = 'lex_row' in self.sym_options
        use_lex_col = 'lex_col' in self.sym_options
        fixed_r = 'r' in self.sym_options
        
        sym_methods = []
        
        # r: Fix first r columns of row 0 to 1
        if fixed_r:
            sym_methods.append('r')
            for j in range(self.r):
                self.model.addConstr(self.x[0, j] == 1, f"fix_row0_one_{j}")
            for j in range(self.r, self.b):
                self.model.addConstr(self.x[0, j] == 0, f"fix_row0_zero_{j}")

        # Lex ordering on consecutive rows (rows 0..v-2)
        if use_lex_row:
            sym_methods.append("lex_row")
            for i in range(self.v - 1):
                a_row = [self.x[i,   j] for j in range(self.b)]
                b_row = [self.x[i+1, j] for j in range(self.b)]
                if fixed_r:
                    self._add_lex_le(b_row, a_row, self.b, f"row_lex_{i}")
                else:
                    self._add_lex_le(a_row, b_row, self.b, f"row_lex_{i}")

        # Lex ordering on consecutive columns (col[j] <=_lex col[j+1])
        if use_lex_col:
            sym_methods.append("lex_col")
            for j in range(self.b - 1):
                a_col = [self.x[i, j]   for i in range(self.v)]
                b_col = [self.x[i, j+1] for i in range(self.v)]
                if fixed_r:
                    self._add_lex_le(b_col, a_col, self.v, f"col_lex_{j}")
                else:
                    self._add_lex_le(a_col, b_col, self.v, f"col_lex_{j}")

        self.applied_sym = "+".join(sym_methods) if sym_methods else "none"

        if self.verbose:
            print(f"Added symmetry breaking: {self.applied_sym}")
    
    def solve(self, time_limit=3600):
        """Solve the MIP model"""
        
        # Set parameters
        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('MIPGap', 0.0)  # Find optimal solution
        
        # Additional tuning for better performance
        self.model.setParam('Cuts', 2)  # Aggressive cuts
        self.model.setParam('Heuristics', 0.1)  # 10% time on heuristics
        
        start_time = time.time()
        
        if self.verbose:
            print(f"\nSolving MIP model...")
        
        try:
            self.model.optimize()
            solve_time = time.time() - start_time
            return self._extract_solution(solve_time)
        except gp.GurobiError as e:
            solve_time = time.time() - start_time
            if 'Model too large' in str(e) or 'size-limited' in str(e):
                if self.verbose:
                    print(f"\n❌ ERROR: Model exceeds Gurobi free license limits")
                    print(f"   Free license: max 2000 variables and 2000 constraints")
                    print(f"   Your model: {self.model.NumVars} variables, {self.model.NumConstrs + (self.model.NumQConstrs if hasattr(self.model, 'NumQConstrs') else 0)} constraints")
                    print(f"   Solution: Use a smaller instance or get a full Gurobi license")
                return {
                    'status': 'LICENSE_ERROR',
                    'lambda': None,
                    'matrix': None,
                    'time': solve_time,
                    'gap': None,
                    'nodes': 0,
                    'valid': False,
                    'error': str(e),
                    'n_vars': getattr(self, 'num_vars', 0),
                    'n_constrs': getattr(self, 'num_constrs', 0),
                    'applied_sym': getattr(self, 'applied_sym', 'none')
                }
            else:
                raise
    
    def _extract_solution(self, solve_time):
        """Extract solution from solved model"""
        
        if self.model.status == GRB.OPTIMAL:
            # Extract matrix
            matrix = np.zeros((self.v, self.b), dtype=int)
            for i in range(self.v):
                for j in range(self.b):
                    matrix[i, j] = round(self.x[i, j].x)
            
            optimal_lambda = round(self.lambda_var.x)
            
            # Verify solution
            valid, actual_lambda = self._verify_solution(matrix, optimal_lambda)
            
            if self.verbose:
                print(f"\nSOLVED in {solve_time:.3f}s")
                print(f"Optimal lambda: {optimal_lambda}")
                print(f"MIP Gap: {self.model.MIPGap:.6f}")
                print(f"Nodes explored: {int(self.model.NodeCount)}")
            
            return {
                'status': 'OPTIMAL',
                'lambda': optimal_lambda,
                'matrix': matrix,
                'time': solve_time,
                'gap': self.model.MIPGap,
                'nodes': int(self.model.NodeCount),
                'valid': valid,
                'n_vars': getattr(self, 'num_vars', 0),
                'n_constrs': getattr(self, 'num_constrs', 0),
                'applied_sym': getattr(self, 'applied_sym', 'none')
            }
        
        elif self.model.status == GRB.TIME_LIMIT:
            if self.verbose:
                print(f"\nTIMEOUT after {solve_time:.3f}s")
            
            # Try to extract incumbent solution if available
            if self.model.SolCount > 0:
                matrix = np.zeros((self.v, self.b), dtype=int)
                for i in range(self.v):
                    for j in range(self.b):
                        matrix[i, j] = round(self.x[i, j].x)
                
                incumbent_lambda = round(self.lambda_var.x)
                
                if self.verbose:
                    print(f"Incumbent solution found: lambda = {incumbent_lambda}")
                    print(f"MIP Gap: {self.model.MIPGap:.6f}")
                
                return {
                    'status': 'TIMEOUT_WITH_SOLUTION',
                    'lambda': incumbent_lambda,
                    'matrix': matrix,
                    'time': solve_time,
                    'gap': self.model.MIPGap,
                    'nodes': int(self.model.NodeCount),
                    'valid': True,
                    'n_vars': getattr(self, 'num_vars', 0),
                    'n_constrs': getattr(self, 'num_constrs', 0),
                    'applied_sym': getattr(self, 'applied_sym', 'none')
                }
            else:
                return {
                    'status': 'TIMEOUT',
                    'lambda': None,
                    'matrix': None,
                    'time': solve_time,
                    'gap': None,
                    'nodes': int(self.model.NodeCount),
                    'valid': False,
                    'n_vars': getattr(self, 'num_vars', 0),
                    'n_constrs': getattr(self, 'num_constrs', 0),
                    'applied_sym': getattr(self, 'applied_sym', 'none')
                }
        
        elif self.model.status == GRB.INFEASIBLE:
            if self.verbose:
                print(f"\nINFEASIBLE")
            
            return {
                'status': 'INFEASIBLE',
                'lambda': None,
                'matrix': None,
                'time': solve_time,
                'gap': None,
                'nodes': 0,
                'valid': False,
                'n_vars': getattr(self, 'num_vars', 0),
                'n_constrs': getattr(self, 'num_constrs', 0),
                'applied_sym': getattr(self, 'applied_sym', 'none')
            }
        
        else:
            if self.verbose:
                print(f"\nUNKNOWN STATUS: {self.model.status}")
            
            return {
                'status': f'UNKNOWN_{self.model.status}',
                'lambda': None,
                'matrix': None,
                'time': solve_time,
                'gap': None,
                'nodes': int(self.model.NodeCount),
                'valid': False,
                'n_vars': getattr(self, 'num_vars', 0),
                'n_constrs': getattr(self, 'num_constrs', 0),
                'applied_sym': getattr(self, 'applied_sym', 'none')
            }
    
    def _verify_solution(self, matrix, target_lambda):
        """Verify that solution satisfies all constraints"""
        
        # Check row sums
        for i in range(self.v):
            row_sum = np.sum(matrix[i])
            if row_sum != self.r:
                if self.verbose:
                    print(f"WARNING: Row {i} sum = {row_sum}, expected {self.r}")
                return False, -1
        
        # Check overlaps
        max_overlap = 0
        for i1 in range(self.v):
            for i2 in range(i1 + 1, self.v):
                overlap = np.sum(matrix[i1] * matrix[i2])
                max_overlap = max(max_overlap, overlap)
                if overlap > target_lambda:
                    if self.verbose:
                        print(f"WARNING: Overlap({i1},{i2}) = {overlap} > {target_lambda}")
                    return False, overlap
        
        return True, max_overlap
    
    def print_matrix(self, matrix):
        """Print matrix in readable format"""
        print("\nSolution matrix:")
        for i in range(self.v):
            row_str = ''.join(str(int(x)) for x in matrix[i])
            row_sum = int(np.sum(matrix[i]))
            print(f"  Row {i:2d}: {row_str} (sum={row_sum})")


def parse_input_file(filepath):
    """Parse v, b, r from input file"""
    params = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        v_match = re.search(r'v\s*=\s*(\d+)', content)
        b_match = re.search(r'b\s*=\s*(\d+)', content)
        r_match = re.search(r'r\s*=\s*(\d+)', content)
        
        if v_match:
            params['v'] = int(v_match.group(1))
        if b_match:
            params['b'] = int(b_match.group(1))
        if r_match:
            params['r'] = int(r_match.group(1))
        
        return params
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
        return None


def solve_from_file(filepath, sym_options=None, timeout=120, quiet=False):
    """Read params from file and solve. Returns dict of results."""
    start_time = time.time()  # Start measuring immediately to include parsing time
    
    filename = os.path.basename(filepath)
    print(f"\n{'='*60}")
    print(f"Processing file: {filename}")
    print(f"{'='*60}")
    
    params = parse_input_file(filepath)
    if not params or 'v' not in params or 'b' not in params or 'r' not in params:
        print("Error: Could not extract v, b, r from file.")
        return {
            'File': filename,
            'v': None,
            'b': None,
            'r': None,
            'Lower Bound': None,
            'Optimal Lambda': None,
            'Variables': 0,
            'Clauses': 0,
            'Sym Method': 'none',
            'Time (s)': 0,
            'MIP Gap': None,
            'Nodes': None,
            'Status': 'Parse Error'
        }
    
    v, b, r = params['v'], params['b'], params['r']
    print(f"Parameters: v={v}, b={b}, r={r}")
    
    try:
        solver = PortfolioMIP(v, b, r, sym_options=sym_options, verbose=not quiet)
        solver.build_model(use_symmetry_breaking=True, linearize=True)
        
        encoding_time = time.time() - start_time
        remaining_timeout = max(0, timeout - encoding_time)
        result = solver.solve(time_limit=remaining_timeout)
        total_time = time.time() - start_time
        
        result_data = {
            'File': filename,
            'v': v,
            'b': b,
            'r': r,
            'Lower Bound': solver.lambda_lb,
            'Optimal Lambda': result['lambda'],
            'Variables': result.get('n_vars', 0),
            'Clauses': result.get('n_constrs', 0),
            'Sym Method': result.get('applied_sym', 'none'),
            'Time (s)': round(total_time, 3),
            'MIP Gap': round(result['gap'], 6) if result['gap'] is not None else None,
            'Nodes': result['nodes'],
            'Status': result['status']
        }
        
        if result['status'] in ['OPTIMAL', 'TIMEOUT_WITH_SOLUTION'] and result['matrix'] is not None:
            print(f"\nRESULT for {filename}:")
            print(f"Optimal lambda: {result['lambda']}")
            print(f"Lower bound: {solver.lambda_lb}")
            print(f"Gap from lower bound: {result['lambda'] - solver.lambda_lb}")
            print(f"Total time: {total_time:.3f}s")
            print(f"Nodes explored: {result['nodes']}")
            
            if not quiet:
                solver.print_matrix(result['matrix'])
        else:
            print(f"\nNo solution found for {filename}")
            print(f"Status: {result['status']}")
        
        return result_data
    
    except Exception as e:
        print(f"Solver error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'File': filename,
            'v': v,
            'b': b,
            'r': r,
            'Lower Bound': None,
            'Optimal Lambda': None,
            'Variables': 0,
            'Clauses': 0,
            'Sym Method': 'none',
            'Time (s)': round(time.time() - start_time, 3),
            'MIP Gap': None,
            'Nodes': None,
            'Status': f'Error: {str(e)}'
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='MIP-based Portfolio Optimization solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Opd_mip_gurobi.py --input input/small
  python Opd_mip_gurobi.py --input input/small/test1.txt --timeout 300
  python Opd_mip_gurobi.py --input input/medium --quiet
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input file or directory path')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Timeout per instance (seconds)')
    parser.add_argument('--sym', nargs='+', default=[],
                       help='Symmetry breaking methods (e.g., lex_row lex_col)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    input_path = args.input
    
    # Handle relative paths
    if not os.path.exists(input_path):
        potential_path = os.path.join('input', input_path)
        if os.path.exists(potential_path):
            input_path = potential_path
    
    # Process directory or single file
    if os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        files = sorted([f for f in os.listdir(input_path) if f.endswith('.txt')])
        
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
            COLUMNS = ['File', 'v', 'b', 'r', 'Lower Bound', 'Optimal Lambda', 'Variables', 'Clauses', 'Sym Method', 'Time (s)', 'MIP Gap', 'Nodes', 'Status']
            
            # Solve all files
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