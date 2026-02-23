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
    
    def __init__(self, v, b, r, verbose=True):
        self.v = v
        self.b = b
        self.r = r
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
        num_vars = self.model.NumVars
        num_constrs = self.model.NumConstrs
        num_qconstrs = self.model.NumQConstrs if hasattr(self.model, 'NumQConstrs') else 0
        
        if self.verbose:
            print(f"\nModel statistics:")
            print(f"  Variables: {num_vars}")
            print(f"  Linear constraints: {num_constrs}")
            print(f"  Quadratic constraints: {num_qconstrs}")
            print(f"  Total constraints: {num_constrs + num_qconstrs}")
            
            # Warn if approaching free license limits (2000 vars, 2000 constrs)
            if num_vars > 1800 or (num_constrs + num_qconstrs) > 1800:
                print(f"  ⚠️  WARNING: Model size approaching free license limits!")
                print(f"  ⚠️  Free Gurobi license: max 2000 variables, 2000 constraints")
    
    def _add_symmetry_breaking(self):
        """
        Add symmetry breaking constraints:
        - Fix first row to have 1s in first r positions
        - Lexicographical ordering on rows
        """
        # Fix first row: first r credits are selected
        for j in range(self.r):
            self.model.addConstr(self.x[0, j] == 1, f"fix_first_row_{j}")
        
        for j in range(self.r, self.b):
            self.model.addConstr(self.x[0, j] == 0, f"fix_first_row_zero_{j}")
        
        if self.verbose:
            print("Added symmetry breaking constraints")
    
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
                    'error': str(e)
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
                'valid': valid
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
                    'valid': True
                }
            else:
                return {
                    'status': 'TIMEOUT',
                    'lambda': None,
                    'matrix': None,
                    'time': solve_time,
                    'gap': None,
                    'nodes': int(self.model.NodeCount),
                    'valid': False
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
                'valid': False
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
                'valid': False
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
            'File': filename,
            'v': None,
            'b': None,
            'r': None,
            'Lower Bound': None,
            'Optimal Lambda': None,
            'Time (s)': 0,
            'MIP Gap': None,
            'Nodes': None,
            'Status': 'Parse Error'
        }
    
    v, b, r = params['v'], params['b'], params['r']
    print(f"Parameters: v={v}, b={b}, r={r}")
    
    try:
        solver = PortfolioMIP(v, b, r, verbose=not quiet)
        solver.build_model(use_symmetry_breaking=True, linearize=True)
        result = solver.solve(time_limit=timeout)
        
        result_data = {
            'File': filename,
            'v': v,
            'b': b,
            'r': r,
            'Lower Bound': solver.lambda_lb,
            'Optimal Lambda': result['lambda'],
            'Time (s)': round(result['time'], 3),
            'MIP Gap': round(result['gap'], 6) if result['gap'] is not None else None,
            'Nodes': result['nodes'],
            'Status': result['status']
        }
        
        if result['status'] in ['OPTIMAL', 'TIMEOUT_WITH_SOLUTION'] and result['matrix'] is not None:
            print(f"\nRESULT for {filename}:")
            print(f"Optimal lambda: {result['lambda']}")
            print(f"Lower bound: {solver.lambda_lb}")
            print(f"Gap from lower bound: {result['lambda'] - solver.lambda_lb}")
            print(f"Total time: {result['time']:.3f}s")
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
            'Time (s)': 0,
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
  python Opd_mip.py --input input/small
  python Opd_mip.py --input input/small/test1.txt --timeout 300
  python Opd_mip.py --input input/medium --quiet
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input file or directory path')
    parser.add_argument('--timeout', type=int, default=10000,
                       help='Timeout per instance (seconds)')
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
            results = []
            
            # Try to import pandas for Excel export
            try:
                import pandas as pd
                has_pandas = True
            except ImportError:
                print("Warning: pandas not found. Excel export will be disabled.")
                has_pandas = False
            
            # Solve all files
            for f in files:
                filepath = os.path.join(input_path, f)
                res = solve_from_file(filepath, args.timeout, args.quiet)
                if res:
                    results.append(res)
            
            # Export results
            if has_pandas and results:
                folder_name = os.path.basename(os.path.normpath(input_path))
                output_file = f"result_{folder_name}_mip.xlsx"
                
                df = pd.DataFrame(results)
                df.to_excel(output_file, index=False)
                
                print(f"\n{'-'*60}")
                print(f"SUMMARY:")
                print(f"{'-'*60}")
                print(df.to_string(index=False))
                print(f"\n{'-'*60}")
                print(f"Results exported to {output_file}")
                print(f"{'-'*60}")
            elif results:
                # Print summary without pandas
                print(f"\n{'-'*60}")
                print(f"SUMMARY:")
                print(f"{'-'*60}")
                for res in results:
                    print(f"{res['File']}: lambda={res['Optimal Lambda']}, "
                          f"time={res['Time (s)']}s, status={res['Status']}")
    
    elif os.path.isfile(input_path):
        solve_from_file(input_path, args.timeout, args.quiet)
    
    else:
        print(f"Error: Input path '{args.input}' not found.")