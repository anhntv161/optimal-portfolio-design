"""
SAT-based OPD Solver using Reified Bound (then-activate) Strategy

The OPD problem:
- Given: v (rows/sub-pools), b (columns/credits), r (row weight)
- Variables: x[i][k] in {0,1}  for i in [0..v-1], k in [0..b-1]
- Constraints:
  1. Each row sums to r (constant-weight)
  2. Minimize max dot product (overlap) between all pairs of rows

Reified Bound Strategy ("Build once - activate gradually"):
  - Build the ENTIRE cardinality structure (Totalizer) for every pair (i,j) ONCE.
  - For each bound value M (0..r), create a reified literal b_M such that:
        b_M => (D_{i,j} <= M)  for all pairs i < j
    encoded as CNF:   (-b_M  OR  -o[i][j][M])
    where o[i][j][M] is 0-indexed: o[M]=1 means D_{ij} >= M+1
  - Chain monotonicity:  b_M => b_{M+1}
    encoded as:          (-b_M  OR  b_{M+1})

Search direction (UB -> LB, assumption-based):
    for M = r down to lb:
        solver.solve(assumptions=[b_M])
        if SAT  -> record best solution; try M-1
        if UNSAT -> optimal = last SAT value; stop

Key difference from Opd_incremental_sat_preliminary.py:
  - OLD: adds a new at-most-M CardEnc on every iteration (rebuilds aux vars).
  - NEW: adds ALL reified clauses ONCE, then only flips one assumption literal.

Timeout:
  Cadical195 does NOT support threading-based interrupt (raises NotImplementedError).
  We use SIGALRM (Unix) to time out each solve() call safely in the main thread.
"""

from pysat.solvers import Cadical195
from pysat.card import CardEnc, EncType
import argparse
import itertools
import math
import os
import re
import time
# ---------------------------------------------------------------------------
# Totalizer Construction (upward clauses only)
# ---------------------------------------------------------------------------

def _totalizer_merge(left_out, right_out, var_counter, clauses):
    """
    Merge two totalizer output lists into a combined output list.

    Semantics (0-indexed):
        left_out[i]  = 1  =>  left_sum  >= i+1
        right_out[j] = 1  =>  right_sum >= j+1
        out[s]       = 1  =>  total_sum >= s+1

    Only UPWARD clauses are added (sufficient for reified at-most encoding):
        For i in [0..a], j in [0..b], s = i+j (s >= 1):
            (-left_out[i-1]  OR  -right_out[j-1]  OR  out[s-1])
        meaning: left >= i  AND  right >= j  =>  total >= i+j
    """
    a = len(left_out)
    b = len(right_out)
    out_size = a + b

    out = []
    for _ in range(out_size):
        out.append(var_counter[0])
        var_counter[0] += 1

    for i in range(a + 1):
        for j in range(b + 1):
            s = i + j
            if s == 0 or s > out_size:
                continue
            cl = []
            if i >= 1:
                cl.append(-left_out[i - 1])
            if j >= 1:
                cl.append(-right_out[j - 1])
            cl.append(out[s - 1])
            clauses.append(cl)

    return out


def _build_totalizer_manual(input_lits, var_counter, clauses):
    """
    Recursively build a Totalizer for input_lits.

    Returns: out[0..n-1] where out[t] = 1  =>  sum(input_lits) >= t+1
    Only upward clauses are generated.
    """
    n = len(input_lits)
    if n == 0:
        return []
    if n == 1:
        return [input_lits[0]]
    mid = n // 2
    left_out  = _build_totalizer_manual(input_lits[:mid], var_counter, clauses)
    right_out = _build_totalizer_manual(input_lits[mid:], var_counter, clauses)
    return _totalizer_merge(left_out, right_out, var_counter, clauses)


# ---------------------------------------------------------------------------
# Main Solver Class
# ---------------------------------------------------------------------------

class OpdReifiedBound:
    """
    SAT-based OPD solver using the Reified Bound (then-activate) strategy.

    Build phase (once):
      1. x[i][k]    - primary variables
      2. s[i][j][k] - AND-gate vars: s = x[i][k] AND x[j][k]
      3. Totalizer per pair (i,j) -> output vars o[i][j][*]
      4. b_M literals for M in [0..r]
      5. Formula F: row weights + AND encoding + symmetry breaking + totalizers
      6. Reified clauses: (-b_M OR -o[i][j][M]) for all pairs, all M
      7. Monotonicity: (-b_M OR b_{M+1}) for M < r

    Search (UB -> LB):
      For M = r, r-1, ..., lb:
          SAT? (assume b_M)  -> record best, continue to M-1
          UNSAT?             -> optimal proven = last SAT's M; stop
    """

    def __init__(self, v, b, r, verbose=True):
        self.v = v
        self.b = b
        self.r = r
        self.verbose = verbose
        self._next_var = 1

        # x[i][k]
        self.x = [[self._alloc() for _ in range(b)] for _ in range(v)]

        # pairs i < j
        self.pairs = list(itertools.combinations(range(v), 2))

        # s[(i,j)][k]
        self.s = {(i, j): [self._alloc() for _ in range(b)]
                  for (i, j) in self.pairs}

        # b_M for M = 0..r
        self.b_lit = {M: self._alloc() for M in range(r + 1)}

        if verbose:
            print(f"OPD Reified Bound: v={v}, b={b}, r={r}")
            print(f"  x vars:   {v * b}")
            print(f"  s vars:   {len(self.pairs) * b}")
            print(f"  b_M vars: {r + 1}")

    def _alloc(self):
        v = self._next_var
        self._next_var += 1
        return v

    def compute_lower_bound(self):
        """Theoretical lower bound: ceil(r*(r*v - b) / (b*(v-1)))"""
        if self.v <= 1:
            return 0
        num = self.r * (self.r * self.v - self.b)
        den = self.b * (self.v - 1)
        if den <= 0:
            return 0
        return max(0, math.ceil(num / den))

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_formula(self):
        """
        Build Cadical195 solver with full formula F + reified bounds.

        Returns: (solver, o_vars)
            o_vars: dict (i,j) -> list of output vars [o_0..o_{b-1}]
                    o[t] = 1 => D_{ij} >= t+1
        """
        solver = Cadical195()

        # (A) Row cardinality: sum_k x[i][k] = r
        if self.verbose:
            print("  Building row cardinality constraints...")
        for i in range(self.v):
            cnf = CardEnc.equals(
                lits=self.x[i],
                bound=self.r,
                top_id=self._next_var - 1,
                encoding=EncType.totalizer,
            )
            self._next_var = cnf.nv + 1
            for cl in cnf.clauses:
                solver.add_clause(cl)

        # (B) AND encoding: s[i][j][k] <-> x[i][k] AND x[j][k]
        if self.verbose:
            print("  Building AND-gate encoding...")
        for (i, j) in self.pairs:
            sv = self.s[(i, j)]
            for k in range(self.b):
                s_var, xi, xj = sv[k], self.x[i][k], self.x[j][k]
                solver.add_clause([-s_var, xi])
                solver.add_clause([-s_var, xj])
                solver.add_clause([-xi, -xj, s_var])

        # (C) Symmetry breaking
        if self.verbose:
            print("  Building symmetry breaking...")
        self._add_symmetry_breaking(solver)

        # (D) Totalizers for all pairs
        if self.verbose:
            print("  Building Totalizers for all pairs...")
        o_vars = {}
        var_counter = [self._next_var]
        tot_clauses = []
        for (i, j) in self.pairs:
            pair_cls = []
            out = _build_totalizer_manual(list(self.s[(i, j)]), var_counter, pair_cls)
            tot_clauses.extend(pair_cls)
            o_vars[(i, j)] = out
        self._next_var = var_counter[0]
        for cl in tot_clauses:
            solver.add_clause(cl)
        if self.verbose:
            print(f"    Totalizer clauses: {len(tot_clauses)}")

        # (E) Reified bound clauses
        #   b_M => (D_{ij} <= M)  i.e.  b_M => NOT o[M]
        #   CNF: (-b_M  OR  -o[M])    (o[M]=1 means D>=M+1)
        if self.verbose:
            print("  Building reified bound clauses...")
        reif_cnt = 0
        for M in range(self.r + 1):
            bM = self.b_lit[M]
            for (i, j) in self.pairs:
                out = o_vars[(i, j)]
                if M < len(out):
                    solver.add_clause([-bM, -out[M]])
                    reif_cnt += 1
            if M < self.r:
                solver.add_clause([-bM, self.b_lit[M + 1]])
                reif_cnt += 1
        if self.verbose:
            print(f"    Reified + monotonicity clauses: {reif_cnt}")
            print(f"  Total variables: {self._next_var - 1}")

        return solver, o_vars

    def _add_symmetry_breaking(self, solver):
        """
        Add symmetry breaking constraints:
        - Fix first row to have 1s in first r positions
        """
        # Fix first row: first r credits are selected
        for k in range(self.r):
            solver.add_clause([self.x[0][k]])
        for k in range(self.r, self.b):
            solver.add_clause([-self.x[0][k]])
        if self.verbose:
            print("    Added symmetry breaking constraints (fixed first row)")

    # ------------------------------------------------------------------
    # Solve: UB -> LB, assumption-based
    # ------------------------------------------------------------------

    def solve(self):
        """
        Assumption-based Incremental SAT: search M from r (UB) down to lb (LB).

          - SAT   at M  -> update best solution; try M-1
          - UNSAT at M  -> optimal proven = last SAT M; stop

        Returns: (optimal_M, matrix, total_time)
        """
        lb = self.compute_lower_bound()
        if self.verbose:
            print(f"\n  Lower bound: {lb},  Upper bound: {self.r}")
            print("\nBuilding formula (once)...")

        t0 = time.time()
        solver, o_vars = self.build_formula()
        build_time = time.time() - t0
        if self.verbose:
            print(f"Build time: {build_time:.3f}s")
            print(f"\nIncremental search (UB -> LB): M = {self.r} .. {lb}")

        best_M      = self.r + 1   # sentinel
        best_matrix = None
        total_solve = 0.0

        for M in range(self.r, lb - 1, -1):
            bM = self.b_lit[M]

            if self.verbose:
                print(f"  M = {M:3d} ...", end=" ", flush=True)

            t_iter = time.time()
            sat = solver.solve(assumptions=[bM])
            if sat:
                model = solver.get_model()

            elapsed = time.time() - t_iter
            total_solve += elapsed

            if sat:
                matrix  = self._extract_matrix(model)
                actual  = self._compute_max_overlap(matrix)
                best_M  = actual
                best_matrix = matrix
                if self.verbose:
                    print(f"SAT  (actual lambda = {actual})  [{elapsed:.3f}s]"
                          f"  -> try M = {M - 1}")
            else:
                if self.verbose:
                    print(f"UNSAT  [{elapsed:.3f}s]  -> optimal proven = {best_M}")
                break

        solver.delete()
        return best_M, best_matrix, build_time + total_solve

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _extract_matrix(self, model):
        mset = set(model)
        return [[1 if self.x[i][k] in mset else 0 for k in range(self.b)]
                for i in range(self.v)]

    def _compute_max_overlap(self, matrix):
        max_ov = 0
        for (i, j) in self.pairs:
            ov = sum(matrix[i][k] * matrix[j][k] for k in range(self.b))
            if ov > max_ov:
                max_ov = ov
        return max_ov

    def verify_solution(self, matrix, claimed_M):
        """Returns (errors_list, actual_max_overlap)."""
        errors = []
        for i in range(self.v):
            s = sum(matrix[i])
            if s != self.r:
                errors.append(f"Row {i}: sum={s}, expected {self.r}")
        actual = self._compute_max_overlap(matrix)
        if actual > claimed_M:
            errors.append(f"Max overlap={actual} > claimed M={claimed_M}")
        return errors, actual

    def print_matrix(self, matrix):
        print("\nSolution matrix:")
        for i, row in enumerate(matrix):
            print(f"  Row {i:3d}: {''.join(str(c) for c in row)}  (sum={sum(row)})")

    def print_overlap_matrix(self, matrix):
        print("\nPairwise overlap matrix:")
        print("     " + "".join(f"{j:4d}" for j in range(self.v)))
        for i in range(self.v):
            row_str = f"  {i:2d} "
            for j in range(self.v):
                if i == j:
                    row_str += "   -"
                else:
                    ov = sum(matrix[i][k] * matrix[j][k] for k in range(self.b))
                    row_str += f"{ov:4d}"
            print(row_str)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def parse_input_file(filepath):
    """Parse v, b, r from OPD input file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        vm = re.search(r'v\s*=\s*(\d+)', content)
        bm = re.search(r'b\s*=\s*(\d+)', content)
        rm = re.search(r'r\s*=\s*(\d+)', content)
        if vm and bm and rm:
            return {'v': int(vm.group(1)), 'b': int(bm.group(1)), 'r': int(rm.group(1))}
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    return None


import multiprocessing
import sys

def _solve_worker(v, b, r, quiet, queue):
    try:
        odp = OpdReifiedBound(v, b, r, verbose=not quiet)
        optimal_M, matrix, total_time = odp.solve()
        
        errors = []
        actual = optimal_M
        if matrix is not None:
            errors, actual = odp.verify_solution(matrix, optimal_M)
            
        queue.put({
            'optimal_lambda': optimal_M,
            'matrix': matrix,
            'total_time': total_time,
            'status': 'Verification Failed' if errors else ('Solved' if matrix is not None else 'No Solution'),
            'errors': errors,
            'actual': actual
        })
    except Exception as e:
        queue.put({'error': str(e)})

def solve_from_file(filepath, overall_timeout=600, quiet=False):
    """Read params from file and solve. Returns result dict."""
    filename = os.path.basename(filepath)
    if not quiet:
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")

    params = parse_input_file(filepath)
    if not params:
        return {'File': filename, 'v': None, 'b': None, 'r': None,
                'Optimal Lambda': None, 'Time (s)': 0, 'Status': 'Parse Error'}

    v, b, r = params['v'], params['b'], params['r']
    if not quiet:
        print(f"Parameters: v={v}, b={b}, r={r}")

    queue = multiprocessing.Queue()
    start_time = time.time()

    p = multiprocessing.Process(target=_solve_worker, args=(v, b, r, quiet, queue))
    p.start()

    try:
        p.join(timeout=overall_timeout)
        
        elapsed = time.time() - start_time
        
        if p.is_alive():
            if not quiet: print(f"\nTIMEOUT strictly enforced after {overall_timeout} seconds.")
            p.terminate()
            p.join()
            return {
                'File': filename, 'v': v, 'b': b, 'r': r,
                'Optimal Lambda': None, 'Time (s)': round(elapsed, 3), 'Status': 'Timeout'
            }
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Đã tiếp nhận yêu cầu dừng từ người dùng. Đang hủy tiến trình...")
        p.terminate()
        p.join()
        sys.exit(1)
        
    if not queue.empty():
        result = queue.get()
        if 'error' in result:
            if not quiet: print(f"Solver error: {result['error']}")
            return {
                'File': filename, 'v': v, 'b': b, 'r': r,
                'Optimal Lambda': None, 'Time (s)': 0, 'Status': f"Error: {result['error']}"
            }
            
        optimal_M = result['optimal_lambda']
        matrix = result['matrix']
        total_time = result['total_time']
        status = result['status']
        
        if status == 'Verification Failed' and not quiet:
            for e in result['errors']:
                print(f"  VERIFY ERROR: {e}")
        elif matrix is not None and not quiet:
            actual = result['actual']
            lb = max(0, math.ceil(r * (r * v - b) / (b * (v - 1)))) if v > 1 else 0
            print(f"\nRESULT:")
            print(f"  Optimal lambda = {optimal_M}")
            print(f"  Verified max overlap = {actual}")
            print(f"  Total time = {total_time:.3f}s")
            if optimal_M == lb:
                print(f"  ✓ Optimal! (matches lower bound {lb})")
            else:
                print(f"  Gap from lower bound: {optimal_M - lb}")
        elif matrix is None and not quiet:
            print(f"\nBest M = {optimal_M}, Time = {total_time:.3f}s")

        return {'File': filename, 'v': v, 'b': b, 'r': r,
                'Optimal Lambda': optimal_M,
                'Time (s)': round(total_time, 3),
                'Status': status}
    else:
        if not quiet: print(f"Process crashed or returned no result.")
        return {
            'File': filename, 'v': v, 'b': b, 'r': r,
            'Optimal Lambda': None, 'Time (s)': round(elapsed, 3), 'Status': 'Crashed'
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OPD Solver - Reified Bound (then-activate) Incremental SAT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy: Build once, activate gradually (UB -> LB search).
  1. Build ALL Totalizers and reified literals b_M once.
  2. Search: assume b_M for M = r, r-1, ..., lb.
     SAT   -> record solution, try M-1.
     UNSAT -> optimal proven = last SAT M; stop.

Examples:
  python Opd_reified_bound_sat.py --input input/small/small_1.txt
  python Opd_reified_bound_sat.py --input input/small
  python Opd_reified_bound_sat.py --input input/small --timeout 120 --quiet
        """,
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input file or directory of .txt files')
    parser.add_argument('--overall_timeout', type=int, default=3600,
                        help='Allowed Execution Time in seconds (default: 600)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()
    input_path = args.input

    if not os.path.exists(input_path):
        alt = os.path.join('input', input_path)
        if os.path.exists(alt):
            input_path = alt

    COLUMNS = ['File', 'v', 'b', 'r', 'Optimal Lambda', 'Time (s)', 'Status']

    if os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        files = sorted(f for f in os.listdir(input_path) if f.endswith('.txt'))
        if not files:
            print("No .txt files found.")
        else:
            print(f"Found {len(files)} file(s).\n")
            try:
                import openpyxl; has_xlsx = True
            except ImportError:
                print("Warning: openpyxl not found - Excel export disabled.")
                has_xlsx = False

            folder_name = os.path.basename(os.path.normpath(input_path))
            script_name = os.path.splitext(os.path.basename(__file__))[0]
            output_xlsx = f"result_{folder_name}_{script_name}.xlsx"

            for fname in files:
                res = solve_from_file(os.path.join(input_path, fname),
                                      overall_timeout=args.overall_timeout, quiet=args.quiet)
                if res and has_xlsx:
                    if not os.path.exists(output_xlsx):
                        wb = openpyxl.Workbook(); ws = wb.active
                        ws.append(COLUMNS)
                    else:
                        wb = openpyxl.load_workbook(output_xlsx); ws = wb.active
                    ws.append([res.get(c) for c in COLUMNS])
                    wb.save(output_xlsx)

            print(f"\n{'='*60}")
            print(f"Done. Results saved to: {output_xlsx}" if has_xlsx else "Done.")

    elif os.path.isfile(input_path):
        solve_from_file(input_path, overall_timeout=args.overall_timeout, quiet=args.quiet)
    else:
        print(f"Error: '{args.input}' not found.")
