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

from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
import argparse
import itertools
import math
import os
import re
import time
import multiprocessing
import sys

ENCODING_MAP = {
    'sortnetwrk': EncType.sortnetwrk,
    'seqcounter': EncType.seqcounter,
    'totalizer': EncType.totalizer,
    'mtotalizer': EncType.mtotalizer,
    'pairwise': EncType.pairwise
}
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

    def __init__(self, v, b, r, solver_name='cadical195', encoding_name='sortnetwrk', sym_options=None, verbose=True):
        self.v = v
        self.b = b
        self.r = r
        self.solver_name = solver_name
        self.encoding_name = encoding_name
        self.encoding_type = ENCODING_MAP.get(encoding_name, EncType.sortnetwrk)
        self.sym_options = sym_options or []
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
            print(f"  Solver: {self.solver_name} | Encoding: {self.encoding_name}")
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
        solver = Solver(name=self.solver_name)

        # (A) Row cardinality: sum_k x[i][k] = r
        if self.verbose:
            print("  Building row cardinality constraints...")
        for i in range(self.v):
            cnf = CardEnc.equals(
                lits=self.x[i],
                bound=self.r,
                top_id=self._next_var - 1,
                encoding=self.encoding_type,
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

    def _add_lex_constraint(self, solver, a, b_vec, n):
        """Add SAT encoding of a <=_lex b_vec (binary vectors of length n)."""
        solver.add_clause([-a[0], b_vec[0]])
        prev_p = None

        for k in range(n - 1):
            p_k = self._next_var
            self._next_var += 1

            if prev_p is None:
                solver.add_clause([-p_k, -a[0], b_vec[0]])
                solver.add_clause([-p_k, a[0], -b_vec[0]])
                solver.add_clause([-a[0], -b_vec[0], p_k])
                solver.add_clause([a[0], b_vec[0], p_k])
            else:
                solver.add_clause([-p_k, prev_p])
                solver.add_clause([-p_k, -a[k], b_vec[k]])
                solver.add_clause([-p_k, a[k], -b_vec[k]])
                solver.add_clause([-prev_p, -a[k], -b_vec[k], p_k])
                solver.add_clause([-prev_p, a[k], b_vec[k], p_k])

            solver.add_clause([-p_k, -a[k + 1], b_vec[k + 1]])
            prev_p = p_k

    def _add_symmetry_breaking(self, solver):
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
                    self._add_lex_constraint(solver, self.x[i + 1], self.x[i], self.b)
                else:
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
            print(f"    Added symmetry breaking: {self.applied_sym}")

    # ------------------------------------------------------------------
    # Solve: UB -> LB, assumption-based
    # ------------------------------------------------------------------

    def solve(self):
        """
        Assumption-based Incremental SAT with BINARY SEARCH on M.

        Binary search over [lb, r]:
          lo = lb,  hi = r,  best_M = r+1 (sentinel)
          while lo <= hi:
              mid = (lo + hi) // 2
              assume b_mid:
                SAT   -> best_M = actual_M; hi = actual_M - 1  (go lower)
                UNSAT -> lo = mid + 1                           (go higher)

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
            print(f"\nBinary search: M in [{lb}, {self.r}]")

        best_M      = self.r + 1   # sentinel: no solution yet
        best_matrix = None
        total_solve = 0.0
        iteration   = 0

        lo, hi = lb, self.r

        while lo <= hi:
            mid = (lo + hi) // 2
            bM  = self.b_lit[mid]
            iteration += 1

            if self.verbose:
                print(f"  [{iteration}] lo={lo} hi={hi}  mid={mid} ...",
                      end=" ", flush=True)

            t_iter = time.time()
            sat = solver.solve(assumptions=[bM])
            if sat:
                model = solver.get_model()

            elapsed = time.time() - t_iter
            total_solve += elapsed

            if sat:
                matrix = self._extract_matrix(model)
                actual = self._compute_max_overlap(matrix)
                best_M = actual
                best_matrix = matrix
                if self.verbose:
                    print(f"SAT  (actual lambda={actual})  [{elapsed:.3f}s]"
                          f"  -> hi = {actual - 1}")
                hi = actual - 1   # try to find a better (lower) M
            else:
                if self.verbose:
                    print(f"UNSAT  [{elapsed:.3f}s]"
                          f"  -> lo = {mid + 1}")
                lo = mid + 1      # M must be at least mid+1

        if self.verbose and best_matrix is not None:
            print(f"\n  Binary search done in {iteration} iteration(s)."
                  f"  Optimal proven = {best_M}")

        n_vars = self._next_var - 1
        n_clauses = solver.nof_clauses()
        solver.delete()
        return best_M, best_matrix, build_time + total_solve, n_vars, n_clauses

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


def _solve_worker(v, b, r, solver_name, encoding_name, sym_options, quiet, queue):
    try:
        t_start = time.time()
        odp = OpdReifiedBound(v, b, r, solver_name=solver_name, encoding_name=encoding_name, sym_options=sym_options, verbose=not quiet)
        optimal_M, matrix, _, n_vars, n_clauses = odp.solve()
        total_time = time.time() - t_start
        
        errors = []
        actual = optimal_M
        if matrix is not None:
            errors, actual = odp.verify_solution(matrix, optimal_M)
            
        queue.put({
            'optimal_lambda': optimal_M,
            'matrix': matrix,
            'total_time': total_time,
            'n_vars': n_vars,
            'n_clauses': n_clauses,
            'applied_sym': getattr(odp, 'applied_sym', 'none'),
            'status': 'Verification Failed' if errors else ('Solved' if matrix is not None else 'No Solution'),
            'errors': errors,
            'actual': actual
        })
    except Exception as e:
        queue.put({'error': str(e)})

def solve_from_file(filepath, solver_name='cadical195', encoding_name='sortnetwrk', sym_options=None, timeout=600, quiet=False):
    """Read params from file and solve. Returns result dict."""
    filename = os.path.basename(filepath)
    if not quiet:
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")

    params = parse_input_file(filepath)
    if not params:
        return {'File': filename, 'v': None, 'b': None, 'r': None,
                'Solver': solver_name, 'Encoding': encoding_name,
                'Optimal Lambda': None, 'Variables': 0, 'Clauses': 0, 'Sym Method': 'none', 'Time (s)': 0, 'Status': 'Parse Error'}

    v, b, r = params['v'], params['b'], params['r']
    if not quiet:
        print(f"Parameters: v={v}, b={b}, r={r}")

    queue = multiprocessing.Queue()
    start_time = time.time()

    p = multiprocessing.Process(target=_solve_worker, args=(v, b, r, solver_name, encoding_name, sym_options, quiet, queue))
    p.start()

    try:
        p.join(timeout=timeout)
        
        elapsed = time.time() - start_time
        
        if p.is_alive():
            if not quiet: print(f"\nTIMEOUT strictly enforced after {timeout} seconds.")
            p.terminate()
            p.join()
            return {
                'File': filename, 'v': v, 'b': b, 'r': r,
                'Solver': solver_name, 'Encoding': encoding_name,
                'Optimal Lambda': None, 'Variables': 0, 'Clauses': 0, 'Sym Method': "+".join(sym_options) if sym_options else 'none', 'Time (s)': round(elapsed, 3), 'Status': 'Timeout'
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
                'Solver': solver_name, 'Encoding': encoding_name,
                'Optimal Lambda': None, 'Variables': 0, 'Clauses': 0, 'Sym Method': 'none', 'Time (s)': 0, 'Status': f"Error: {result['error']}"
            }
            
        optimal_M = result['optimal_lambda']
        matrix = result['matrix']
        total_time = elapsed
        status = result['status']
        n_vars = result.get('n_vars', 0)
        n_clauses = result.get('n_clauses', 0)
        applied_sym = result.get('applied_sym', 'none')
        
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
            print(f"  Solver: {solver_name}, Encoding: {encoding_name}")
            print(f"  Variables: {n_vars}, Clauses: {n_clauses}, Sym Method: {applied_sym}")
            if optimal_M == lb:
                print(f"  ✓ Optimal! (matches lower bound {lb})")
            else:
                print(f"  Gap from lower bound: {optimal_M - lb}")
        elif matrix is None and not quiet:
            print(f"\nBest M = {optimal_M}, Time = {total_time:.3f}s")

        return {'File': filename, 'v': v, 'b': b, 'r': r,
                'Solver': solver_name, 'Encoding': encoding_name,
                'Optimal Lambda': optimal_M, 'Variables': n_vars, 'Clauses': n_clauses, 'Sym Method': applied_sym,
                'Time (s)': round(total_time, 3),
                'Status': status}
    else:
        if not quiet: print(f"Process crashed or returned no result.")
        return {
            'File': filename, 'v': v, 'b': b, 'r': r,
            'Solver': solver_name, 'Encoding': encoding_name,
            'Optimal Lambda': None, 'Variables': 0, 'Clauses': 0, 'Sym Method': 'none', 'Time (s)': round(elapsed, 3), 'Status': 'Crashed'
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
  python Opd_incremental_sat_ver2_binary_sym.py --input input/small/small_1.txt
  python Opd_incremental_sat_ver2_binary_sym.py --input input/small --sym lex_row lex_col r --timeout 300
  python Opd_incremental_sat_ver2_binary_sym.py --input input/small --solver glucose4 --encoding seqcounter
        """,
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input file or directory of .txt files')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Allowed Execution Time in seconds (default: 600)')
    parser.add_argument('--solver', type=str, default='cadical195', choices=['cadical195', 'glucose3', 'glucose4', 'minisat22', 'lingeling'],
                        help='SAT solver to use')
    parser.add_argument('--encoding', type=str, default='sortnetwrk', choices=['sortnetwrk', 'seqcounter', 'totalizer', 'mtotalizer', 'pairwise'],
                        help='Cardinality encoding method')
    parser.add_argument('--sym', nargs='+', default=[],
                        help='Symmetry breaking methods (e.g., lex_row lex_col r)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()
    input_path = args.input

    if not os.path.exists(input_path):
        alt = os.path.join('input', input_path)
        if os.path.exists(alt):
            input_path = alt

    COLUMNS = ['File', 'v', 'b', 'r', 'Solver', 'Encoding', 'Optimal Lambda', 'Variables', 'Clauses', 'Sym Method', 'Time (s)', 'Status']

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
                                      solver_name=args.solver, encoding_name=args.encoding,
                                      sym_options=args.sym, timeout=args.timeout, quiet=args.quiet)
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
        solve_from_file(input_path, solver_name=args.solver, encoding_name=args.encoding,
                        sym_options=args.sym, timeout=args.timeout, quiet=args.quiet)
    else:
        print(f"Error: '{args.input}' not found.")
