#!/usr/bin/env python3
"""
Portfolio Optimisation - Weighted MaxSAT với Threshold Variables (Exact Min-Max)

==============================================================================
BÀI TOÁN
==============================================================================
Cho bộ tham số (v, b, r):
  - v : số sub-pool (hàng của ma trận nhị phân)
  - b : số tài sản (cột của ma trận nhị phân)
  - r : kích thước mỗi sub-pool (tổng mỗi hàng = r)

Ma trận quyết định: x[i][j] ∈ {0,1}, i ∈ [1..v], j ∈ [1..b]

Mục tiêu: Tìm ma trận x sao cho:
  (1) Tổng mỗi hàng = r  (ràng buộc kích thước sub-pool)
  (2) Tối thiểu hoá λ = max_{i<k} Σ_j x[i][j]·x[k][j]
      (tối thiểu hoá độ chồng lấp lớn nhất giữa mọi cặp sub-pool)

Chặn dưới lý thuyết cho λ:
  λ ≥ ⌈ r(rv - b) / (b(v-1)) ⌉  (≥ 0)

==============================================================================
PHƯƠNG PHÁP - WEIGHTED MAXSAT VỚI THRESHOLD VARIABLES
==============================================================================
Bài toán min-max được mã hoá thành một bài toán Weighted Partial MaxSAT:

1. BIẾN QUYẾT ĐỊNH
   - x_{i,j}: biến SAT chính, biểu diễn x[i][j] ∈ {0,1}.
   - y_{i,k,j}: biến phụ, y_{i,k,j} = x[i][j] AND x[k][j]
     (tích phần tử giữa hàng i và hàng k tại cột j).
   - T_m: biến ngưỡng (threshold) cho từng giá trị m ∈ [lb, ub).
     T_m = 1 ⟺ "cho phép λ > m" (tức là vi phạm ràng buộc overlap ≤ m).

2. MỆNH ĐỀ CỨNG (Hard Clauses)
   - Ràng buộc kích thước sub-pool:
       Σ_j x[i][j] = r, ∀i  (mã hoá bằng CardEnc.equals)
   - Ràng buộc tích phần tử:
       y_{i,k,j} ← x[i][j] ∧ x[k][j]  ∀i<k, ∀j
   - Ràng buộc ngưỡng:
       Nếu T_m = 0 (không vi phạm ngưỡng m) thì
       Σ_j y_{i,k,j} ≤ m  ∀i<k  (mã hoá bằng CardEnc.atmost)
       Tức là: (¬T_m ∧ overlap_{i,k} > m) không thể đồng thời xảy ra.
   - Ràng buộc đối xứng (Symmetry Breaking) — tuỳ chọn:
       * fixed_r  : cố định hàng 0 = [1..1, 0..0] (r số 1 đầu tiên)
       * lex_row  : x[i] ≤_lex x[i+1]  (thứ tự lexicographic tăng giữa hàng liên tiếp)
       * lex_col  : col[j] ≤_lex col[j+1]  (thứ tự lexicographic tăng giữa cột liên tiếp)

3. MỆNH ĐỀ MỀM (Soft Clauses)
   - ¬T_m  với trọng số 1, ∀m ∈ [lb, ub)
   - MaxSAT sẽ cố gắng đặt tất cả T_m = 0 (không vi phạm ngưỡng nào).
   - T_m = 1 (vi phạm mệnh đề mềm) → chi phí +1.
   - Chi phí tối ưu = λ* - lb (số ngưỡng bị vi phạm).
   - Suy ra λ* = lb + cost_optimal.

4. SOLVER (tuỳ chọn qua --engine)
   - rc2       : RC2 (PySAT) — MaxSAT solver chạy trong tiến trình Python.
   - evalmaxsat: EvalMaxSAT binary — gọi qua subprocess với file WCNF tạm.

5. CARDINALITY ENCODING (tuỳ chọn qua --card-enc)
   - seqcounter, totalizer, cardnetwrk, sortnetwrk, kmtotalizer

==============================================================================
SỬ DỤNG
==============================================================================
  python Opd_EvalMaxSAT_ver3.py --input input/small
  python Opd_EvalMaxSAT_ver3.py --input input/small/small_2.txt
  python Opd_EvalMaxSAT_ver3.py --engine evalmaxsat --input input/small/small_2.txt
  python Opd_EvalMaxSAT_ver3.py --input input/small --sym lex_row lex_col
  python Opd_EvalMaxSAT_ver3.py --input input/small --card-enc totalizer --timeout 600
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from pysat.card import CardEnc, EncType

@dataclass
class SolveResult:
    model: Optional[List[int]]
    cost: Optional[int]
    solve_time_s: float
    status: str

def parse_evalmaxsat_output(stdout: str, nvars: int) -> Tuple[Optional[List[int]], Optional[int]]:
    cost = None
    assign: Dict[int, int] = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('o '):
            try:
                cost = int(line[2:].strip())
            except ValueError:
                pass
            continue
        if not line.startswith('v '):
            continue
        payload = line[2:].strip()
        if not payload:
            continue
        toks = payload.split()
        if len(toks) == nvars and all(t in ('0', '1') for t in toks):
            for i, bit in enumerate(toks, start=1):
                assign[i] = i if bit == '1' else -i
            continue
        if len(toks) == 1 and re.fullmatch(r'[01]+', toks[0]) and len(toks[0]) == nvars:
            for i, bit in enumerate(toks[0], start=1):
                assign[i] = i if bit == '1' else -i
            continue
        for t in toks:
            try:
                lit = int(t)
            except ValueError:
                continue
            if lit == 0:
                continue
            assign[abs(lit)] = lit
    model = list(assign.values()) if assign else None
    return model, cost

class PortfolioExactEvalMaxSAT:
    def __init__(
        self,
        v: int,
        b: int,
        r: int,
        *,
        verbose: bool = True,
        sym_options: Optional[List[str]] = None,
        card_encoding: str = "seqcounter",
    ):
        self.v = int(v)
        self.b = int(b)
        self.r = int(r)
        self.verbose = verbose
        self.sym_options = sym_options or []

        if self.v <= 0 or self.b <= 0 or self.r < 0:
            raise ValueError("Invalid parameters: require v>0, b>0, r>=0")
        if self.r > self.b:
            raise ValueError(f"Infeasible: r={self.r} cannot exceed b={self.b}")

        self.var_id = lambda i, j: (i - 1) * self.b + j
        self.total_vars = self.v * self.b
        self.next_var_id = self.total_vars + 1

        self.lb = self.calculate_lower_bound()
        self.upper_bound = min(self.r, self.lb + 5)

        enc_map = {
            "seqcounter": EncType.seqcounter,
            "totalizer": EncType.totalizer,
            "cardnetwrk": EncType.cardnetwrk,
            "sortnetwrk": EncType.sortnetwrk,
            "kmtotalizer": EncType.kmtotalizer,
        }
        if card_encoding not in enc_map:
            raise ValueError(f"Unknown card_encoding '{card_encoding}'. Choose from: {sorted(enc_map)}")
        self.eq_encoding = enc_map[card_encoding]

        self.wcnf = WCNF()
        
        self.y_vars = {}
        self.T_vars = {}

        self._add_hard_pool_size_constraints()
        self._add_symmetry_breaking()

        if self.v >= 2 and self.r >= 1:
            self._add_min_max_constraints()
        else:
            if self.verbose:
                print("\nTrivial instance: v<2 or r<1 => λ = 0")

    def _new_var(self) -> int:
        vid = self.next_var_id
        self.next_var_id += 1
        return vid

    def _add_hard_pool_size_constraints(self) -> None:
        if self.verbose:
            print("\n=== Adding Hard Clauses: Pool Sizes ===")
        for i in range(1, self.v + 1):
            pool_vars = [self.var_id(i, j) for j in range(1, self.b + 1)]
            cnf = CardEnc.equals(
                lits=pool_vars,
                bound=self.r,
                encoding=self.eq_encoding,
                top_id=self.next_var_id - 1,
            )
            for cl in cnf.clauses:
                self.wcnf.append(cl)
            self.next_var_id = max(self.next_var_id, cnf.nv + 1)

    def _add_lex_constraint(self, a: List[int], b_vec: List[int], n: int) -> None:
        self.wcnf.append([-a[0], b_vec[0]])
        prev_p = None
        for k in range(n - 1):
            p_k = self._new_var()
            if prev_p is None:
                self.wcnf.append([-p_k, -a[0], b_vec[0]])
                self.wcnf.append([-p_k, a[0], -b_vec[0]])
                self.wcnf.append([-a[0], -b_vec[0], p_k])
                self.wcnf.append([a[0], b_vec[0], p_k])
            else:
                self.wcnf.append([-p_k, prev_p])
                self.wcnf.append([-p_k, -a[k], b_vec[k]])
                self.wcnf.append([-p_k, a[k], -b_vec[k]])
                self.wcnf.append([-prev_p, -a[k], -b_vec[k], p_k])
                self.wcnf.append([-prev_p, a[k], b_vec[k], p_k])
            self.wcnf.append([-p_k, -a[k + 1], b_vec[k + 1]])
            prev_p = p_k

    def _add_symmetry_breaking(self) -> None:
        fixed_r = 'r' in self.sym_options
        use_lex_row = 'lex_row' in self.sym_options
        use_lex_col = 'lex_col' in self.sym_options
        sym_methods = []

        if fixed_r:
            for j in range(1, self.r + 1):
                self.wcnf.append([self.var_id(1, j)])
            for j in range(self.r + 1, self.b + 1):
                self.wcnf.append([-self.var_id(1, j)])
            sym_methods.append("fixed_r")

        if use_lex_row:
            sym_methods.append("lex_row")
            for i in range(1, self.v):
                row_i = [self.var_id(i, j) for j in range(1, self.b + 1)]
                row_i_plus_1 = [self.var_id(i + 1, j) for j in range(1, self.b + 1)]
                if fixed_r:
                    self._add_lex_constraint(row_i_plus_1, row_i, self.b)
                else:
                    self._add_lex_constraint(row_i, row_i_plus_1, self.b)

        if use_lex_col:
            sym_methods.append("lex_col")
            for j in range(1, self.b):
                col_j = [self.var_id(i, j) for i in range(1, self.v + 1)]
                col_j1 = [self.var_id(i, j + 1) for i in range(1, self.v + 1)]
                if fixed_r:
                    self._add_lex_constraint(col_j1, col_j, self.v)
                else:
                    self._add_lex_constraint(col_j, col_j1, self.v)

        self.applied_sym = "+".join(sym_methods) if sym_methods else "none"

    def _add_min_max_constraints(self) -> None:
        if self.verbose:
            print("\n=== Adding Min-Max Threshold Constraints ===")
            print(f"Theoretical Lower Bound λ >= {self.lb}")
            print(f"Search range for λ (Threshold Variables): [{self.lb}, {self.upper_bound})")

        for i, k in itertools.combinations(range(1, self.v + 1), 2):
            for j in range(1, self.b + 1):
                y = self._new_var()
                self.y_vars[(i, k, j)] = y
                self.wcnf.append([-self.var_id(i, j), -self.var_id(k, j), y])

        for m in range(self.lb, self.upper_bound):
            if m not in self.T_vars:
                self.T_vars[m] = self._new_var()
                
            T_m = self.T_vars[m]
            
            for i, k in itertools.combinations(range(1, self.v + 1), 2):
                y_lits = [self.y_vars[(i, k, j)] for j in range(1, self.b + 1)]
                cnf = CardEnc.atmost(
                    lits=y_lits, 
                    bound=m, 
                    encoding=self.eq_encoding,
                    top_id=self.next_var_id - 1
                )
                self.next_var_id = max(self.next_var_id, cnf.nv + 1)
                
                for clause in cnf.clauses:
                    self.wcnf.append(clause + [T_m])

        for m in range(self.lb, self.upper_bound):
            self.wcnf.append([-self.T_vars[m]], weight=1)

    def solve(
        self,
        *,
        engine: str = "rc2",
        timeout: int = 3600,
        solver_name: str = "g3",
        verbose: bool = True,
    ) -> SolveResult:
        engine = engine.lower().strip()
        start_time = time.time()

        if verbose:
            print("\n=== Solving MaxSAT ===")
            print(f"Engine: {engine}")
            print(f"Vars: {self.next_var_id - 1}")
            print(f"Hard clauses: {len(self.wcnf.hard)}")
            print(f"Soft clauses: {len(self.wcnf.soft)}")

        if engine == "rc2":
            try:
                with RC2(self.wcnf, solver=solver_name, adapt=True, verbose=0) as rc2:
                    model = rc2.compute()
                    cost = rc2.cost
                t = time.time() - start_time
                status = "Solved" if model else "No Solution"
                return SolveResult(model=model, cost=cost, solve_time_s=t, status=status)
            except Exception as e:
                t = time.time() - start_time
                return SolveResult(model=None, cost=None, solve_time_s=t, status=f"Error: {e}")

        if engine == "evalmaxsat":
            wcnf_filename = self.write_wcnf_file()
            solver_path = "/home/anhntv161/KLTN/2. Code/Base/src/EvalMaxSAT/build/main/EvalMaxSAT_bin"
            if not os.path.exists(solver_path):
                t = time.time() - start_time
                return SolveResult(model=None, cost=None, solve_time_s=t, status=f"Error: {solver_path} not found")

            try:
                result = subprocess.run(
                    [solver_path, wcnf_filename],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
                t = time.time() - start_time
                nvars = self.next_var_id - 1
                model, cost = parse_evalmaxsat_output(result.stdout, nvars)
                status = "Solved" if model else "No Solution"
                return SolveResult(model=model, cost=cost, solve_time_s=t, status=status)
            except subprocess.TimeoutExpired:
                t = time.time() - start_time
                return SolveResult(model=None, cost=None, solve_time_s=t, status=f"Timeout after {timeout}s")
            except Exception as e:
                t = time.time() - start_time
                return SolveResult(model=None, cost=None, solve_time_s=t, status=f"Error: {e}")

        t = time.time() - start_time
        return SolveResult(model=None, cost=None, solve_time_s=t, status=f"Unknown engine '{engine}'")

    def write_wcnf_file(self, filename: str = "problem_eval.wcnf") -> str:
        with open(filename, "w", encoding="utf-8") as f:
            total_clauses = len(self.wcnf.hard) + len(self.wcnf.soft)
            weights = getattr(self.wcnf, "wght", [])
            top_weight = (sum(weights) + 1) if weights else (len(self.wcnf.soft) + 1)
            f.write(f"p wcnf {self.next_var_id - 1} {total_clauses} {top_weight}\n")
            for clause in self.wcnf.hard:
                f.write(str(top_weight) + " " + " ".join(map(str, clause)) + " 0\n")
            for idx, clause in enumerate(self.wcnf.soft):
                w = weights[idx] if idx < len(weights) else 1
                f.write(f"{w} " + " ".join(map(str, clause)) + " 0\n")
        return filename

    def decode_solution(self, model: Optional[Sequence[int]]) -> Optional[List[Set[int]]]:
        if not model:
            return None
        model_set = set(model)
        sub_pools: List[Set[int]] = [set() for _ in range(self.v)]
        for i in range(1, self.v + 1):
            pool = sub_pools[i - 1]
            for j in range(1, self.b + 1):
                if self.var_id(i, j) in model_set:
                    pool.add(j)
        return sub_pools

    def calculate_lambda(self, sub_pools: Optional[List[Set[int]]]) -> Optional[int]:
        if sub_pools is None:
            return None
        if self.v < 2:
            return 0
        max_intersection = 0
        for i1, i2 in itertools.combinations(range(self.v), 2):
            max_intersection = max(max_intersection, len(sub_pools[i1] & sub_pools[i2]))
        return max_intersection

    def calculate_lower_bound(self) -> int:
        if self.v <= 1:
            return 0
        numerator = self.r * (self.r * self.v - self.b)
        denominator = self.b * (self.v - 1)
        if denominator == 0:
            return 0
        return max(0, math.ceil(numerator / denominator))

    def print_solution(self, sub_pools: Optional[List[Set[int]]], lambda_val: Optional[int], *, cost: Optional[int] = None) -> None:
        if sub_pools is None:
            print("\nNo solution to print.")
            return

        print("\n" + "=" * 60)
        print("SOLUTION")
        print("=" * 60)

        lb = self.calculate_lower_bound()
        print(f"Maximum intersection size (λ): {lambda_val}")
        print(f"Theoretical lower bound:      {lb}")
        if cost is not None:
            print(f"MaxSAT reported soft cost:    {cost}")

        if lambda_val is not None and lambda_val == lb:
            print("✓ Optimal (matches lower bound)")
        elif lambda_val is not None:
            gap = lambda_val - lb
            pct = (gap / max(1, lb)) * 100.0
            print(f"Gap from lower bound:         {gap} ({pct:.1f}%)")

        print("\nSub-pools:")
        for i, pool in enumerate(sub_pools, 1):
            print(f"  Sub-pool {i:2d} (size {len(pool)}): {sorted(pool)}")

        ok = True
        for i, pool in enumerate(sub_pools, 1):
            if len(pool) != self.r:
                ok = False
                print(f"  ✗ Size violation: pool {i} has {len(pool)} instead of {self.r}")
        if ok:
            print(f"\n✓ All sub-pools have correct size r={self.r}")

        print("\nIntersection matrix (preview):")
        nshow = min(self.v, 10)
        print("     " + "".join(f"{i:4d}" for i in range(1, nshow + 1)) + (" ..." if self.v > nshow else ""))
        for i1 in range(nshow):
            row = [("  -" if i1 == i2 else f"{len(sub_pools[i1] & sub_pools[i2]):4d}") for i2 in range(nshow)]
            print(f"{i1+1:3d} " + "".join(row) + (" ..." if self.v > nshow else ""))

        if self.v > nshow:
            print("  ... (matrix truncated)")
        print("=" * 60)

def parse_input_file(filepath: str) -> Optional[Dict[str, int]]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        def grab(name: str) -> Optional[int]:
            m = re.search(rf"{name}\s*[:=]\s*(\d+)", content)
            return int(m.group(1)) if m else None

        v = grab("v")
        b = grab("b")
        r = grab("r")

        if v is None or b is None or r is None:
            return None
        return {"v": v, "b": b, "r": r}
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
        return None

def solve_from_file(
    filepath: str,
    *,
    engine: str,
    timeout: int,
    solver_name: str,
    quiet: bool,
    sym_options: List[str],
    card_encoding: str,
) -> Dict[str, object]:
    start_time = time.time()
    filename = os.path.basename(filepath)
    print("\n" + "=" * 60)
    print(f"Processing file: {filename}")
    print("=" * 60)

    params = parse_input_file(filepath)
    if not params:
        print("Error: Could not extract v, b, r from file.")
        return {
            "File": filename,
            "v": None,
            "b": None,
            "r": None,
            "Engine": engine,
            "Encoding": card_encoding,
            "Optimal Lambda": None,
            "Variables": 0,
            "Clauses": 0,
            "Sym Method": "none",
            "Time (s)": 0.0,
            "Status": "Parse Error",
        }

    v, b, r = params["v"], params["b"], params["r"]
    if not quiet:
        print(f"Parameters: v={v}, b={b}, r={r}")

    try:
        problem = PortfolioExactEvalMaxSAT(v, b, r, verbose=not quiet, sym_options=sym_options, card_encoding=card_encoding)
        
        remaining_timeout = max(0, timeout - (time.time() - start_time))
        sol = problem.solve(engine=engine, timeout=remaining_timeout, solver_name=solver_name, verbose=not quiet)
        elapsed = time.time() - start_time

        sub_pools = problem.decode_solution(sol.model)
        lambda_val = problem.calculate_lambda(sub_pools)
        best_lambda = lambda_val if lambda_val is not None else sol.cost
        
        n_vars = problem.next_var_id - 1
        n_clauses = len(problem.wcnf.hard) + len(problem.wcnf.soft)
        applied_sym = getattr(problem, 'applied_sym', 'none')

        status = sol.status
        if sub_pools is not None:
            for pool in sub_pools:
                if len(pool) != r:
                    status = "Invalid (size)"
                    break

        result_data = {
            "File": filename,
            "v": v,
            "b": b,
            "r": r,
            "Engine": engine,
            "Encoding": card_encoding,
            "Optimal Lambda": best_lambda,
            "Variables": n_vars,
            "Clauses": n_clauses,
            "Sym Method": applied_sym,
            "Time (s)": round(elapsed, 3),
            "Status": status,
        }

        if sub_pools is not None and not quiet:
            print(f"\nRESULT for {filename}:")
            print(f"Optimal lambda: {lambda_val}")
            print(f"Total time:     {elapsed:.3f}s")
            print(f"Engine: {engine}, Encoding: {card_encoding}")
            print(f"Variables: {n_vars}, Clauses: {n_clauses}, Sym Method: {applied_sym}")
            problem.print_solution(sub_pools, lambda_val, cost=sol.cost)

        elif sub_pools is None and not quiet:
            print(f"\nNo solution found for {filename}")

        return result_data

    except Exception as e:
        import traceback
        print(f"Solver error: {e}")
        traceback.print_exc()
        elapsed = time.time() - start_time
        return {
            "File": filename,
            "v": v,
            "b": b,
            "r": r,
            "Engine": engine,
            "Encoding": card_encoding,
            "Optimal Lambda": None,
            "Variables": n_vars if 'n_vars' in locals() else 0,
            "Clauses": n_clauses if 'n_clauses' in locals() else 0,
            "Sym Method": applied_sym if 'applied_sym' in locals() else "none",
            "Time (s)": round(elapsed, 3),
            "Status": f"Error: {str(e)}",
        }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact Min-Max Portfolio Optimization using Threshold variables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  python Opd_EvalMaxSAT_ver3.py --input input/small/small_2.txt
  python Opd_EvalMaxSAT_ver3.py --engine evalmaxsat --input input/small/small_2.txt
  python Opd_EvalMaxSAT_ver3.py --input input/small
"""
    )
    parser.add_argument("--input", type=str, required=True, help="Input file or directory path")
    parser.add_argument("--engine", type=str, default="rc2", choices=["rc2", "evalmaxsat"], help="Solver engine")
    parser.add_argument("--solver", type=str, default="g3", help="Underlying SAT solver name for RC2 (e.g., g3, g4, cadical)")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per solve call (seconds) for external solver")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--sym", nargs='+', default=[], help="Symmetry breaking methods (e.g., lex_row lex_col r)")
    parser.add_argument(
        "--card-enc",
        type=str,
        default="seqcounter",
        choices=["seqcounter", "totalizer", "cardnetwrk", "sortnetwrk", "kmtotalizer"],
        help="Cardinality encoding used for bounds",
    )

    args = parser.parse_args()
    input_path = args.input
    if not os.path.exists(input_path):
        potential_path = os.path.join("input", input_path)
        if os.path.exists(potential_path):
            input_path = potential_path

    if os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        files = sorted(f for f in os.listdir(input_path) if f.endswith(".txt"))
        if not files:
            print("No .txt files found in directory.")
            return

        print(f"Found {len(files)} input files.")

        try:
            import openpyxl
            has_openpyxl = True
        except Exception:
            print("Warning: openpyxl not found. Excel export disabled.")
            has_openpyxl = False

        folder_name = os.path.basename(os.path.normpath(input_path))
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        output_file = f"result_{folder_name}_{script_name}.xlsx"
        columns = ['File', 'v', 'b', 'r', 'Engine', 'Encoding', 'Optimal Lambda', 'Variables', 'Clauses', 'Sym Method', 'Time (s)', 'Status']

        wb = None
        ws = None
        if has_openpyxl:
            if not os.path.exists(output_file):
                wb = openpyxl.Workbook()
                ws = wb.active
                ws.append(columns)
                wb.save(output_file)

        for fname in files:
            fp = os.path.join(input_path, fname)
            res = solve_from_file(
                fp,
                engine=args.engine,
                timeout=args.timeout,
                solver_name=args.solver,
                quiet=args.quiet,
                sym_options=args.sym,
                card_encoding=args.card_enc,
            )

            if res and has_openpyxl:
                wb = openpyxl.load_workbook(output_file)
                ws = wb.active
                ws.append([res.get(c) for c in columns])
                wb.save(output_file)
                print("\n" + "-" * 60)
                print(f"Result for {fname} appended to {output_file}")
                print("-" * 60)

    elif os.path.isfile(input_path):
        solve_from_file(
            input_path,
            engine=args.engine,
            timeout=args.timeout,
            solver_name=args.solver,
            quiet=args.quiet,
            sym_options=args.sym,
            card_encoding=args.card_enc,
        )
    else:
        print(f"Error: Input path '{args.input}' not found.")

if __name__ == "__main__":
    main()