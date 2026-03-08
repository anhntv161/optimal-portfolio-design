#!/usr/bin/env python3
"""
Portfolio Optimisation (PO) via MaxSAT (RC2) — minimax λ

This script encodes the Portfolio Optimisation (PO) problem from:
  Flener, Pearson, Reyna. "Financial Portfolio Optimisation" (CP 2004)

PO problem:
- v sub-pools (rows), universe size b (credits), each sub-pool has size r
- Objective: minimize the maximum pairwise intersection size λ

MaxSAT encoding (correct minimax):
- Hard:
  * For each i: sum_j x[i,j] == r
  * For each pair (i1,i2): build overlap count I(i1,i2) = sum_j (x[i1,j] AND x[i2,j])
  * Link all overlaps to a global unary λ:
      If I(i1,i2) >= k  =>  λ >= k
- Soft:
  * For k=1..r: prefer λ < k, i.e., soft clause (¬L_k) where L_k means (λ >= k)
    With monotonicity L_k -> L_{k-1}, the number of violated soft clauses equals λ.

Notes:
- Default solver engine is RC2 (PySAT). You can optionally use an external solver
  (EvalMaxSAT_bin) by setting --engine evalmaxsat.
- The previous version had a critical parsing bug for 'v ...' output lines from
  external solvers; this version includes a robust parser that supports:
  * DIMACS literal assignments: v 1 -2 3 ... 0
  * bitstrings: v 010101...
  * multi-line v blocks

Author: (patched by ChatGPT)
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
from pysat.card import CardEnc, EncType, ITotalizer


@dataclass
class SolveResult:
    model: Optional[List[int]]
    cost: Optional[int]
    solve_time_s: float
    status: str


def parse_evalmaxsat_output(stdout: str, nvars: int) -> Tuple[Optional[List[int]], Optional[int]]:
    """
    Robustly parse MaxSAT solver output.

    Supports:
      - 'o <int>' objective lines
      - one or multiple 'v ...' lines containing:
         * DIMACS signed literals ending with 0
         * a single 0/1 bitstring of length nvars
         * nvars 0/1 tokens

    Returns: (model, cost)
      model is a list of signed ints (DIMACS style): positive => True, negative => False.
    """
    cost = None
    assign: Dict[int, int] = {}  # abs(var) -> signed literal

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

        # Case A: nvars tokens all 0/1
        if len(toks) == nvars and all(t in ('0', '1') for t in toks):
            for i, bit in enumerate(toks, start=1):
                assign[i] = i if bit == '1' else -i
            continue

        # Case B: single bitstring
        if len(toks) == 1 and re.fullmatch(r'[01]+', toks[0]) and len(toks[0]) == nvars:
            for i, bit in enumerate(toks[0], start=1):
                assign[i] = i if bit == '1' else -i
            continue

        # Default: DIMACS literals
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


class PortfolioMaxSAT:
    """
    Portfolio Optimisation via MaxSAT (minimize max overlap λ).

    Variables:
      x(i,j)  for i in 1..v, j in 1..b  : credit j is in sub-pool i

    Additional:
      L_k for k in 1..r : λ >= k   (unary, monotone)
    """

    def __init__(
        self,
        v: int,
        b: int,
        r: int,
        *,
        verbose: bool = True,
        symbreak: bool = False,
        card_encoding: str = "seqcounter",
    ):
        self.v = int(v)
        self.b = int(b)
        self.r = int(r)
        self.verbose = verbose
        self.symbreak = symbreak

        if self.v <= 0 or self.b <= 0 or self.r < 0:
            raise ValueError("Invalid parameters: require v>0, b>0, r>=0")
        if self.r > self.b:
            raise ValueError(f"Infeasible: r={self.r} cannot exceed b={self.b}")

        # Base variable mapping: x(i,j) = (i-1)*b + j
        self.var_id = lambda i, j: (i - 1) * self.b + j
        self.total_vars = self.v * self.b

        # Next free variable id for auxiliaries (ANDs, totalizers, lambda vars)
        self.next_var_id = self.total_vars + 1

        # Choose cardinality encoding for equals constraints
        enc_map = {
            "seqcounter": EncType.seqcounter,
            "totalizer": EncType.totalizer,
            "cardnetwrk": EncType.cardnetwrk,
            "sortnetwrk": EncType.sortnetwrk,
        }
        if card_encoding not in enc_map:
            raise ValueError(f"Unknown card_encoding '{card_encoding}'. Choose from: {sorted(enc_map)}")
        self.eq_encoding = enc_map[card_encoding]

        self.wcnf = WCNF()

        # Build formula
        self._add_hard_pool_size_constraints()
        if self.symbreak:
            self._add_symmetry_breaking()

        # If v==1, λ is trivially 0
        self.lambda_ge: List[int] = [0]  # 1-indexed, lambda_ge[k] is var for L_k
        if self.v >= 2 and self.r >= 1:
            self._add_lambda_vars_and_objective()
            self._add_pairwise_overlap_linking()
        else:
            if self.verbose:
                print("\nTrivial instance: v<2 or r<1 => λ = 0 (no pairwise overlaps).")

    def _new_var(self) -> int:
        vid = self.next_var_id
        self.next_var_id += 1
        return vid

    # ---------------------- HARD constraints ----------------------

    def _add_hard_pool_size_constraints(self) -> None:
        """For each sub-pool i: sum_j x(i,j) == r"""
        if self.verbose:
            print("\n=== Adding Hard Clauses: Pool Sizes ===")
            print(f"Each sub-pool must contain exactly r={self.r} credits")

        for i in range(1, self.v + 1):
            pool_vars = [self.var_id(i, j) for j in range(1, self.b + 1)]
            cnf = CardEnc.equals(
                lits=pool_vars,
                bound=self.r,
                encoding=self.eq_encoding,
                top_id=self.next_var_id - 1,
            )

            # Add clauses
            for cl in cnf.clauses:
                self.wcnf.append(cl)

            # Update next_var_id (CNFPlus.nv is the maximum var id used)
            self.next_var_id = max(self.next_var_id, cnf.nv + 1)

            if self.verbose:
                print(f"  Sub-pool {i}: added equals({self.r}) with {len(cnf.clauses)} clauses")

        if self.verbose:
            print(f"Hard clauses so far: {len(self.wcnf.hard)}")
            print(f"Variables so far (incl. aux): {self.next_var_id - 1}")

    def _add_symmetry_breaking(self) -> None:
        """
        Optional symmetry breaking.

        Safe in the CP2004 PO formulation because credits are unlabeled and the
        objective depends only on intersection structure.

        We fix the first pool to contain credits 1..r.
        """
        if self.verbose:
            print("\n=== Adding Symmetry Breaking ===")
            print("Fixing sub-pool 1 to contain credits 1..r (and exclude r+1..b).")

        # Force x(1,1..r)=True
        for j in range(1, self.r + 1):
            self.wcnf.append([self.var_id(1, j)])  # unit clause

        # Force x(1,r+1..b)=False
        for j in range(self.r + 1, self.b + 1):
            self.wcnf.append([-self.var_id(1, j)])

        if self.verbose:
            print("Symmetry breaking constraints added.")

    def _add_lambda_vars_and_objective(self) -> None:
        """
        Create unary λ variables L_k (λ>=k) with monotonicity, add Corrádi lower bound,
        and add soft clauses (¬L_k) to minimize λ.
        """
        if self.verbose:
            print("\n=== Adding Global λ (unary) and Objective ===")

        # Create L_k for k=1..r
        self.lambda_ge = [0] + [self._new_var() for _ in range(1, self.r + 1)]

        # Monotonicity: L_k -> L_{k-1}  (for k>=2)
        for k in range(2, self.r + 1):
            self.wcnf.append([-self.lambda_ge[k], self.lambda_ge[k - 1]])

        # Enforce theoretical lower bound (Corrádi) as hard constraint if > 0
        lb = self.calculate_lower_bound()
        if lb > 0:
            self.wcnf.append([self.lambda_ge[lb]])
            if self.verbose:
                print(f"Added hard lower bound: λ >= {lb}")

        # Soft objective: minimize number of true L_k => minimize λ
        for k in range(1, self.r + 1):
            self.wcnf.append([-self.lambda_ge[k]], weight=1)

        if self.verbose:
            print(f"Added {self.r} soft clauses to minimize λ.")
            print(f"Variables so far (incl. aux): {self.next_var_id - 1}")

    def _add_pairwise_overlap_linking(self) -> None:
        """
        For each pair (i1,i2), create AND-vars y_j = x(i1,j) ∧ x(i2,j),
        build a totalizer over y_j up to ubound=r, and link totalizer rhs to L_k:

          rhs[k-1] == (sum y_j >= k)  =>  L_k

        i.e., clause: (¬rhs[k-1] ∨ L_k) for all k.
        """
        if self.verbose:
            print("\n=== Adding Pairwise Overlap Constraints ===")
            print("Building overlap totalizers and linking to global λ")

        ub = min(self.r, self.b)
        pair_count = 0
        and_var_count = 0
        totalizer_clause_count = 0

        for i1, i2 in itertools.combinations(range(1, self.v + 1), 2):
            pair_count += 1

            # AND variables y_j for overlap in this pair
            y_lits: List[int] = []
            for j in range(1, self.b + 1):
                x1 = self.var_id(i1, j)
                x2 = self.var_id(i2, j)
                y = self._new_var()
                and_var_count += 1

                # y <-> (x1 ∧ x2)
                self.wcnf.append([-y, x1])
                self.wcnf.append([-y, x2])
                self.wcnf.append([-x1, -x2, y])

                y_lits.append(y)

            # Totalizer for sum(y_lits) with max enforceable bound ub
            t = ITotalizer(lits=y_lits, ubound=ub, top_id=self.next_var_id - 1)

            for cl in t.cnf.clauses:
                self.wcnf.append(cl)
            totalizer_clause_count += len(t.cnf.clauses)

            # Update next_var_id to avoid collisions
            # (t.cnf.nv is max var used in the totalizer CNF)
            self.next_var_id = max(self.next_var_id, t.cnf.nv + 1)

            # Link thresholds to global λ
            # By PySAT docs: rhs[k] corresponds to "sum >= k+1".
            rhs = t.rhs
            for k in range(1, ub + 1):
                if k - 1 < len(rhs):
                    self.wcnf.append([-rhs[k - 1], self.lambda_ge[k]])
                else:
                    # Should not happen, but keep safe
                    raise RuntimeError("Unexpected totalizer rhs length; cannot link to λ.")

            # If ub < r, then overlaps can't reach k>ub, no need to link.

            # Release totalizer resources
            t.delete()

            if self.verbose and pair_count % 10 == 0:
                print(f"  Processed {pair_count} pairs...")

        if self.verbose:
            print(f"Pairs processed: {pair_count}")
            print(f"AND vars created: {and_var_count}")
            print(f"Totalizer clauses added: {totalizer_clause_count}")
            print(f"Total hard clauses: {len(self.wcnf.hard)}")
            print(f"Total soft clauses: {len(self.wcnf.soft)}")
            print(f"Total variables (incl. aux): {self.next_var_id - 1}")

    # ---------------------- Solving ----------------------

    def solve(
        self,
        *,
        engine: str = "rc2",
        timeout: int = 3600,
        solver_name: str = "g3",
        verbose: bool = True,
    ) -> SolveResult:
        """
        Solve the MaxSAT instance.

        engine:
          - "rc2": use PySAT RC2 directly (recommended)
          - "evalmaxsat": use external ./EvalMaxSAT_bin on a WCNF file
        """
        engine = engine.lower().strip()
        start_time = time.time()

        if verbose:
            print("\n=== Solving MaxSAT ===")
            print(f"Engine: {engine}")
            print(f"Problem: v={self.v}, b={self.b}, r={self.r}")
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
                if verbose:
                    print(f"RC2 finished in {t:.3f}s; cost={cost}")
                return SolveResult(model=model, cost=cost, solve_time_s=t, status=status)
            except Exception as e:
                t = time.time() - start_time
                if verbose:
                    print(f"RC2 error: {e}")
                return SolveResult(model=None, cost=None, solve_time_s=t, status=f"Error: {e}")

        if engine == "evalmaxsat":
            # Write WCNF file
            wcnf_filename = self.write_wcnf_file()
            solver_path = "/home/anhntv161/KLTN/2. Code/Base/EvalMaxSAT/build/main/EvalMaxSAT_bin"
            if not os.path.exists(solver_path):
                t = time.time() - start_time
                msg = f"Error: external solver {solver_path} not found."
                if verbose:
                    print(msg)
                return SolveResult(model=None, cost=None, solve_time_s=t, status=msg)

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
                if verbose:
                    print(f"External solver finished in {t:.3f}s; cost={cost}")
                return SolveResult(model=model, cost=cost, solve_time_s=t, status=status)
            except subprocess.TimeoutExpired:
                t = time.time() - start_time
                msg = f"Timeout after {timeout}s"
                if verbose:
                    print(msg)
                return SolveResult(model=None, cost=None, solve_time_s=t, status=msg)
            except Exception as e:
                t = time.time() - start_time
                msg = f"Error: {e}"
                if verbose:
                    print(msg)
                return SolveResult(model=None, cost=None, solve_time_s=t, status=msg)

        t = time.time() - start_time
        msg = f"Unknown engine '{engine}'. Use 'rc2' or 'evalmaxsat'."
        if verbose:
            print(msg)
        return SolveResult(model=None, cost=None, solve_time_s=t, status=msg)

    # ---------------------- IO / Utility ----------------------

    def write_wcnf_file(self, filename: str = "problem_eval.wcnf") -> str:
        """Write the formula to a valid WCNF file."""
        with open(filename, "w", encoding="utf-8") as f:
            total_clauses = len(self.wcnf.hard) + len(self.wcnf.soft)

            weights = getattr(self.wcnf, "wght", [])
            top_weight = (sum(weights) + 1) if weights else (len(self.wcnf.soft) + 1)

            f.write(f"p wcnf {self.next_var_id - 1} {total_clauses} {top_weight}\n")

            # hard
            for clause in self.wcnf.hard:
                f.write(str(top_weight) + " " + " ".join(map(str, clause)) + " 0\n")

            # soft
            for idx, clause in enumerate(self.wcnf.soft):
                w = weights[idx] if idx < len(weights) else 1
                f.write(f"{w} " + " ".join(map(str, clause)) + " 0\n")

        return filename

    def decode_solution(self, model: Optional[Sequence[int]]) -> Optional[List[Set[int]]]:
        """Decode a DIMACS-like model into list of sub-pools (1..b credits)."""
        if not model:
            return None

        model_set = set(model)
        sub_pools: List[Set[int]] = [set() for _ in range(self.v)]

        for i in range(1, self.v + 1):
            pool = sub_pools[i - 1]
            for j in range(1, self.b + 1):
                var = self.var_id(i, j)
                if var in model_set:  # True assignment
                    pool.add(j)

        return sub_pools

    def calculate_lambda(self, sub_pools: Optional[List[Set[int]]]) -> Optional[int]:
        """Compute maximum pairwise intersection size."""
        if sub_pools is None:
            return None
        if self.v < 2:
            return 0

        max_intersection = 0
        for i1, i2 in itertools.combinations(range(self.v), 2):
            max_intersection = max(max_intersection, len(sub_pools[i1] & sub_pools[i2]))
        return max_intersection

    def calculate_lower_bound(self) -> int:
        """
        Corrádi lower bound used in the CP2004 paper:
          λ ≥ ceil( r (r v - b) / ( b (v-1) ) ), and λ ≥ 0
        """
        if self.v <= 1:
            return 0
        numerator = self.r * (self.r * self.v - self.b)
        denominator = self.b * (self.v - 1)
        if denominator == 0:
            return 0
        return max(0, math.ceil(numerator / denominator))

    def print_solution(self, sub_pools: Optional[List[Set[int]]], lambda_val: Optional[int], *, cost: Optional[int] = None) -> None:
        """Pretty-print solution and a small verification."""
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
            print(f"MaxSAT reported cost:         {cost}")

        if lambda_val is not None and lambda_val == lb:
            print("✓ Optimal (matches lower bound)")
        elif lambda_val is not None:
            gap = lambda_val - lb
            pct = (gap / max(1, lb)) * 100.0
            print(f"Gap from lower bound:         {gap} ({pct:.1f}%)")

        # Pools
        print("\nSub-pools:")
        for i, pool in enumerate(sub_pools, 1):
            print(f"  Sub-pool {i:2d} (size {len(pool)}): {sorted(pool)}")

        # Verify sizes
        ok = True
        for i, pool in enumerate(sub_pools, 1):
            if len(pool) != self.r:
                ok = False
                print(f"  ✗ Size violation: pool {i} has {len(pool)} instead of {self.r}")
        if ok:
            print(f"\n✓ All sub-pools have correct size r={self.r}")

        # Small intersection matrix preview
        print("\nIntersection matrix (preview):")
        nshow = min(self.v, 10)
        print("     " + "".join(f"{i:4d}" for i in range(1, nshow + 1)) + (" ..." if self.v > nshow else ""))
        for i1 in range(nshow):
            row = [("  -" if i1 == i2 else f"{len(sub_pools[i1] & sub_pools[i2]):4d}") for i2 in range(nshow)]
            print(f"{i1+1:3d} " + "".join(row) + (" ..." if self.v > nshow else ""))

        if self.v > nshow:
            print("  ... (matrix truncated)")
        print("=" * 60)


# ---------------------- Input handling ----------------------

def parse_input_file(filepath: str) -> Optional[Dict[str, int]]:
    """
    Parse v, b, r from a text file.
    Accepts patterns like:
      v = 10
      b=350
      r: 100
    """
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
    symbreak: bool,
    card_encoding: str,
) -> Dict[str, object]:
    """Read params from file and solve. Returns a dict suitable for printing/export."""
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
            "Optimal Lambda": None,
            "Time (s)": 0.0,
            "Status": "Parse Error",
        }

    v, b, r = params["v"], params["b"], params["r"]
    if not quiet:
        print(f"Parameters: v={v}, b={b}, r={r}")

    try:
        start = time.time()
        problem = PortfolioMaxSAT(v, b, r, verbose=not quiet, symbreak=symbreak, card_encoding=card_encoding)
        sol = problem.solve(engine=engine, timeout=timeout, solver_name=solver_name, verbose=not quiet)
        elapsed = time.time() - start

        sub_pools = problem.decode_solution(sol.model)
        lambda_val = problem.calculate_lambda(sub_pools)

        # Basic validation
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
            "Optimal Lambda": lambda_val,
            "Time (s)": round(elapsed, 3),
            "Status": status,
        }

        if sub_pools is not None and not quiet:
            print(f"\nRESULT for {filename}:")
            print(f"Optimal lambda: {lambda_val}")
            print(f"MaxSAT cost:    {sol.cost}")
            print(f"Total time:     {elapsed:.3f}s")
            problem.print_solution(sub_pools, lambda_val, cost=sol.cost)

            if sol.cost is not None and lambda_val is not None and sol.cost != lambda_val:
                print("WARNING: cost != computed λ. (If using external solver, this may mean output format mismatch.)")

        elif sub_pools is None and not quiet:
            print(f"\nNo solution found for {filename}")

        return result_data

    except Exception as e:
        import traceback

        print(f"Solver error: {e}")
        traceback.print_exc()
        return {
            "File": filename,
            "v": v,
            "b": b,
            "r": r,
            "Engine": engine,
            "Optimal Lambda": None,
            "Time (s)": 0.0,
            "Status": f"Error: {str(e)}",
        }


# ---------------------- CLI ----------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MaxSAT-based PO solver (minimax λ) using RC2 or external EvalMaxSAT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  # Solve a single instance (default RC2)
  python Opd_EvalMaxSAT_ver2.py --input input/small/small_2.txt

  # Solve using external EvalMaxSAT_bin
  python Opd_EvalMaxSAT_ver2.py --engine evalmaxsat --input input/small/small_2.txt

  # Process all .txt instances in a directory and export to Excel
  python Opd_EvalMaxSAT_ver2.py --input input/small
""",
    )

    parser.add_argument("--input", type=str, required=True, help="Input file or directory path")
    parser.add_argument("--engine", type=str, default="rc2", choices=["rc2", "evalmaxsat"], help="Solver engine")
    parser.add_argument("--solver", type=str, default="g3", help="Underlying SAT solver name for RC2 (e.g., g3, g4, cadical)")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per solve call (seconds) for external solver")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--symbreak", action="store_true", help="Enable symmetry breaking (fix first pool to 1..r)")
    parser.add_argument(
        "--card-enc",
        type=str,
        default="seqcounter",
        choices=["seqcounter", "totalizer", "cardnetwrk", "sortnetwrk"],
        help="Cardinality encoding used for equals constraints",
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
        columns = ["File", "v", "b", "r", "Engine", "Optimal Lambda", "Time (s)", "Status"]

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
                symbreak=args.symbreak,
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
            symbreak=args.symbreak,
            card_encoding=args.card_enc,
        )
    else:
        print(f"Error: Input path '{args.input}' not found.")


if __name__ == "__main__":
    main()
