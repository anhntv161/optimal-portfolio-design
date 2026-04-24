#!/usr/bin/env python3
"""
Hybrid OPD solver: Gurobi MIP provides incumbents while Incremental SAT
(Reified Bound + Binary Search) verifies/improves below the current best
lambda in parallel.

Strategy:
  1. MIP (Gurobi) runs in a background process, emitting each new incumbent
     lambda via a queue.
  2. Whenever MIP finds a new best lambda, we (re)start an Incremental SAT
     worker that searches in range [lb, best_lambda - 1] using binary search.
  3. If SAT finds a solution  → update best lambda, start new SAT job.
     If SAT returns UNSAT for the full range → current best is OPTIMAL.
  4. If MIP proves optimality on its own → done.
  5. On timeout → return best found so far.

The Incremental SAT side uses OpdReifiedBound:
  - Builds the full Totalizer structure ONCE per worker invocation.
  - Uses assumption-based binary search over [lb, target_ub - 1].
  - Returns the optimal lambda found within that range (or None if UNSAT).
"""

import argparse
import itertools
import math
import multiprocessing as mp
import os
import queue as queue_mod
import re
import sys
import threading
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType


ENCODING_MAP = {
    'sortnetwrk': EncType.sortnetwrk,
    'seqcounter':  EncType.seqcounter,
    'totalizer':   EncType.totalizer,
    'mtotalizer':  EncType.mtotalizer,
    'pairwise':    EncType.pairwise,
    'cardnetwrk':  EncType.cardnetwrk,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def parse_input_file(filepath):
    """Parse v, b, r from input file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        v_m = re.search(r'v\s*=\s*(\d+)', content)
        b_m = re.search(r'b\s*=\s*(\d+)', content)
        r_m = re.search(r'r\s*=\s*(\d+)', content)
        if v_m and b_m and r_m:
            return {'v': int(v_m.group(1)), 'b': int(b_m.group(1)), 'r': int(r_m.group(1))}
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
    return None


def compute_lower_bound(v, b, r):
    """Corrádi lower bound on lambda."""
    if v <= 1:
        return 0
    num = r * (r * v - b)
    den = b * (v - 1)
    if den == 0:
        return 0
    return max(0, math.ceil(num / den))


# ---------------------------------------------------------------------------
# MIP side (Gurobi)  — identical to Opd_hybrid_mip_sat.py
# ---------------------------------------------------------------------------

class PortfolioMIP:
    """MIP-based solver for OPD (Gurobi)."""

    def __init__(self, v, b, r, sym_options=None, verbose=True):
        self.v = v
        self.b = b
        self.r = r
        self.sym_options = sym_options or []
        self.verbose = verbose
        self.lambda_lb = compute_lower_bound(v, b, r)

        if self.verbose:
            print(f"OPD Instance: v={v}, b={b}, r={r}")
            print("Solver: Gurobi MIP")
            print(f"Lower bound on lambda: {self.lambda_lb}")

    def build_model(self, linearize=True):
        self.model = gp.Model("opd_hybrid_mip_incsat")
        if not self.verbose:
            self.model.setParam('OutputFlag', 0)

        self.x = self.model.addVars(self.v, self.b, vtype=GRB.BINARY, name="x")
        self.lambda_var = self.model.addVar(
            vtype=GRB.INTEGER, lb=self.lambda_lb, ub=self.r, name="lambda"
        )

        for i in range(self.v):
            self.model.addConstr(
                gp.quicksum(self.x[i, j] for j in range(self.b)) == self.r,
                f"row_sum_{i}"
            )

        if linearize:
            self.y = self.model.addVars(
                [(i1, i2, j) for i1 in range(self.v)
                              for i2 in range(i1 + 1, self.v)
                              for j in range(self.b)],
                vtype=GRB.BINARY, name="y"
            )
            for i1 in range(self.v):
                for i2 in range(i1 + 1, self.v):
                    for j in range(self.b):
                        self.model.addConstr(self.y[i1, i2, j] <= self.x[i1, j])
                        self.model.addConstr(self.y[i1, i2, j] <= self.x[i2, j])
                        self.model.addConstr(self.y[i1, i2, j] >= self.x[i1, j] + self.x[i2, j] - 1)
                    self.model.addConstr(
                        gp.quicksum(self.y[i1, i2, j] for j in range(self.b)) <= self.lambda_var,
                        f"overlap_{i1}_{i2}"
                    )
        else:
            for i1 in range(self.v):
                for i2 in range(i1 + 1, self.v):
                    self.model.addConstr(
                        gp.quicksum(self.x[i1, j] * self.x[i2, j] for j in range(self.b)) <= self.lambda_var,
                        f"overlap_{i1}_{i2}"
                    )

        self._add_symmetry_breaking()
        self.model.setObjective(self.lambda_var, GRB.MINIMIZE)
        self.model.update()
        self.num_vars = self.model.NumVars
        self.num_constrs = self.model.NumConstrs

    def _add_lex_le(self, a, b_vec, n, prefix):
        p = {0: 1}
        for k in range(n):
            p_k_is_const = isinstance(p[k], int)
            if k < n - 1:
                eq_k = self.model.addVar(vtype=GRB.BINARY, name=f"{prefix}_eq_{k}")
                self.model.addConstr(a[k] - b_vec[k] <= 1 - eq_k, f"{prefix}_eq1_{k}")
                self.model.addConstr(b_vec[k] - a[k] <= 1 - eq_k, f"{prefix}_eq2_{k}")
                self.model.addConstr(eq_k >= a[k] + b_vec[k] - 1,  f"{prefix}_eq3_{k}")
                self.model.addConstr(eq_k >= 1 - a[k] - b_vec[k],  f"{prefix}_eq4_{k}")
                if p_k_is_const:
                    p[k + 1] = eq_k
                else:
                    pk1 = self.model.addVar(vtype=GRB.BINARY, name=f"{prefix}_p_{k+1}")
                    self.model.addConstr(pk1 <= p[k],             f"{prefix}_p1_{k+1}")
                    self.model.addConstr(pk1 <= eq_k,             f"{prefix}_p2_{k+1}")
                    self.model.addConstr(pk1 >= p[k] + eq_k - 1, f"{prefix}_p3_{k+1}")
                    p[k + 1] = pk1
            if p_k_is_const:
                self.model.addConstr(a[k] <= b_vec[k], f"{prefix}_lex_{k}")
            else:
                self.model.addConstr(a[k] - b_vec[k] <= 1 - p[k], f"{prefix}_lex_{k}")

    def _add_symmetry_breaking(self):
        use_lex_row = 'lex_row' in self.sym_options
        use_lex_col = 'lex_col' in self.sym_options
        fixed_r = 'r' in self.sym_options
        sym_methods = []

        if fixed_r:
            sym_methods.append('r')
            for j in range(self.r):
                self.model.addConstr(self.x[0, j] == 1, f"fix_row0_one_{j}")
            for j in range(self.r, self.b):
                self.model.addConstr(self.x[0, j] == 0, f"fix_row0_zero_{j}")

        if use_lex_row:
            sym_methods.append('lex_row')
            for i in range(self.v - 1):
                a_row = [self.x[i, j] for j in range(self.b)]
                b_row = [self.x[i + 1, j] for j in range(self.b)]
                if fixed_r:
                    self._add_lex_le(b_row, a_row, self.b, f"row_lex_{i}")
                else:
                    self._add_lex_le(a_row, b_row, self.b, f"row_lex_{i}")

        if use_lex_col:
            sym_methods.append('lex_col')
            for j in range(self.b - 1):
                a_col = [self.x[i, j] for i in range(self.v)]
                b_col = [self.x[i, j + 1] for i in range(self.v)]
                if fixed_r:
                    self._add_lex_le(b_col, a_col, self.v, f"col_lex_{j}")
                else:
                    self._add_lex_le(a_col, b_col, self.v, f"col_lex_{j}")

        self.applied_sym = '+'.join(sym_methods) if sym_methods else 'none'


def _extract_mip_matrix(model):
    v, b = model._v, model._b
    matrix = np.zeros((v, b), dtype=int)
    for i in range(v):
        for j in range(b):
            matrix[i, j] = round(model.cbGetSolution(model._x[i, j]))
    return matrix


def _mip_callback(model, where):
    if model._stop_event.is_set():
        model.terminate()
        return
    if where == GRB.Callback.MIPSOL:
        incumbent = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJ)))
        if incumbent < model._best_sent:
            matrix = _extract_mip_matrix(model)
            model._best_sent = incumbent
            model._queue.put({
                'type': 'MIP_INCUMBENT',
                'lambda': incumbent,
                'matrix': matrix.tolist(),
                'time': time.time() - model._start_time,
            })
            if incumbent <= model._lambda_lb:
                model._queue.put({
                    'type': 'MIP_PROVED_BY_LB',
                    'lambda': incumbent,
                    'matrix': matrix.tolist(),
                    'time': time.time() - model._start_time,
                })
                model.terminate()


def mip_worker(v, b, r, sym_options, linearize, timeout, quiet, out_queue, stop_event):
    try:
        start_time = time.time()
        solver = PortfolioMIP(v, b, r, sym_options=sym_options, verbose=not quiet)
        solver.build_model(linearize=linearize)

        mdl = solver.model
        mdl._queue = out_queue
        mdl._stop_event = stop_event
        mdl._start_time = start_time
        mdl._x = solver.x
        mdl._v = v
        mdl._b = b
        mdl._lambda_lb = solver.lambda_lb
        mdl._best_sent = float('inf')

        mdl.setParam('TimeLimit', timeout)
        mdl.setParam('MIPGap', 0.0)
        mdl.setParam('Cuts', 2)
        mdl.setParam('Heuristics', 0.1)

        mdl.optimize(_mip_callback)
        total_time = time.time() - start_time

        result = {
            'type': 'MIP_DONE',
            'status': 'UNKNOWN',
            'lambda': None,
            'matrix': None,
            'time': total_time,
            'gap': None,
            'nodes': int(mdl.NodeCount) if hasattr(mdl, 'NodeCount') else 0,
            'n_vars': getattr(solver, 'num_vars', 0),
            'n_constrs': getattr(solver, 'num_constrs', 0),
            'applied_sym': getattr(solver, 'applied_sym', 'none'),
        }

        if mdl.SolCount > 0:
            matrix = np.zeros((v, b), dtype=int)
            for i in range(v):
                for j in range(b):
                    matrix[i, j] = round(solver.x[i, j].X)
            lam = round(solver.lambda_var.X)
            result['lambda'] = int(lam)
            result['matrix'] = matrix.tolist()
            result['gap'] = float(getattr(mdl, 'MIPGap', 0.0))

        if mdl.status == GRB.OPTIMAL:
            result['status'] = 'OPTIMAL'
        elif mdl.status == GRB.TIME_LIMIT:
            result['status'] = 'TIMEOUT_WITH_SOLUTION' if mdl.SolCount > 0 else 'TIMEOUT'
        elif mdl.status == GRB.INTERRUPTED:
            result['status'] = 'INTERRUPTED'
        elif mdl.status == GRB.INFEASIBLE:
            result['status'] = 'INFEASIBLE'
        else:
            result['status'] = f'UNKNOWN_{mdl.status}'

        out_queue.put(result)
    except Exception as e:
        out_queue.put({'type': 'MIP_ERROR', 'error': str(e)})


# ---------------------------------------------------------------------------
# Incremental SAT side (OpdReifiedBound)
# ---------------------------------------------------------------------------

def _totalizer_merge(left_out, right_out, var_counter, clauses):
    """Merge two totalizer output lists (upward clauses only)."""
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
    """Recursively build upward Totalizer. Returns output vars."""
    n = len(input_lits)
    if n == 0:
        return []
    if n == 1:
        return [input_lits[0]]
    mid = n // 2
    left_out  = _build_totalizer_manual(input_lits[:mid], var_counter, clauses)
    right_out = _build_totalizer_manual(input_lits[mid:], var_counter, clauses)
    return _totalizer_merge(left_out, right_out, var_counter, clauses)


class OpdReifiedBound:
    """
    Reified Bound + Binary Search Incremental SAT solver for OPD.

    Build once: full Totalizer for every pair, b_M assumption literals.
    Search: binary search in [lo, hi] using assumption b_M.
    """

    def __init__(self, v, b, r, solver_name='cadical195', encoding_name='sortnetwrk',
                 sym_options=None, verbose=True):
        self.v = v
        self.b = b
        self.r = r
        self.solver_name = solver_name
        self.encoding_name = encoding_name
        self.encoding_type = ENCODING_MAP.get(encoding_name, EncType.sortnetwrk)
        self.sym_options = sym_options or []
        self.verbose = verbose
        self._next_var = 1

        self.x     = [[self._alloc() for _ in range(b)] for _ in range(v)]
        self.pairs = list(itertools.combinations(range(v), 2))
        self.s     = {(i, j): [self._alloc() for _ in range(b)] for (i, j) in self.pairs}
        self.b_lit = {M: self._alloc() for M in range(r + 1)}

    def _alloc(self):
        v = self._next_var
        self._next_var += 1
        return v

    def _add_lex_constraint(self, solver, a, b_vec, n):
        """SAT encoding of a <=_lex b_vec."""
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
        fixed_r   = 'r'       in self.sym_options
        use_lex_row = 'lex_row' in self.sym_options
        use_lex_col = 'lex_col' in self.sym_options
        sym_methods = []

        if fixed_r:
            for j in range(self.r):
                solver.add_clause([self.x[0][j]])
            for j in range(self.r, self.b):
                solver.add_clause([-self.x[0][j]])
            sym_methods.append('r')

        if use_lex_row:
            sym_methods.append('lex_row')
            for i in range(self.v - 1):
                if fixed_r:
                    self._add_lex_constraint(solver, self.x[i + 1], self.x[i], self.b)
                else:
                    self._add_lex_constraint(solver, self.x[i], self.x[i + 1], self.b)

        if use_lex_col:
            sym_methods.append('lex_col')
            for j in range(self.b - 1):
                col_j  = [self.x[i][j]     for i in range(self.v)]
                col_j1 = [self.x[i][j + 1] for i in range(self.v)]
                if fixed_r:
                    self._add_lex_constraint(solver, col_j1, col_j, self.v)
                else:
                    self._add_lex_constraint(solver, col_j, col_j1, self.v)

        self.applied_sym = '+'.join(sym_methods) if sym_methods else 'none'

    def build_formula(self):
        """Build solver with full formula. Returns (solver, o_vars)."""
        solver = Solver(name=self.solver_name)

        # Row cardinality
        for i in range(self.v):
            cnf = CardEnc.equals(
                lits=self.x[i], bound=self.r,
                top_id=self._next_var - 1, encoding=self.encoding_type,
            )
            self._next_var = cnf.nv + 1
            for cl in cnf.clauses:
                solver.add_clause(cl)

        # AND gates: s[(i,j)][k] <-> x[i][k] AND x[j][k]
        for (i, j) in self.pairs:
            sv = self.s[(i, j)]
            for k in range(self.b):
                s_var, xi, xj = sv[k], self.x[i][k], self.x[j][k]
                solver.add_clause([-s_var, xi])
                solver.add_clause([-s_var, xj])
                solver.add_clause([-xi, -xj, s_var])

        # Symmetry breaking
        self._add_symmetry_breaking(solver)

        # Totalizers
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

        # Reified bound clauses: b_M => (D_{ij} <= M)  i.e.  b_M => NOT o[M]
        # CNF: (-b_M OR -o[M])    (o[M]=1 means D >= M+1)
        for M in range(self.r + 1):
            bM = self.b_lit[M]
            for (i, j) in self.pairs:
                out = o_vars[(i, j)]
                if M < len(out):
                    solver.add_clause([-bM, -out[M]])
            # Monotonicity: b_M => b_{M+1}
            if M < self.r:
                solver.add_clause([-bM, self.b_lit[M + 1]])

        return solver, o_vars

    def _extract_matrix(self, model):
        mset = set(model)
        return [[1 if self.x[i][k] in mset else 0
                 for k in range(self.b)] for i in range(self.v)]

    def _compute_max_overlap(self, matrix):
        max_ov = 0
        for (i, j) in self.pairs:
            ov = sum(matrix[i][k] * matrix[j][k] for k in range(self.b))
            if ov > max_ov:
                max_ov = ov
        return max_ov

    def solve_range(self, lo, hi, timeout):
        """
        Binary search in [lo, hi] with assumption-based SAT.
        Returns (best_M, best_matrix, elapsed, n_vars, n_clauses)
        where best_M is None if no solution found in [lo, hi].
        """
        t0 = time.time()
        solver, _ = self.build_formula()
        build_time = time.time() - t0

        best_M      = None
        best_matrix = None
        timed_out   = [False]

        def _interrupt():
            timed_out[0] = True
            try:
                solver.interrupt()
            except NotImplementedError:
                pass

        deadline = time.time() + timeout - build_time
        iteration = 0

        while lo <= hi:
            remaining = deadline - time.time()
            if remaining <= 0:
                timed_out[0] = True
                break

            mid = (lo + hi) // 2
            bM  = self.b_lit[mid]
            iteration += 1

            timer = threading.Timer(remaining, _interrupt)
            timer.start()
            try:
                sat = solver.solve_limited(assumptions=[bM], expect_interrupt=True)
                if sat:
                    model = solver.get_model()
            finally:
                timer.cancel()

            if timed_out[0] or sat is None:
                break

            if sat:
                matrix = self._extract_matrix(model)
                actual = self._compute_max_overlap(matrix)
                best_M      = actual
                best_matrix = matrix
                hi = actual - 1   # try to go lower
            else:
                lo = mid + 1      # must be at least mid+1

        elapsed   = time.time() - t0
        n_vars    = self._next_var - 1
        n_clauses = solver.nof_clauses()
        solver.delete()

        status = 'TIMEOUT' if timed_out[0] and best_M is None else \
                 'SAT'     if best_M is not None else \
                 'UNSAT'
        return status, best_M, best_matrix, elapsed, n_vars, n_clauses


def sat_worker(v, b, r, solver_name, encoding_name, sym_options,
               target_ub, timeout, quiet, out_queue, job_id):
    """
    Run OpdReifiedBound binary search in [lb, target_ub - 1].
    Puts SAT_RESULT into out_queue.
    """
    try:
        start_time = time.time()
        lb = compute_lower_bound(v, b, r)
        lo = lb
        hi = target_ub - 1   # search strictly below current best

        if lo > hi:
            out_queue.put({
                'type': 'SAT_RESULT', 'job_id': job_id, 'target_ub': target_ub,
                'status': 'UNSAT', 'lambda': None, 'matrix': None,
                'solve_time': 0.0, 'n_vars': 0, 'n_clauses': 0,
            })
            return

        solver_obj = OpdReifiedBound(
            v, b, r,
            solver_name=solver_name,
            encoding_name=encoding_name,
            sym_options=sym_options,
            verbose=not quiet,
        )
        status, best_M, best_matrix, elapsed, n_vars, n_clauses = \
            solver_obj.solve_range(lo, hi, timeout)

        out_queue.put({
            'type':       'SAT_RESULT',
            'job_id':     job_id,
            'target_ub':  target_ub,
            'status':     status,
            'lambda':     best_M,
            'matrix':     best_matrix,
            'solve_time': elapsed,
            'total_time': time.time() - start_time,
            'n_vars':     n_vars,
            'n_clauses':  n_clauses,
        })
    except Exception as e:
        import traceback
        out_queue.put({
            'type':      'SAT_ERROR',
            'job_id':    job_id,
            'target_ub': target_ub,
            'error':     str(e),
            'tb':        traceback.format_exc(),
        })


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------

def terminate_process(proc):
    if proc is None:
        return
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)


# ---------------------------------------------------------------------------
# Hybrid coordinator
# ---------------------------------------------------------------------------

class HybridCoordinator:
    """
    Coordinate Gurobi MIP and Incremental SAT (Reified Bound + Binary Search).

    - MIP runs continuously, emitting incumbents.
    - Each new best lambda triggers an IncSAT job searching [lb, best_lambda - 1].
    - If IncSAT finds a solution → update best + start new job.
    - If IncSAT returns UNSAT → best is OPTIMAL.
    """

    def __init__(self, v, b, r, timeout,
                 mip_sym=None, sat_sym=None,
                 sat_solver='cadical195', sat_encoding='sortnetwrk',
                 linearize=True, quiet=False):
        self.v = v
        self.b = b
        self.r = r
        self.timeout    = timeout
        self.mip_sym    = mip_sym or []
        self.sat_sym    = sat_sym if sat_sym is not None else list(self.mip_sym)
        self.sat_solver   = sat_solver
        self.sat_encoding = sat_encoding
        self.linearize  = linearize
        self.quiet      = quiet
        self.lambda_lb  = compute_lower_bound(v, b, r)

        self.best_lambda = None
        self.best_matrix = None
        self.best_source = None
        self.final_reason = None

        self.mip_queue      = mp.Queue()
        self.sat_queue      = mp.Queue()
        self.mip_stop_event = mp.Event()

        self.mip_proc = None
        self.sat_proc = None
        self.current_sat_job_id  = 0
        self.current_sat_target_ub = None

    def log(self, msg):
        if not self.quiet:
            print(msg)

    def start_mip(self):
        self.mip_proc = mp.Process(
            target=mip_worker,
            args=(self.v, self.b, self.r, self.mip_sym, self.linearize,
                  self.timeout, self.quiet, self.mip_queue, self.mip_stop_event),
        )
        self.mip_proc.start()

    def start_sat(self, target_ub, remaining_time):
        """Start IncSAT to search [lb, target_ub - 1]."""
        if target_ub - 1 < self.lambda_lb:
            self.log(f"[Coordinator] SAT target_ub={target_ub}: no room below lb={self.lambda_lb}")
            return
        # Already running for the same target
        if (self.sat_proc is not None
                and self.sat_proc.is_alive()
                and self.current_sat_target_ub == target_ub):
            return

        terminate_process(self.sat_proc)
        self.current_sat_job_id += 1
        job_id = self.current_sat_job_id
        self.current_sat_target_ub = target_ub

        self.sat_proc = mp.Process(
            target=sat_worker,
            args=(self.v, self.b, self.r,
                  self.sat_solver, self.sat_encoding, self.sat_sym,
                  target_ub, remaining_time, self.quiet,
                  self.sat_queue, job_id),
        )
        self.sat_proc.start()
        self.log(f"[Coordinator] Started IncSAT job {job_id}: search [λ_lb, {target_ub - 1}]")

    def stop_all(self):
        self.mip_stop_event.set()
        terminate_process(self.sat_proc)
        terminate_process(self.mip_proc)

    def update_best(self, lam, matrix, source):
        if lam is None:
            return False
        if self.best_lambda is None or lam < self.best_lambda:
            self.best_lambda = int(lam)
            self.best_matrix = matrix
            self.best_source = source
            self.log(f"[Coordinator] New best lambda = {self.best_lambda} from {source}")
            return True
        return False

    def ensure_sat_on_best(self, remaining_time):
        if self.best_lambda is None:
            return
        desired_ub = self.best_lambda  # search strictly below this
        if desired_ub <= self.lambda_lb:
            return
        self.start_sat(desired_ub, remaining_time)

    def solve(self):
        start_time = time.time()
        deadline   = start_time + self.timeout
        self.start_mip()

        if not self.quiet:
            print(f"Hybrid solve started: timeout={self.timeout}s, lb={self.lambda_lb}")
            print(f"MIP sym={self.mip_sym or 'none'}, "
                  f"SAT solver={self.sat_solver}, encoding={self.sat_encoding}, "
                  f"SAT sym={self.sat_sym or 'none'}")

        mip_finished = False

        try:
            while True:
                now = time.time()
                remaining = deadline - now
                if remaining <= 0:
                    self.final_reason = 'TIMEOUT'
                    break

                # ── Drain MIP messages ──────────────────────────────────────
                while True:
                    try:
                        msg = self.mip_queue.get_nowait()
                    except queue_mod.Empty:
                        break

                    mtype = msg.get('type')

                    if mtype == 'MIP_ERROR':
                        raise RuntimeError(f"MIP worker error: {msg['error']}")

                    if mtype == 'MIP_INCUMBENT':
                        improved = self.update_best(msg['lambda'], msg['matrix'], 'MIP')
                        if self.best_lambda == self.lambda_lb:
                            self.final_reason = 'LOWER_BOUND_REACHED'
                            self.stop_all()
                            return self._build_result(time.time() - start_time, 'OPTIMAL')
                        if improved:
                            self.ensure_sat_on_best(deadline - time.time())

                    elif mtype == 'MIP_PROVED_BY_LB':
                        self.update_best(msg['lambda'], msg['matrix'], 'MIP')
                        self.final_reason = 'MIP_PROVED_BY_LB'
                        self.stop_all()
                        return self._build_result(time.time() - start_time, 'OPTIMAL')

                    elif mtype == 'MIP_DONE':
                        mip_finished = True
                        if msg.get('lambda') is not None:
                            self.update_best(msg['lambda'], msg['matrix'], 'MIP')
                        status = msg.get('status', 'UNKNOWN')
                        if status == 'OPTIMAL':
                            self.final_reason = 'MIP_OPTIMAL'
                            self.stop_all()
                            return self._build_result(time.time() - start_time, 'OPTIMAL', extra=msg)
                        self.log(f"[Coordinator] MIP done: status={status}")
                        if self.best_lambda is not None:
                            self.ensure_sat_on_best(deadline - time.time())

                # ── Drain SAT messages ──────────────────────────────────────
                while True:
                    try:
                        msg = self.sat_queue.get_nowait()
                    except queue_mod.Empty:
                        break

                    mtype = msg.get('type')

                    if mtype == 'SAT_ERROR':
                        if msg.get('job_id') == self.current_sat_job_id:
                            self.log(f"[Coordinator] SAT error (job {msg['job_id']}): {msg['error']}")
                        continue

                    if mtype != 'SAT_RESULT':
                        continue
                    if msg.get('job_id') != self.current_sat_job_id:
                        continue  # stale result

                    target_ub = msg['target_ub']
                    status    = msg['status']
                    self.log(f"[Coordinator] IncSAT job {msg['job_id']} "
                             f"(ub={target_ub}) → {status}")
                    self.sat_proc = None
                    self.current_sat_target_ub = None

                    if status == 'SAT':
                        # IncSAT found a solution with lambda = msg['lambda']
                        improved = self.update_best(msg['lambda'], msg['matrix'], 'IncSAT')
                        if self.best_lambda is not None and self.best_lambda <= self.lambda_lb:
                            self.final_reason = 'SAT_REACHED_LOWER_BOUND'
                            self.stop_all()
                            return self._build_result(time.time() - start_time, 'OPTIMAL')
                        # Try to go lower
                        self.ensure_sat_on_best(deadline - time.time())

                    elif status == 'UNSAT':
                        # IncSAT proved no solution with lambda < target_ub
                        # => target_ub == best_lambda is OPTIMAL
                        if (self.best_lambda is not None
                                and target_ub == self.best_lambda):
                            self.final_reason = f'SAT_PROVED_UNSAT_BELOW_{target_ub}'
                            self.stop_all()
                            return self._build_result(time.time() - start_time, 'OPTIMAL')
                        self.log(f"[Coordinator] IncSAT UNSAT for ub={target_ub} "
                                 f"(best={self.best_lambda})")

                    elif status == 'TIMEOUT':
                        self.log(f"[Coordinator] IncSAT timeout for ub={target_ub}")

                # Keep IncSAT running if we have a best
                if self.best_lambda is not None:
                    self.ensure_sat_on_best(deadline - time.time())

                mip_alive = self.mip_proc is not None and self.mip_proc.is_alive()
                sat_alive = self.sat_proc is not None and self.sat_proc.is_alive()
                if mip_finished and not mip_alive and not sat_alive:
                    self.final_reason = 'NO_ACTIVE_WORKERS'
                    break

                time.sleep(0.05)

        finally:
            self.stop_all()

        if self.best_lambda is not None and self.best_lambda == self.lambda_lb:
            ovr_status = 'OPTIMAL'
        elif self.best_lambda is not None:
            ovr_status = 'BEST_FOUND'
        else:
            ovr_status = 'NO_SOLUTION'

        return self._build_result(time.time() - start_time, ovr_status)

    def _build_result(self, total_time, status, extra=None):
        extra = extra or {}
        return {
            'status':         status,
            'optimal_lambda': self.best_lambda,
            'matrix':         self.best_matrix,
            'time':           total_time,
            'lower_bound':    self.lambda_lb,
            'best_source':    self.best_source,
            'final_reason':   self.final_reason,
            'mip_status':     extra.get('status'),
            'mip_gap':        extra.get('gap'),
            'mip_nodes':      extra.get('nodes'),
            'mip_vars':       extra.get('n_vars'),
            'mip_constrs':    extra.get('n_constrs'),
            'mip_sym':        '+'.join(self.mip_sym) if self.mip_sym else 'none',
            'sat_solver':     self.sat_solver,
            'sat_encoding':   self.sat_encoding,
            'sat_sym':        '+'.join(self.sat_sym) if self.sat_sym else 'none',
        }


# ---------------------------------------------------------------------------
# File-level interface
# ---------------------------------------------------------------------------

def print_matrix(matrix):
    if matrix is None:
        return
    print("\nSolution matrix:")
    for i, row in enumerate(matrix):
        row_str = ''.join(str(int(x)) for x in row)
        print(f"  Row {i:2d}: {row_str} (sum={sum(row)})")


def solve_from_file(filepath, timeout=600,
                    mip_sym=None, sat_sym=None,
                    sat_solver='cadical195', sat_encoding='sortnetwrk',
                    quadratic=False, quiet=False):
    start_time = time.time()
    filename = os.path.basename(filepath)

    if not quiet:
        print(f"\n{'='*60}")
        print(f"Processing file: {filename}")
        print(f"{'='*60}")

    params = parse_input_file(filepath)
    if not params:
        if not quiet:
            print("Error: Could not extract v, b, r from file.")
        return {
            'File': filename, 'v': None, 'b': None, 'r': None,
            'Lower Bound': None, 'Optimal Lambda': None,
            'Best Source': None, 'Final Reason': None,
            'MIP Status': None, 'MIP Gap': None, 'MIP Nodes': None,
            'MIP Sym': '+'.join(mip_sym) if mip_sym else 'none',
            'SAT Solver': sat_solver, 'SAT Encoding': sat_encoding,
            'SAT Sym': '+'.join(sat_sym) if sat_sym else 'none',
            'Time (s)': 0, 'Status': 'Parse Error',
        }

    v, b, r = params['v'], params['b'], params['r']
    if not quiet:
        print(f"Parameters: v={v}, b={b}, r={r}")

    coordinator = HybridCoordinator(
        v=v, b=b, r=r, timeout=timeout,
        mip_sym=mip_sym, sat_sym=sat_sym,
        sat_solver=sat_solver, sat_encoding=sat_encoding,
        linearize=not quadratic,
        quiet=quiet,
    )

    try:
        result = coordinator.solve()
        total_time = time.time() - start_time

        result_data = {
            'File': filename,
            'v': v, 'b': b, 'r': r,
            'Lower Bound': result['lower_bound'],
            'Optimal Lambda': result['optimal_lambda'],
            'Best Source': result['best_source'],
            'Final Reason': result['final_reason'],
            'MIP Status': result['mip_status'],
            'MIP Gap': round(result['mip_gap'], 6) if result['mip_gap'] is not None else None,
            'MIP Nodes': result['mip_nodes'],
            'MIP Sym': result['mip_sym'],
            'SAT Solver': result['sat_solver'],
            'SAT Encoding': result['sat_encoding'],
            'SAT Sym': result['sat_sym'],
            'Time (s)': round(total_time, 3),
            'Status': result['status'],
        }

        if not quiet:
            print(f"\nRESULT for {filename}:")
            print(f"  Status       : {result['status']}")
            print(f"  Best lambda  : {result['optimal_lambda']}")
            print(f"  Lower bound  : {result['lower_bound']}")
            print(f"  Best source  : {result['best_source']}")
            print(f"  Final reason : {result['final_reason']}")
            print(f"  Total time   : {total_time:.3f}s")
            print_matrix(result['matrix'])

        return result_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'File': filename, 'v': v, 'b': b, 'r': r,
            'Lower Bound': compute_lower_bound(v, b, r),
            'Optimal Lambda': None, 'Best Source': None, 'Final Reason': None,
            'MIP Status': None, 'MIP Gap': None, 'MIP Nodes': None,
            'MIP Sym': '+'.join(mip_sym) if mip_sym else 'none',
            'SAT Solver': sat_solver, 'SAT Encoding': sat_encoding,
            'SAT Sym': '+'.join(sat_sym) if sat_sym else 'none',
            'Time (s)': round(time.time() - start_time, 3),
            'Status': f'Error: {str(e)}',
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hybrid OPD solver: Gurobi MIP + Incremental SAT (Reified Bound + Binary Search).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Opd_hybrid_mip_incsat.py --input src/input/small
  python Opd_hybrid_mip_incsat.py --input src/input/small/small_1.txt --timeout 300
  python Opd_hybrid_mip_incsat.py --input src/input/medium --timeout 3600 --sym lex_row r
  python Opd_hybrid_mip_incsat.py --input src/input/small --mip-sym r --sat-sym lex_row r
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input file or directory path')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Overall timeout per instance (seconds)')
    parser.add_argument('--solver', type=str, default='cadical195',
                        choices=['cadical195', 'glucose3', 'glucose4', 'minisat22', 'lingeling'],
                        help='SAT solver for Incremental SAT side')
    parser.add_argument('--encoding', type=str, default='sortnetwrk',
                        choices=list(ENCODING_MAP.keys()),
                        help='Cardinality encoding for SAT (default: sortnetwrk)')
    parser.add_argument('--sym', nargs='+', default=[],
                        help='Symmetry breaking for both MIP and SAT (e.g. lex_row lex_col r)')
    parser.add_argument('--mip-sym', nargs='+', default=None,
                        help='Symmetry breaking only for MIP')
    parser.add_argument('--sat-sym', nargs='+', default=None,
                        help='Symmetry breaking only for SAT')
    parser.add_argument('--quadratic', action='store_true',
                        help='Use quadratic overlap constraints in MIP (no linearization)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        potential = os.path.join('input', input_path)
        if os.path.exists(potential):
            input_path = potential

    mip_sym = args.mip_sym if args.mip_sym is not None else list(args.sym)
    sat_sym = args.sat_sym if args.sat_sym is not None else list(args.sym)

    COLUMNS = [
        'File', 'v', 'b', 'r', 'Lower Bound', 'Optimal Lambda',
        'Best Source', 'Final Reason',
        'MIP Status', 'MIP Gap', 'MIP Nodes', 'MIP Sym',
        'SAT Solver', 'SAT Encoding', 'SAT Sym', 'Time (s)', 'Status'
    ]

    if os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        files = sorted(f for f in os.listdir(input_path) if f.endswith('.txt'))
        if not files:
            print('No .txt files found.')
            sys.exit(0)

        print(f"Found {len(files)} input files.")

        try:
            import openpyxl
            has_openpyxl = True
        except ImportError:
            print('Warning: openpyxl not found. Excel export disabled.')
            has_openpyxl = False

        folder_name = os.path.basename(os.path.normpath(input_path))
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        output_file = f"result_{folder_name}_{script_name}.xlsx"

        for f in files:
            filepath = os.path.join(input_path, f)
            res = solve_from_file(
                filepath,
                timeout=args.timeout,
                mip_sym=mip_sym,
                sat_sym=sat_sym,
                sat_solver=args.solver,
                sat_encoding=args.encoding,
                quadratic=args.quadratic,
                quiet=args.quiet,
            )
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
        solve_from_file(
            input_path,
            timeout=args.timeout,
            mip_sym=mip_sym,
            sat_sym=sat_sym,
            sat_solver=args.solver,
            sat_encoding=args.encoding,
            quadratic=args.quadratic,
            quiet=args.quiet,
        )
    else:
        print(f"Error: Input path '{args.input}' not found.")
        sys.exit(1)
