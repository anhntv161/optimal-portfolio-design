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
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver


ENCODING_MAP = {
    'sortnetwrk': EncType.sortnetwrk,
    'cardnetwrk': EncType.cardnetwrk,
    'seqcounter': EncType.seqcounter,
    'totalizer': EncType.totalizer,
    'mtotalizer': EncType.mtotalizer,
    'pairwise': EncType.pairwise,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def parse_input_file(filepath):
    """Parse v, b, r from input file."""
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


def compute_lower_bound(v, b, r):
    """Compute theoretical lower bound on lambda."""
    if v <= 1:
        return 0
    numerator = r * (r * v - b)
    denominator = b * (v - 1)
    if denominator == 0:
        return 0
    return max(0, math.ceil(numerator / denominator))


# ---------------------------------------------------------------------------
# MIP side (Gurobi)
# ---------------------------------------------------------------------------

class PortfolioMIP:
    """MIP-based solver for OPD."""

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

    def build_model(self, use_symmetry_breaking=True, linearize=True):
        self.model = gp.Model("opd_hybrid_mip")

        if not self.verbose:
            self.model.setParam('OutputFlag', 0)

        self.x = self.model.addVars(self.v, self.b, vtype=GRB.BINARY, name="x")
        self.lambda_var = self.model.addVar(vtype=GRB.INTEGER, lb=self.lambda_lb, ub=self.r, name="lambda")

        for i in range(self.v):
            self.model.addConstr(
                gp.quicksum(self.x[i, j] for j in range(self.b)) == self.r,
                f"row_sum_{i}"
            )

        if linearize:
            self.y = self.model.addVars(
                [(i1, i2, j) for i1 in range(self.v) for i2 in range(i1 + 1, self.v) for j in range(self.b)],
                vtype=GRB.BINARY,
                name="y"
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

        if use_symmetry_breaking:
            self._add_symmetry_breaking()
        else:
            self.applied_sym = 'none'

        self.model.setObjective(self.lambda_var, GRB.MINIMIZE)
        self.model.update()

        self.num_vars = self.model.NumVars
        self.num_constrs = self.model.NumConstrs + (self.model.NumQConstrs if hasattr(self.model, 'NumQConstrs') else 0)

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

    @staticmethod
    def verify_solution(matrix, r, target_lambda):
        v = len(matrix)
        b = len(matrix[0]) if matrix else 0

        for i in range(v):
            if int(np.sum(matrix[i])) != r:
                return False, -1

        max_overlap = 0
        for i1 in range(v):
            for i2 in range(i1 + 1, v):
                overlap = int(np.sum(matrix[i1] * matrix[i2]))
                max_overlap = max(max_overlap, overlap)
                if overlap > target_lambda:
                    return False, overlap
        return True, max_overlap


def _extract_cb_matrix(model):
    v = model._v
    b = model._b
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
            matrix = _extract_cb_matrix(model)
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
        solver.build_model(use_symmetry_breaking=True, linearize=linearize)

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
# SAT side (PySAT)
# ---------------------------------------------------------------------------

class OpdSAT:
    """SAT-based solver for OPD."""

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

    def add_exactly_r(self, solver, vars_):
        cnf = CardEnc.equals(lits=vars_, bound=self.r, top_id=self.var_id, encoding=self.encoding_type)
        solver.append_formula(cnf)
        self.var_id = cnf.nv

    def add_at_most_k(self, solver, vars_, k):
        if k >= len(vars_):
            return
        cnf = CardEnc.atmost(lits=vars_, bound=k, top_id=self.var_id, encoding=self.encoding_type)
        solver.append_formula(cnf)
        self.var_id = cnf.nv

    def add_row_constraints(self, solver):
        for i in range(self.v):
            self.add_exactly_r(solver, self.x[i])

    def add_overlap_constraints(self, solver, max_overlap):
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

    def _add_lex_constraint(self, solver, a, b_vec, n):
        solver.add_clause([-a[0], b_vec[0]])
        prev_p = None

        for k in range(n - 1):
            p_k = self.var_id
            self.var_id += 1

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

    def add_symmetry_breaking(self, solver):
        fixed_r = 'r' in self.sym_options
        use_lex_row = 'lex_row' in self.sym_options
        use_lex_col = 'lex_col' in self.sym_options

        if fixed_r:
            for j in range(self.r):
                solver.add_clause([self.x[0][j]])
            for j in range(self.r, self.b):
                solver.add_clause([-self.x[0][j]])

        if use_lex_row:
            for i in range(self.v - 1):
                if fixed_r:
                    self._add_lex_constraint(solver, self.x[i + 1], self.x[i], self.b)
                else:
                    self._add_lex_constraint(solver, self.x[i], self.x[i + 1], self.b)

        if use_lex_col:
            for j in range(self.b - 1):
                col_j = [self.x[i][j] for i in range(self.v)]
                col_j1 = [self.x[i][j + 1] for i in range(self.v)]
                if fixed_r:
                    self._add_lex_constraint(solver, col_j1, col_j, self.v)
                else:
                    self._add_lex_constraint(solver, col_j, col_j1, self.v)

    def extract_matrix(self, model):
        model_set = set(model)
        matrix = []
        for i in range(self.v):
            row = []
            for j in range(self.b):
                row.append(1 if self.x[i][j] in model_set else 0)
            matrix.append(row)
        return matrix

    @staticmethod
    def verify_solution(matrix, r, target_lambda):
        v = len(matrix)
        b = len(matrix[0]) if matrix else 0

        for i in range(v):
            if sum(matrix[i]) != r:
                return False, -1

        max_overlap = 0
        for i1, i2 in itertools.combinations(range(v), 2):
            overlap = sum(matrix[i1][j] * matrix[i2][j] for j in range(b))
            max_overlap = max(max_overlap, overlap)
            if overlap > target_lambda:
                return False, overlap
        return True, max_overlap

    def solve_with_max_overlap(self, max_overlap, timeout):
        solver = Solver(name=self.solver_name)
        start_time = time.time()

        self.add_row_constraints(solver)
        self.add_overlap_constraints(solver, max_overlap)
        self.add_symmetry_breaking(solver)

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
            solver.delete()
            return 'TIMEOUT', None, solve_time, n_vars, n_clauses

        if sat:
            model = solver.get_model()
            matrix = self.extract_matrix(model)
            valid, _ = self.verify_solution(matrix, self.r, max_overlap)
            solver.delete()
            if valid:
                return 'SAT', matrix, solve_time, n_vars, n_clauses
            return 'INVALID', matrix, solve_time, n_vars, n_clauses

        solver.delete()
        return 'UNSAT', None, solve_time, n_vars, n_clauses



def sat_worker(v, b, r, solver_name, encoding_name, sym_options, target_lambda, timeout, quiet, out_queue, job_id):
    try:
        start_time = time.time()
        solver = OpdSAT(v, b, r, solver_name=solver_name, encoding_name=encoding_name,
                        sym_options=sym_options, verbose=not quiet)
        status, matrix, solve_time, n_vars, n_clauses = solver.solve_with_max_overlap(target_lambda, timeout)
        out_queue.put({
            'type': 'SAT_RESULT',
            'job_id': job_id,
            'target': target_lambda,
            'status': status,
            'matrix': matrix,
            'solve_time': solve_time,
            'total_time': time.time() - start_time,
            'n_vars': n_vars,
            'n_clauses': n_clauses,
        })
    except Exception as e:
        out_queue.put({
            'type': 'SAT_ERROR',
            'job_id': job_id,
            'target': target_lambda,
            'error': str(e),
        })


# ---------------------------------------------------------------------------
# Hybrid coordinator
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


class HybridCoordinator:
    """Coordinate Gurobi MIP and SAT exact checks on incumbent-1."""

    def __init__(self, v, b, r, timeout, mip_sym=None, sat_sym=None,
                 sat_solver='cadical195', sat_encoding='sortnetwrk',
                 linearize=True, quiet=False):
        self.v = v
        self.b = b
        self.r = r
        self.timeout = timeout
        self.mip_sym = mip_sym or []
        self.sat_sym = sat_sym if sat_sym is not None else list(self.mip_sym)
        self.sat_solver = sat_solver
        self.sat_encoding = sat_encoding
        self.linearize = linearize
        self.quiet = quiet
        self.lambda_lb = compute_lower_bound(v, b, r)

        self.best_lambda = None
        self.best_matrix = None
        self.best_source = None
        self.final_reason = None

        self.mip_queue = mp.Queue()
        self.sat_queue = mp.Queue()
        self.mip_stop_event = mp.Event()

        self.mip_proc = None
        self.sat_proc = None
        self.current_sat_target = None
        self.current_sat_job_id = 0

    def log(self, msg):
        if not self.quiet:
            print(msg)

    def start_mip(self):
        self.mip_proc = mp.Process(
            target=mip_worker,
            args=(self.v, self.b, self.r, self.mip_sym, self.linearize,
                  self.timeout, self.quiet, self.mip_queue, self.mip_stop_event)
        )
        self.mip_proc.start()

    def start_sat(self, target, remaining_time):
        if target < self.lambda_lb:
            return
        self.current_sat_job_id += 1
        job_id = self.current_sat_job_id
        terminate_process(self.sat_proc)
        self.sat_proc = mp.Process(
            target=sat_worker,
            args=(self.v, self.b, self.r, self.sat_solver, self.sat_encoding,
                  self.sat_sym, target, remaining_time, self.quiet,
                  self.sat_queue, job_id)
        )
        self.current_sat_target = target
        self.sat_proc.start()
        self.log(f"[Coordinator] Started SAT check for lambda <= {target}")

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

    def ensure_sat_matches_best(self, remaining_time):
        if self.best_lambda is None:
            return
        desired_target = self.best_lambda - 1
        if desired_target < self.lambda_lb:
            return
        if self.sat_proc is not None and self.sat_proc.is_alive() and self.current_sat_target == desired_target:
            return
        self.start_sat(desired_target, remaining_time)

    def solve(self):
        start_time = time.time()
        deadline = start_time + self.timeout
        self.start_mip()

        if not self.quiet:
            print(f"Hybrid solve started with timeout={self.timeout}s")
            print(f"Theoretical lower bound = {self.lambda_lb}")

        mip_finished = False

        try:
            while True:
                now = time.time()
                remaining_time = deadline - now
                if remaining_time <= 0:
                    self.final_reason = 'TIMEOUT'
                    break

                # Drain MIP messages
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
                            self.ensure_sat_matches_best(remaining_time)

                    elif mtype == 'MIP_PROVED_BY_LB':
                        self.update_best(msg['lambda'], msg['matrix'], 'MIP')
                        self.final_reason = 'MIP_INCUMBENT_EQUALS_LB'
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
                        self.log(f"[Coordinator] MIP finished with status {status}")

                # Drain SAT messages
                while True:
                    try:
                        msg = self.sat_queue.get_nowait()
                    except queue_mod.Empty:
                        break

                    mtype = msg.get('type')
                    if mtype == 'SAT_ERROR':
                        if msg.get('job_id') == self.current_sat_job_id:
                            self.log(f"[Coordinator] SAT worker error for target {msg.get('target')}: {msg.get('error')}")
                        continue

                    if mtype != 'SAT_RESULT':
                        continue

                    if msg.get('job_id') != self.current_sat_job_id:
                        # stale SAT result from an older target, ignore
                        continue

                    target = msg['target']
                    status = msg['status']
                    self.log(f"[Coordinator] SAT finished target {target} with status {status}")
                    self.sat_proc = None

                    if status == 'UNSAT':
                        # If SAT proves infeasible for best_lambda - 1, then best_lambda is optimal.
                        if self.best_lambda is not None and target == self.best_lambda - 1:
                            self.final_reason = f'SAT_PROVED_UNSAT_AT_{target}'
                            self.stop_all()
                            return self._build_result(time.time() - start_time, 'OPTIMAL')

                    elif status == 'SAT':
                        improved = self.update_best(target, msg['matrix'], 'SAT')
                        if self.best_lambda == self.lambda_lb:
                            self.final_reason = 'SAT_REACHED_LOWER_BOUND'
                            self.stop_all()
                            return self._build_result(time.time() - start_time, 'OPTIMAL')
                        if improved:
                            self.ensure_sat_matches_best(deadline - time.time())
                        else:
                            self.ensure_sat_matches_best(deadline - time.time())

                    elif status == 'TIMEOUT':
                        self.log(f"[Coordinator] SAT timeout on target {target}")

                if self.best_lambda is not None:
                    self.ensure_sat_matches_best(deadline - time.time())

                # Exit if both workers are inactive and nothing else can happen
                mip_alive = self.mip_proc is not None and self.mip_proc.is_alive()
                sat_alive = self.sat_proc is not None and self.sat_proc.is_alive()
                if mip_finished and not mip_alive and not sat_alive:
                    self.final_reason = 'NO_ACTIVE_WORKERS'
                    break

                time.sleep(0.05)

        finally:
            self.stop_all()

        if self.best_lambda is not None and self.best_lambda == self.lambda_lb:
            status = 'OPTIMAL'
        elif self.best_lambda is not None:
            status = 'BEST_FOUND'
        else:
            status = 'NO_SOLUTION'
        return self._build_result(time.time() - start_time, status)

    def _build_result(self, total_time, status, extra=None):
        extra = extra or {}
        return {
            'status': status,
            'optimal_lambda': self.best_lambda,
            'matrix': self.best_matrix,
            'time': total_time,
            'lower_bound': self.lambda_lb,
            'best_source': self.best_source,
            'final_reason': self.final_reason,
            'mip_status': extra.get('status'),
            'mip_gap': extra.get('gap'),
            'mip_nodes': extra.get('nodes'),
            'mip_vars': extra.get('n_vars'),
            'mip_constrs': extra.get('n_constrs'),
            'mip_sym': '+'.join(self.mip_sym) if self.mip_sym else 'none',
            'sat_solver': self.sat_solver,
            'sat_encoding': self.sat_encoding,
            'sat_sym': '+'.join(self.sat_sym) if self.sat_sym else 'none',
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


def solve_from_file(filepath, timeout=600, mip_sym=None, sat_sym=None,
                    sat_solver='cadical195', sat_encoding='sortnetwrk',
                    quadratic=False, quiet=False):
    start_time = time.time()
    filename = os.path.basename(filepath)

    if not quiet:
        print(f"\n{'='*60}")
        print(f"Processing file: {filename}")
        print(f"{'='*60}")

    params = parse_input_file(filepath)
    if not params or not all(k in params for k in ('v', 'b', 'r')):
        if not quiet:
            print("Error: Could not extract v, b, r from file.")
        return {
            'File': filename,
            'v': None,
            'b': None,
            'r': None,
            'Optimal Lambda': None,
            'Lower Bound': None,
            'Best Source': None,
            'Time (s)': 0,
            'Status': 'Parse Error',
        }

    v, b, r = params['v'], params['b'], params['r']
    if not quiet:
        print(f"Parameters: v={v}, b={b}, r={r}")

    coordinator = HybridCoordinator(
        v=v,
        b=b,
        r=r,
        timeout=timeout,
        mip_sym=mip_sym,
        sat_sym=sat_sym,
        sat_solver=sat_solver,
        sat_encoding=sat_encoding,
        linearize=not quadratic,
        quiet=quiet,
    )

    try:
        result = coordinator.solve()
        total_time = time.time() - start_time

        result_data = {
            'File': filename,
            'v': v,
            'b': b,
            'r': r,
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
            print(f"Status: {result['status']}")
            print(f"Best lambda: {result['optimal_lambda']}")
            print(f"Lower bound: {result['lower_bound']}")
            print(f"Best source: {result['best_source']}")
            print(f"Final reason: {result['final_reason']}")
            print(f"Total time: {total_time:.3f}s")
            print_matrix(result['matrix'])

        return result_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'File': filename,
            'v': v,
            'b': b,
            'r': r,
            'Lower Bound': compute_lower_bound(v, b, r),
            'Optimal Lambda': None,
            'Best Source': None,
            'Final Reason': None,
            'MIP Status': None,
            'MIP Gap': None,
            'MIP Nodes': None,
            'MIP Sym': '+'.join(mip_sym) if mip_sym else 'none',
            'SAT Solver': sat_solver,
            'SAT Encoding': sat_encoding,
            'SAT Sym': '+'.join(sat_sym) if sat_sym else '+'.join(mip_sym) if mip_sym else 'none',
            'Time (s)': round(time.time() - start_time, 3),
            'Status': f'Error: {str(e)}',
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hybrid OPD solver: Gurobi MIP provides incumbents, SAT checks incumbent-1 in parallel.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Opd_hybrid_mip_sat.py --input input/small/test1.txt --timeout 300
  python Opd_hybrid_mip_sat.py --input input/small --timeout 600 --sym lex_row lex_col r
  python Opd_hybrid_mip_sat.py --input input/small/test1.txt --solver cadical195 --encoding sortnetwrk
        """
    )

    parser.add_argument('--input', type=str, required=True, help='Input file or directory path')
    parser.add_argument('--timeout', type=int, default=3600, help='Overall timeout per instance (seconds)')
    parser.add_argument('--solver', type=str, default='cadical195',
                        choices=['cadical195', 'glucose3', 'glucose4', 'minisat22', 'lingeling'],
                        help='SAT solver to use')
    parser.add_argument('--encoding', type=str, default='sortnetwrk',
                        choices=['sortnetwrk', 'cardnetwrk', 'seqcounter', 'totalizer', 'mtotalizer', 'pairwise'],
                        help='Cardinality encoding for SAT')
    parser.add_argument('--sym', nargs='+', default=[], help='Shared symmetry breaking methods (e.g. lex_row lex_col r)')
    parser.add_argument('--mip-sym', nargs='+', default=None, help='Symmetry breaking only for MIP')
    parser.add_argument('--sat-sym', nargs='+', default=None, help='Symmetry breaking only for SAT')
    parser.add_argument('--quadratic', action='store_true', help='Use quadratic overlap constraints in MIP instead of linearization')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        potential_path = os.path.join('input', input_path)
        if os.path.exists(potential_path):
            input_path = potential_path

    mip_sym = args.mip_sym if args.mip_sym is not None else list(args.sym)
    sat_sym = args.sat_sym if args.sat_sym is not None else list(args.sym)

    if os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        files = sorted([f for f in os.listdir(input_path) if f.endswith('.txt')])

        if not files:
            print('No .txt files found in directory.')
        else:
            print(f"Found {len(files)} input files.")
            try:
                import openpyxl
                has_openpyxl = True
            except ImportError:
                print('Warning: openpyxl not found. Excel export will be disabled.')
                has_openpyxl = False

            folder_name = os.path.basename(os.path.normpath(input_path))
            script_name = os.path.splitext(os.path.basename(__file__))[0]
            output_file = f"result_{folder_name}_{script_name}.xlsx"
            columns = [
                'File', 'v', 'b', 'r', 'Lower Bound', 'Optimal Lambda', 'Best Source', 'Final Reason',
                'MIP Status', 'MIP Gap', 'MIP Nodes', 'MIP Sym',
                'SAT Solver', 'SAT Encoding', 'SAT Sym', 'Time (s)', 'Status'
            ]

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
                        ws.append(columns)
                    else:
                        wb = openpyxl.load_workbook(output_file)
                        ws = wb.active
                    ws.append([res.get(c) for c in columns])
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
