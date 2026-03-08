# A SAT/MaxSAT-Based Approach to Optimal Financial Portfolio Design: A Comparison with CP and MIP methods

This research project explores and compares different computational approaches to the **Optimal Portfolio Design (OPD)** problem, which is a specialized variant of the **Orthogonal Latin Defect** problem. The core objective is to design optimal portfolio structures by minimizing the maximum overlap between sub-pools while satisfying strict balancing and weight constraints.

The project provides a comprehensive comparison between several state-of-the-art modeling and solving paradigms:
- **SAT (Satisfiability)**: Leveraging CDCL solvers (CaDiCaL, Glucose) with reified cardinality constraints and binary search strategies.
- **MaxSAT (Maximum Satisfiability)**: Modeling the overlap minimization as a soft constraint optimization problem.
- **ILP/MIP (Integer/Mixed Integer Programming)**: Utilizing commercial solvers like Gurobi and CPLEX, as well as open-source alternatives via OR-Tools and PuLP.
- **CP (Constraint Programming)**: Implementing models for ACE and CPLEX CP Optimizer to evaluate performance against traditional combinatorial methods.

## Project Structure

- `src/`: Core source code and implementation scripts.
    - `input/`: Standardized dataset instances (small, medium, large).
    - `Opd.py`: Primary script for problem transformation and XML model generation for CP solvers.
    - `Opd_pure_sat_*.py`: CDCL-based SAT solvers using various encoding strategies (e.g., Sequential Counter, Totalizer).
    - `Opd_ilp.py` & `Opd_mip_*.py`: Mathematical programming models supporting Gurobi, CPLEX, and SCIP.
    - `Opd_reified_bound_sat_*.py`: Advanced SAT implementation using once-built cardinality structures with assumption-based searching.
    - `Opd_maxsat_*.py`: MaxSAT formulations for direct optimization of overlap bounds.
    - `run.sh`: Automated pipeline for CP solver execution.
- `references/`: Supporting documents and mathematical background.

## System Requirements

The project requires Python 3.x and the following core dependencies:

```bash
pip install python-sat pulp docplex gurobipy numpy openpyxl pandas pycsp3 ortools
```

> [!NOTE]
> Commercial solvers (Gurobi, CPLEX) require valid licenses for large-scale instances, though academic licenses are often available.

## Installation & Build Instructions

### 1. Install Dependencies
Initialize your environment and install the minimal required packages:
```bash
cd src
pip install -r requirements.txt
```

### 2. Set Up EvalMaxSAT Solver
`EvalMaxSAT` is an external high-performance MaxSAT solver used in our benchmarks. It must be cloned and built locally:

```bash
cd src
git clone https://github.com/FlorentAvellaneda/EvalMaxSAT.git
cd EvalMaxSAT
mkdir build && cd build
cmake ..
make
```

## Usage

Most solvers support a unified interface using the `--input` flag to process either a single instance or an entire directory.

### 1. Running SAT Solvers
To solve using the CaDiCaL-based SAT approach with binary search:
```bash
python src/Opd_pure_sat_ver1_cadical.py --input input/small/small_1.txt
```

### 2. Running ILP/MIP Solvers
To solve using mathematical programming (e.g., using Gurobi or CPLEX via OR-Tools/PuLP):
```bash
# Default binary search approach
python src/Opd_ilp.py --input input/small

# Direct optimization approach
python src/Opd_ilp.py --input input/small --method direct
```

### 3. Running CP Pipeline
To generate models and solve using the Java-based ACE solver:
```bash
cd src
./run.sh
```

## Results & Verification
The framework automatically logs execution metrics and verifies solution validity (row weights and overlap bounds). Results are exported to Excel files (e.g., `result_small_Opd_ilp.xlsx`) for easy comparative analysis across different solvers and heuristics.

---
*This repository serves as a benchmarking framework for combinatorial optimization in financial engineering.*
