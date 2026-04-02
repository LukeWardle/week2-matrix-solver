# week2-matrix-solver
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Tests](https://img.shields.io/badge/tests-12%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
A production-grade linear system solver that classifies any system Ax = b
as **unique**, **inconsistent**, or **underdetermined** using the
Rouché–Capelli theorem, with SVD-based rank computation for numerical
reliability and condition number checking for stability warnings.
Built as part of Module 1 (Linear Algebra) of an AI Engineering programme.
---
## The Problem It Solves
Given a matrix A and vector b, find x such that Ax = b — or, if no such x
exists, explain precisely why not. Three outcomes are possible:
| System type | Condition | Solution |
|-----------------|------------------------------------|--------------------|
| Unique | rank(A) = rank([A|b]) = n | Exactly one x |
| Inconsistent | rank(A) < rank([A|b]) | No solution |
| Underdetermined | rank(A) = rank([A|b]) < n | Infinitely many |
The classification uses the **Rouché–Capelli theorem**. Rank is computed
via **SVD** rather than row reduction, because Gaussian elimination can
amplify floating-point rounding errors at each step, making the computed
rank unreliable for nearly-singular matrices.
---
## Quick Start (Windows)
```
git clone https://github.com/LukeWardle/week2-matrix-solver
cd week2-matrix-solver
python -m venv venv
venv\Scripts\activate
pip install numpy
```
---
## Usage
### Run all three system-type demonstrations
```
python demo.py
```
Expected output (abridged):
```
=== DEMO 1: Unique Solution ===
rank(A) = 2 | rank([A|b]) = 2 | n = 2
System type: UNIQUE SOLUTION
Condition number: 4.33
Solution: x = [1.6 1.8]
Residual ||Ax - b|| = 0.00000 (verify: Ax = b ✓)
=== DEMO 2: No Solution ===
rank(A) = 1 | rank([A|b]) = 2 | n = 2
System type: NO SOLUTION (inconsistent)
Rouche-Capelli: rank(A) < rank([A|b]) => no solution exists
=== DEMO 3: Infinite Solutions ===
rank(A) = 2 | rank([A|b]) = 2 | n = 3
System type: INFINITE SOLUTIONS (underdetermined)
Minimum-norm solution: x = [1.0 2.0 0.0]
```
### Run the NHS staff allocation scenarios
```
python nhs_resource_optimiser.py
```
### Run the test suite
```
python test_solver.py
```
Expected: `12 tests passed.`
---
## Project Structure
```
week2-matrix-solver/
├── linear_solver.py # Core solver: rank (SVD), condition number,
│ # Rouché–Capelli classification, three solution
│ # methods (solve / lstsq / infeasible report)
├── demo.py # Three demonstrations: unique, inconsistent,
│ # underdetermined — with inline commentary
├── test_solver.py # 12 unit tests: rank, condition number, all
│ # three system types, edge cases
├── nhs_resource_optimiser.py # NHS staff allocation: 3 scenarios
│ # (feasible, infeasible, ill-conditioned)
├── requirements.txt # numpy
└── README.md
```
---
## Mathematical Background
### The Rouché–Capelli Theorem
For the system Ax = b, let [A|b] denote the augmented matrix:
- rank(A) = rank([A|b]) and both equal n → **unique solution**
- rank(A) = rank([A|b]) but both < n → **infinite solutions**
- rank(A) < rank([A|b]) → **no solution**
### Why SVD for Rank?
SVD decomposes A = UΣVᵀ. The rank equals the number of non-negligible
singular values (diagonal entries of Σ above a tolerance). Unlike row
reduction, SVD does not accumulate floating-point errors — it is
numerically stable even for nearly-singular matrices.
### Condition Number
κ(A) = σ_max / σ_min (ratio of largest to smallest singular value).
A condition number of 10⁸ means a 1% change in input data could cause
a 10⁸% change in the computed solution. The solver warns when κ > 1e8.
---
## NHS Application
nhs_resource_optimiser.py models a hospital trust allocating 100 staff
across four departments (ED, Surgery, ICU, Medicine):
- **Scenario 1 — Feasible:** Consistent constraints → unique allocation
- **Scenario 2 — Infeasible:** Minimum requirements exceed total staff →
inconsistent system, solver reports the contradiction
- **Scenario 3 — Ill-conditioned:** Two constraints are nearly identical →
condition number warning, solution is technically valid but unreliable
---
## Limitations
- Requires numpy; no other dependencies, but no sparse matrix support
- Condition number threshold (1e8) is a default, not a universal rule
- Minimum-norm solution for underdetermined systems (via lstsq) is one of
infinitely many valid solutions — not necessarily the most useful one
- No iterative solver for large systems (direct methods only)
- Assumes real-valued matrices; no complex number support
---
## Author
Luke Wardle | Week 2 Thursday Code Session | Module 1: Linear Algebra
Built as part of the AI Engineering programme — UK cohort.
