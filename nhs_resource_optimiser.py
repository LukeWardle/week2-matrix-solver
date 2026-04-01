"""
nhs_resource_optimiser.py

NHS staff allocation optimiser demonstrating all three system types ina realistic healthcare resource planning context.

"""

import numpy as np
from linear_solver import solve_linear_system, analyze_system

LABELS = ["ED", "Surgery", "ICU", "Medicine"]

def optimise_staff_allocation():
  """
  Scenario 1: Feasible constraint set - unique solution.
  Variables: x = [ED, Surgery, ICU, Medicine]
  
  """
  print("\n" + "#"*65)
  print(" Scenario 1: FEASIBLE ALLOCATION")
  print("#"*65)

  # Constraint matrix
  A = np.array([
    [1.0, 1.0, 1.0, 1.0],   # Total: ED + Surg + ICU + Med = 100
    [0.2, 0.0, -1.0, 0.0],  # Cross-training: 0.2ED - ICY = -15
    [0.0, 1.0, 0.0, -2.0],  # Budget ratio: Surg - 2 * Med = 0
    [1.0, 0.0, 0.0, 0.0],   # Minimum ED: ED = 25
  ], dtype=float)

  b = np.array([100.0, -15.0, 0.0, 25.0], dtype=float)

  print("\n Constraints:")
  print(" 1. ED + Surgery + ICU + Medicine = 100 (total headcount)")
  print(" 2. 0.2·ED - ICU = -15 (cross-training)")
  print(" 3. Surgery - 2·Medicine = 0 (budget ratio)")
  print(" 4. ED = 25 (minimum safety)")

  x = solve_linear_system(A, b, verbose=True)

  if x is not None:
    print("\n   --  OPTIMAL ALLOCATION  --")
    for i, label in enumerate(LABELS):
      print(f"  {label:<12s}: {x[i]:6.1f} staff")
    print(f"  {'TOTAL':<12s}: {x.sum():6.1f} staff")

    print("\n ──── CONSTRAINT VERIFICATION ────")
    residuals = A @ x - b
    for i, res in enumerate(residuals):
      status="✓" if abs(res) < 0.01 else "✗"
      print(f" Constraint {i+1}: residual = {res:+.4f} {status}")
  return x

def demonstrate_infeasible_system():
  """
  Scenario 2: Impossible minimum requirements.

  Minimum requirements: ED≥30, Surgery≥35, ICU≥25, Medicine≥20
  Sum of minimums: 30+35+25+20 = 110 > 100 (total budget)

  This system is inconsistent - no feasible allocation exists.
  The solver should detect this and suggest remedies.
  
  """
  print("\n" + "#"*65)
  print(" SCENARIO 2: INFEASIBLE CONSTRAINTS")
  print("#"*65)

  A = np.array([
    [1.0, 1.0, 1.0, 1.0], # Total = 100
    [1.0, 0.0, 0.0, 0.0], # ED = 30
    [0.0, 1.0, 0.0, 0.0], # Surgery = 35
    [0.0, 0.0, 1.0, 0.0], # ICU = 25
    [0.0, 0.0, 0.0, 1.0], # Medicine = 20
  ], dtype=float)

  b = np.array([100.0, 30.0, 35.0, 25.0, 20.0], dtype=float)
  
  print("\n Minimum requirements:")
  print(" ED: 30 staff")
  print(" Surgery: 35 staff")
  print(" ICU: 25 staff")
  print(" Medicine: 20 staff")
  print(f" TOTAL: {30+35+25+20} staff > 100 budget (INFEASIBLE!)")
  
  x = solve_linear_system(A, b, verbose=True)
  if x is None:
    print("\n ──── RECOMMENDED ACTIONS FOR MANAGEMENT ────")
    print(" Option A: Reduce minimum requirements so they sum to ≤ 100")
    print(" Option B: Request additional headcount (increase budget)")
    print(" Option C: Implement cross-training to share staff across depts")

def demonstrate_ill_conditioned():
  """
  Scenario 3: Near-identical constraints cause ill-conditioning.

  Rows 1 and 2 are nearly identical — this is a data quality issue.

  The solver returns an answer, but flags high condition number.
  The solution should be treated with suspicion.

  """
  print("\n" + "#"*65)
  print(" SCENARIO 3: ILL-CONDITIONED SYSTEM")
  print("#"*65)
  A = np.array([
    [1.0, 1.0, 1.0, 1.0 ],
    [1.0, 1.0, 1.0, 1.0001 ],   # nearly identical to row 1
    [0.5, 0.5, 0.5, 0.5 ],      # = 0.5 * row 1
    [2.0, 0.0, 0.0, 0.0 ],
  ], dtype=float)

  b = np.array([100.0, 100.1, 50.0, 30.0], dtype=float)

  print("\n ⚠️ Warning: Row 2 is nearly identical to Row 1.")
  print(" This indicates a data quality problem in the constraints.")

  x = solve_linear_system(A, b, verbose=True)

  if x is not None:
    print("\n ──── COMPUTED ALLOCATION (UNRELIABLE) ────")
    for i, label in enumerate(LABELS):
      print(f" {label:<12s}: {x[i]:8.2f} staff")
      print("\n ──── DATA QUALITY RECOMMENDATIONS ────")
      print(" 1. Review constraint definitions for duplicate or")
      print(" near-duplicate rows")
      print(" 2. Investigate whether two constraints express the")
      print(" same operational requirement")
      print(" 3. Consider removing one of the redundant constraints")

def main():
  """Run all three NHS scenarios."""

  optimise_staff_allocation()
  demonstrate_infeasible_system()
  demonstrate_ill_conditioned()
  print("\n" + "="*65)
  print(" SESSION COMPLETE")
  print("="*65)

if __name__ == "__main__":
  main()