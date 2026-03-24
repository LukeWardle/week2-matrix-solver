"""
demo.py

Demonstration of the linear solver with all three system types.

Each example includes:
  - The mathematical system setup
  - Why this system has the type it does
  - Verification of the solution

"""

import numpy as np
from linear_solver import solve_linear_system, analyze_system

def demo_unique_solution():
  """
  System with a unique solution.

  Equations: 2x₁ +  x₂ = 5
              x₁ + 3x₂ = 7
  
  Solution by elimination:
  From row 1: x₂ = 5 - 2x₁
  Substitute into row 2: x₁ + 3(5 - 2x₁) = 7
    x₁ + 15 - 6x₁ = 7  →  -5x₁ = -8  →  x₁ = 1.6
    x₂ = 5 - 2(1.6) = 1.8 
  Expected solution: x = [1.6, 1.8]
  
  """
  print("\n" + "="*65)
  print(" Demo 1: UNIQUE SOLUTION")
  print("="*65)

  A = np.array([[2, 1],
                [1, 3]], dtype=float)
  b = np.array([5, 7], dtype=float)

  print(f"\n  System:  2x₁ +  x₂ = 5") 
  print(f"           x₁ + 3x₂ = 7") 
  print(f"  Expected: x₁ = 1.6, x₂ = 1.8")

  x = solve_linear_system(A, b, verbose=True)

  if x is not None:
    print(f"\n  Solution: x = {x}")
    verify = A @ x
    print(f"  Verify A@x = {verify} (should be {b})")
    print(f"  Match:  {np.allclose(verify, b)}")

def demo_no_solution():
  """
  Inconsistent system - no solution exists.

  Equations: x₁ + 2x₂ = 3
    2x₁ + 4x₂ = 5 ← should be 6 if consistent

  Row 2 = 2 × Row 1 implies: 2x₁ + 4x₂ = 2×3 = 6
  But b[1] = 5 ≠ 6. Contradiction — no solution.

  rank(A) = 1 (rows are linearly dependent)
  rank([A|b]) = 2 (b adds a new independent direction)
  Since rank(A) < rank([A|b]): INCONSISTENT
  
  """

  print("\n" + "="*65)
  print(" DEMO 2: NO SOLUTION (INCONSISTENT SYSTEM)")
  print("="*65)

  A = np.array([[1,2],
                [2, 3]], dtype=float) # singular: row2 = 2*row1
  b = np.array([3,5], dtype=float) # inconsistent: 5 ≠ 2*3 = 6

  print("\n equations:  x₁ + 2x₂ = 3")
  print(" 2x₁ + 4x₂ = 5")
  print(" Note: Row 2 = 2×Row 1, but b[1]=5 ≠ 2×3=6. Contradiction.")

  x = solve_linear_system(A, b, verbose=True)

  if x is None:
    print(" Confirmed: solver correctly retunred None.")
    print(" Action required: review the constraint definitions.")

def demo_infinite_solutions():
  """
  Underdetermined system - infinitely many solutions.

  Equations: x₁ + 2x₂ + x₃ = 5
             2x₁ + 4x₂ + 3x₃ = 11

  2 equations, 3 unknowns: one free variable.
  rank(A) = rank([A|b]) = 2 < n = 3: INFINITE solutions.

  Any x = particular_solution + t * null_space_vector
  satiesfies the system for any scalar t.
  The minimum-norm solution minimises ||x||₂ across all choices of t.
  
  """
  print("\n" + "="*65)
  print(" DEMO 3: INFINITE SOLUTIONS (UNDERDETERMINED)")
  print("="*65)
  A = np.array([[1, 2, 1],
                [2, 4, 3]], dtype=float)
  b = np.array([5, 11], dtype=float)
  print("\n Equations: x₁ + 2x₂ + x₃ = 5")
  print(" 2x₁ + 4x₂ + 3x₃ = 11")
  print(" Note: 2 equations, 3 unknowns — underdetermined.")
  
  x = solve_linear_system(A, b, verbose=True)

  if x is not None:
    print(f"\n Minimum-norm solution: x = {np.round(x, 4)}")
    verify = A @ x
    print(f" Verify A@x = {np.round(verify, 6)} (should be {b})")
    print(f" Match: {np.allclose(verify, b)}")

    # Show that other solutions also satisfy the system
    print("\n Another valid solution (add null space vector):")
    null_vec = np.array([-2, 1, 0]) # in null space of A
    x_alt = x + 0.5 * null_vec
    verify_alt = A @ x_alt
    print(f" x_alt = x + 0.5*null_vec = {np.round(x_alt, 4)}")
    print(f" A@x_alt = {np.round(verify_alt, 6)} (also satisfies!)")

if __name__ == "__main__":
  demo_unique_solution()
  demo_no_solution()
  demo_infinite_solutions()