"""
test_solver.py

Comprehensive test suite for linear_solver module.

Tests cover:
  - Rank computation (full rank, rank-deficient, tolerance)
  - System analysis (all three types)
  - Solver outputs (unique solutions, None for inconsistent, minimum-norm for infinite)
  - Condition number (well-conditioned, ill-conditioned)
  - Verification: solutions satisfy the original equations

"""

import numpy as np
from linear_solver import (
  compute_rank,
  compute_condition_number,
  analyze_system,
  solve_linear_system,
)

def test_rank_full_rank():
  """Full-rank 2x2 matrix should have rank 2."""
  A = np.array([[1, 2], [3, 4]], dtype=float)
  assert compute_rank(A) == 2, "DFull-rank 2x2 should have rank 2"
  print("✓ rank: full-rank 2x2 = 2")

def test_rank_rank_deficient():
  """Linearly dependent rows: rank should be 1."""
  A = np.array([[1, 2], [2, 4]], dtype=float) # row2 = 2*row1
  assert compute_rank(A) == 1, "Rank-deficient 2x2 should have rank 1"
  print("✓ rank: rank-deficient matrix = 1")

def test_rank_zero_matrix():
  """Zero matrix has rank 0."""
  Z = np.zeros((3, 4))
  assert compute_rank(Z) == 0, "Zero matrix should have rank 0"
  print("✓ rank: zero matrix = 0")

def test_rank_tall_matrix():
  """More rows than columns: max rank = n (columns)."""
  A = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
  assert compute_rank(A) == 2, "3x2 full-column-rank should be 2"
  print("✓ rank: tall full-column-rank matrix = 2")

def test_analyze_unique():
  """Full-rank square system -> unique type."""
  A = np.array([[2, 1], [1, 3]], dtype=float)
  b = np.array([5, 7], dtype=float)
  result = analyze_system(A, b)
  assert result['system_type'] == 'unique', (
    f"Expected unique, got {result['system_type']}"
  )
  print("✓ analyze: full-rank square → unique")

def test_analyze_inconsistent():
  """Singular A with contradictory b -> inconsistent type."""
  A = np.array([[1, 2], [2, 4]], dtype=float)
  b = np.array([3, 5], dtype=float) # 5 ≠ 2*3
  result = analyze_system(A, b)
  assert result['system_type'] == 'inconsistent', (
    f"Expected inconsistent, got {result['system_type']}"
  )
  print("✓ analyze: singular A, contradictory b → inconsistent")

def test_analyze_infinite():
  """Underdetermined consistent system -> infinite type."""
  A = np.array([[1, 2, 1], [2, 4, 3]], dtype=float)
  b = np.array([5, 11], dtype=float)
  result = analyze_system(A, b)
  assert result['system_type'] == 'infinite', (
    f"Expected infinite, got {result['system_type']}"
  )
  print("✓ analyze: underdetermined consistent → infinite")

def test_solve_unique_and_verify():
  """Unique system: solution must satisfy A @ x = b"""
  A = np.array([[2, 1], [1, 3]], dtype=float)
  b = np.array([5, 7], dtype=float)
  x = solve_linear_system(A, b, verbose=False)
  assert x is not None, "Should return a solution"
  assert np.allclose(A @ x, b, atol=1e-10), (
    f"Solution does not satisfy equations: A@x = {A @ x}, b = {b}"
  )
  print("✓ solve: unique solution satisfies A@x = b")

def test_solve_inconsistent_returns_none():
  """Inconsistent system: solver must return None."""
  A = np.array([[1, 2], [2, 4]], dtype=float)
  b = np.array([3, 5], dtype=float)
  x = solve_linear_system(A, b, verbose=False)
  assert x is None, "Inconsistent system should return None"
  print(" ✓ solve: inconsistent system returns None")

def test_solve_infinite_satisfies_equations():
  """Infinite system: minimum-norm solution must satisfy A@x = b."""
  A = np.array([[1, 2, 1], [2, 4, 3]], dtype=float)
  b = np.array([5, 11], dtype=float)
  x = solve_linear_system(A, b, verbose=False)
  assert x is not None, "Should return minimum-norm solution"
  assert np.allclose(A @ x, b, atol=1e-10), (
    f"Minimum-norm solution does not satisfy equations"
  )
  print(" ✓ solve: infinite system — minimum-norm solution satisfies A@x = b")

def test_condition_number_identity():
  """Identity matrix: condition number must be 1.0."""
  I = np.eye(4)
  cond = compute_condition_number(I)
  assert np.isclose(cond, 1.0), (
    f"Identity condition number should be 1.0, got {cond}"
  )
  print(" ✓ condition: κ(I) = 1.0")

def test_condition_number_ill_conditioned():
  """Near-singular matrix: condition number must be very large."""
  A = np.array([[1.0, 1.0 ], [1.0, 1.0 + 1e-5]], dtype=float)
  cond = compute_condition_number(A)
  assert cond > 1e4, (
    f"Near-singular matrix should have κ > 1e4, got {cond:.2e}"
  )
  print(f" ✓ condition: near-singular matrix κ = {cond:.2e} > 1e4")

if __name__ == "__main__":
  print("\n" + "="*60)
  print(" LINEAR SOLVER TEST SUITE")
  print("="*60 + "\n")
  
  test_rank_full_rank()
  test_rank_rank_deficient()
  test_rank_zero_matrix()
  test_rank_tall_matrix()
  test_analyze_unique()
  test_analyze_inconsistent()
  test_analyze_infinite()
  test_solve_unique_and_verify()
  test_solve_inconsistent_returns_none()
  test_solve_infinite_satisfies_equations()
  test_condition_number_identity()
  test_condition_number_ill_conditioned()
  
  print("\n" + "="*60)
  print(" ALL 12 TESTS PASSED")
  print("="*60 + "\n")
