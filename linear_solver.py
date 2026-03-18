"""
linear_solver.py
Robust linear system solver with comprehensive edge case handling.

Handles singular matrices, rank-deficient systems, and numerical
instability. Classifies any system Ax = b as unique, infinite, or inconsistent, and solves accordingley.

Mathematical foundation:
- Rouché-Capelli theorem for system classification
- SVD-based rank computation for numerical stability
- Condition number analysis for reliability assessment

Author: Luke Wardle
Date: 18/03/2026

"""

import numpy as np
from typing import Tuple, Optional, Dict, Any

def compute_rank(A: np.ndarray, tolerance: float = 1e-10) -> int:
  """
  Compute the rank of a matrix using Singlular Value Decomposition.

  SVD is preferred over row reduction for numerical stability.
  Singular values below 'tolerance' are treated as zero, giving a 
  reliable rank estimate even for near-singular matrices.

  Args:
    A:          Input matric (m x n), any shape
    tolerance:  Threshold for treating singular values as zero. 
                Default 1e-10 works for most practical data.

  Returns:
    Rank of the matrix (integer between 0 and min(m, n))


  """
  # compute_uv=False: only compute singular values, not U and V
  # This is faster when we only need the rank
  singular_values = np.linarg.svd(A, compute_uv=False)

  # Count how many singular values exceed our zero threshold
  rank = int(np.sum(singular_values > tolerance))
  return rank

def compute_condition_number(A: np.ndarray) -> float:
  """
  Compute the condition number of a matrix.

  The condition number κ(A) = σ_max / σ_min measures how sensitive
  the solution of Ax = b is to small errors in A or b.

  Interpretation: 
      κ(A) ≈ 1        : well-conditioned, results are reliable 
      κ(A) ≈ 10^6     : moderate ill-conditioning, use with caution 
      κ(A) > 10^10    : severely ill-conditioned, results unreliable 
      κ(A) = inf      : A is singular 

  Args:
    A: Matrix (any shape, but meaningful for square matrices)

  Returns:
    Condition number (float, 1.0 to infinity)
  
  """
  return float(np.linalg.cond(A))

def analyze_system(A: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
  """
  Analyse a linear system Ax = b and determine its properties.

  Applies the Rouché-Capelli theorem to classify the system as unique, infinite, or inconsistent. Also computes condition number
  for numerical stability assessment.

  Args:
    A: Coefficient matrix (m x n)
    b: Right-hand side vector (m,) - must be 1D

  Returns:
    Dictionary containing:
    m, m             : matrix dimensions
    rank_A           : rank of coefficient matrix
    rank_Ab          : rank of augmented matrix [A|b]
    system_type      : 'unique', 'infinite', or 'inconsistent'
    condition_number : κ(A) if square, else None
  
  """
  m, n = A.shape
  rank_A = compute_rank(A)

  # Build augmented matrix [A|b] - append b as extra column
  # b must be reshaped to column vector for hstack
  Ab = np.column_stack([A, b.reshape(-1, 1)])
  rank_Ab = compute_rank(Ab)

  # Apply Rouché-Capelli theorem
  if rank_A < rank_Ab:
    # b has a component outside Col(A) - no solution
    system_type = "inconsistent"
  elif rank_A == n:
    # Full column rank - no free variables - unique solution
    system_type = "unique"
  else:
    # b in Col(A) but rank < n - free variables - infinite solutions
    system_type = 'infinite'

  # Condition number only meaningful for square matrices
  cond_num = compute_condition_number(A) if m == n else None

  return {
    "m": m,
    "n": n,
    "rank_A": rank_A,
    "rank_Ab": rank_Ab,
    "system_type": system_type,
    "condition_number": cond_num,
  }

def solve_linear_system(
    A: np.ndarray,
    b: np.ndarray,
    verbose: bool = True
) -> Optional[np.ndarray]:
  """
  Solve linear system Ax = b with comprehensive edge case handling.

  Automatically detects system type using Rouché-Capelli theorem and applies the appropriate solution method.

  Args:
    A: Coefficient matrix (m x n)
    b: Right-hand side vector (m,) or (m, l)
    verbose: If True, print full diagnostic report

  Returns:
    Solution vector x as 1D array, or None if no solution exists
  
  """
  # Normalise b to 1D
  b_flat = b.flatten()

  # Validate dimensions
  if A.shape[0] != b_flat.shape[0]:
    raise ValueError(
      f"Dimension mismatch: A has {A.shape[0]} rows",
      f"but b has {b_flat.shape[0]} elements."
    )
  
  analysis = analyze_system(A, b_flat)

  if verbose:
    print(f"\n{'='*60}")
    print(" SYSTEM ANALYSIS")
    print(f"{'='*60}")
    print(f" Dimensions : {analysis['m']} equations, {analysis['n']} unknowns")
    print(f" rank(A) : {analysis['rank_A']}")
    print(f" rank([A|b]) : {analysis['rank_Ab']}")
    print(f" System type : {analysis['system_type'].upper()}")
    if analysis['condition_number'] is not None:
      cond = analysis['condition_number']
      print(f" Condition κ :  {cond:.3e}", end="")
      if cond > 1e10:
        print(" ⚠  SEVERELY ILL-CONDITIONED", end="")
      elif cond > 1e6:
        print(" ⚠  MODERATELY ILL-CONDITIONED", end="")
      else:
        print(" ✓  Well-conditioned", end="")
      print()
    print(f"{'='*60}")

  # ── CASE 1: No solution ──────────────────────────────────────── 
  if analysis['system_type'] == 'inconsistent':
    if verbose:
      print(" ❌  No solution exists (inconsistent system).")
      print(" b is not in the column space of A.")
    return None
  
  # ── CASE 2: Unique solution ──────────────────────────────────── 
  elif analysis['system_type'] == 'unique':
    try:
      x = np.linalg.solve(A, b_flat)
      if verbose:
        print("  ✓  Unique solution found.")
      return x
    except np.linalg.LinAlgError:
      # Should not reach here if analysis is correct,
      # but handle defensively.
      if verbose:
        print(" ❌  Solver failed (unexpected singular matrix).")
      return None
  
  # ── CASE 3: Infinite solutions ───────────────────────────────── 
  else:
    # lstsq returns the minimum-norm solution - the solution
    # closest to the origin, from all infinitely many solutions.
    x, residuals, rank, s = np.linalg.lstsq(A, b_flat, rcond=None)
    if verbose:
      print(" ⚠  Infinite solutions exist.")
      print("      Returning minimum-norm solution (closest to origin).")
    return x