#!/usr/bin/env python

import numpy as np

def simplex(c, A, b, isMin=False):
    """
    Solve LP using simplex method.

    Max/Min c^T x
    subject to Ax <= b, x >= 0

    Parameters:
        c : objective coefficients
        A : constraint matrix
        b : RHS
        problem : "max" or "min"
    """

    num_constraints, num_variables = A.shape

    # Convert minimization to maximization
    if isMin:
        c = -c

    # Build tableau
    tableau = np.zeros((num_constraints + 1, num_variables + num_constraints + 1))

    # Constraints
    tableau[:-1, :num_variables] = A
    tableau[:-1, num_variables:num_variables + num_constraints] = np.eye(num_constraints)
    tableau[:-1, -1] = b

    # Objective row (IMPORTANT: negative for max)
    tableau[-1, :num_variables] = -c

    print("tableau:")
    print(tableau)

    while True:
        # Check optimality
        if np.all(tableau[-1, :-1] >= 0):
            break

        # Pivot column (most negative value)
        pivot_col = np.argmin(tableau[-1, :-1])
        print(f"pivot_col: {pivot_col} <- most negative in Z")

        # Check unboundedness
        column = tableau[:-1, pivot_col]
        if np.all(column <= 0):
            raise ValueError("Problem is unbounded.")

        print("column:")
        print(column)

        # Ratio test
        ratios = np.where(column > 0, tableau[:-1, -1] / column, np.inf)
        print(f"ratios: {ratios}")
        pivot_row = np.argmin(ratios)
        print(f"pivot_row: {pivot_row}")

        # Normalize pivot row
        print()
        print(tableau)
        pivot = tableau[pivot_row, pivot_col]
        tableau[pivot_row] /= pivot
        print(f"normalized pivot row with: {pivot}")
        print(tableau)
        print()

        # Eliminate other rows
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

        print("eliminated other rows")
        print(tableau)
        print()

    # Extract solution
    solution = np.zeros(num_variables)

    for j in range(num_variables):
        col = tableau[:, j]
        if np.count_nonzero(col[:-1]) == 1 and np.isclose(col[:-1].max(), 1):
            row = np.argmax(col[:-1])
            solution[j] = tableau[row, -1]

    # Objective value
    objective_value = tableau[-1, -1]

    # If minimization, convert back
    if isMin:
        objective_value = -objective_value

    return solution, objective_value



c = np.array([6, 14, 13])
A = np.array([
    [0.5, 2, 1],
    [1, 2, 4],
])
b = np.array([24, 60])
solution, value = simplex(c, A, b)
print("Solution:", solution)
print("Max value:", value)

