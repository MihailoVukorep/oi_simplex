#!/usr/bin/env python

import numpy as np

# Define the coefficients of the objective function (maximize Z = 3x1 + 2x2)
c = np.array([3, 2])

# Define the constraint coefficients
A = np.array([
    [2, 1],
    [4, 3],
    [2, 5]
])

# Define the right-hand side of the constraints
b = np.array([20, 42, 30])

def simplex(c, A, b):
    """
    Solve the linear programming problem:
    Maximize: Z = c @ x
    Subject to: A @ x <= b, x >= 0
    """
    # Step 1: Set up the initial tableau
    num_constraints, num_variables = A.shape
    tableau = np.zeros((num_constraints + 1, num_variables + num_constraints + 1))

    # Fill the tableau with the constraint coefficients
    tableau[:-1, :num_variables] = A
    tableau[:-1, num_variables:num_variables + num_constraints] = np.eye(num_constraints)
    tableau[:-1, -1] = b

    # Fill the objective function row (negated for maximization)
    tableau[-1, :num_variables] = -c

    print("Initial Tableau:")
    print(tableau)

    # Step 2: Perform the pivoting process until an optimal solution is found
    while True:
        # Step 3: Check if the current solution is optimal (no positive coefficients in the objective row)
        if np.all(tableau[-1, :-1] <= 0):
            print("Optimal solution found!")
            break

        # Step 4: Find the pivot column (most positive coefficient in the objective row)
        pivot_col = np.argmax(tableau[-1, :-1])

        # Step 5: Check for unbounded solution
        if np.all(tableau[:-1, pivot_col] <= 0):
            raise ValueError("Problem is unbounded.")

        # Step 6: Find the pivot row (minimum ratio test)
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf  # Ignore non-positive ratios
        pivot_row = np.argmin(ratios)

        # Step 7: Perform the pivot operation (normalize the pivot row)
        tableau[pivot_row] /= tableau[pivot_row, pivot_col]

        # Step 8: Update the other rows
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

        print("\nTableau after pivoting:")
        print(tableau)

    # Step 9: Extract the solution
    solution = np.zeros(num_variables)
    for i in range(num_constraints):
        # If the row corresponds to a basic variable, extract its value
        if np.argmax(tableau[i, :num_variables]) < num_variables:
            solution[np.argmax(tableau[i, :num_variables])] = tableau[i, -1]

    objective_value = tableau[-1, -1]
    return solution, objective_value



# Solve the problem using the simplex method
solution, objective_value = simplex(c, A, b)

print("\nOptimal Solution:")
print("Values of decision variables:", solution)
print("Maximum value of objective function:", objective_value)
