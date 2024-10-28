#!/usr/bin/env python

import numpy as np

def simplex(c, A, b):
    # 1. table setup
    num_constraints, num_variables = A.shape
    print("A.shape", A.shape)
    tableau = np.zeros((num_constraints + 1, num_variables + num_constraints + 1))
    tableau[:-1, :num_variables] = A
    tableau[:-1, num_variables:num_variables + num_constraints] = np.eye(num_constraints)
    tableau[:-1, -1] = b
    tableau[-1, :num_variables] = c

    print(tableau)

    # 2. pivot process
    while True:
        # 3. check if solution optimal
        if np.all(tableau[-1, :-1] >= 0):
            print("Optimal solution found!")
            break
        else:
            print("not optimal yet")

        # 4. find pivot column (most negative coefficient in the objective row)
        pivot_col = np.argmin(tableau[-1, :-1])
        print("pivot_col:", pivot_col)

        # 5. check for unbounded solution
        if np.all(tableau[:-1, pivot_col] <= 0):
            raise ValueError("Problem is unbounded.")

        # 6. find the pivot row (minimum ratio test)
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col] # RHS/key clm
        ratios[ratios <= 0] = np.inf  # ignore non-positive ratios
        print("ratios:",ratios)
        pivot_row = np.argmin(ratios) # get smallest positive ratio
        print("pivot_row:",pivot_row)

        print("key:", tableau[pivot_row, pivot_col])

        # 7. normalize the pivot row
        tableau[pivot_row] /= tableau[pivot_row, pivot_col] # / entire row with key element
        
        print("after norm:", tableau[pivot_row])

        # 8. update the other rows
        for i in range(len(tableau)):
            if i != pivot_row:
                print(tableau[i], "-=", tableau[i, pivot_col], "*", tableau[pivot_row], "=", tableau[i, pivot_col] * tableau[pivot_row])
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

        print("\nTableau after pivoting:")
        print(tableau)

    # 9. extract the solution
    solution = np.zeros(num_variables)
    for i in range(num_constraints):
        # If the row corresponds to a basic variable, extract its value
        if np.argmax(tableau[i, :num_variables]) < num_variables:
            solution[np.argmax(tableau[i, :num_variables])] = tableau[i, -1]

    objective_value = tableau[-1, -1]
    return solution, objective_value

# Z = 1x1 + 4x2
c = np.array([-1, -4])

A = np.array([
    [2, 1],
    [3, 5],
    [1, 3]
])

# RHS
b = np.array([3, 9, 5])

# Solve the problem using the simplex method
solution, objective_value = simplex(c, A, b)

print("\nOptimal Solution:")
print("Values of decision variables:", solution)
print("Maximum value of objective function:", objective_value)
