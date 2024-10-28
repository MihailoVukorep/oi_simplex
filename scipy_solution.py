#!/usr/bin/env python

from scipy.optimize import linprog

# Define the coefficients of the objective function
# Maximize Z = 3x1 + 2x2  --> linprog() minimizes, so use -1 * coefficients
c = [-3, -2]

# Define the inequality constraints matrix
# Example: 
#   2x1 + x2 <= 20
#   4x1 + 3x2 <= 42
#   2x1 + 5x2 <= 30
A = [
    [2, 1],
    [4, 3],
    [2, 5]
]

# Define the right-hand side of the inequality constraints
b = [20, 42, 30]

# Define the bounds for each variable (x1 >= 0, x2 >= 0)
x_bounds = (0, None)  # (Lower bound, Upper bound) for both variables

# Solve the linear programming problem
result = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, x_bounds], method='simplex')

# Print the results
if result.success:
    print("Optimal solution found:")
    print(f"Objective function value: {round(-result.fun, 2)}")
    print(f"Values of decision variables: {result.x}")
else:
    print("No feasible solution found.")

