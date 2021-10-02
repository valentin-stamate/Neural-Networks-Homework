import numpy as np

from equation_system_parser import EquationSystemParser
from equation_system_solver import SystemSolver
from equation_system_solver_raw import SystemSolverRaw
from matrix import Matrix


def main():
    parser = EquationSystemParser('./input')

    A, B, terms = parser.parse()

    print("Matrx A:")
    Matrix.print(A)
    print()

    print("Matrix B:")
    Matrix.print(B)
    print()

    print("Raw calculated solution  ", SystemSolverRaw.solve_system(A, B, terms))
    print("Numpy calculated solution", SystemSolver.solve_system(A, B, terms))


if __name__ == '__main__':
    main()
