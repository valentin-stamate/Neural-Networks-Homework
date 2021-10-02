from matrix import Matrix


class SystemSolverRaw:
    @staticmethod
    def solve_system(matrix, coeff, terms):
        det = Matrix.determinant(matrix)

        if det == 0:
            print("The determinant of the system is 0.")
            return None

        inverse = Matrix.inverse(matrix)
        result = Matrix.multiply(inverse, coeff)

        final_result = {}
        for i, term in enumerate(terms):
            final_result[term] = result[i][0]

        return final_result
