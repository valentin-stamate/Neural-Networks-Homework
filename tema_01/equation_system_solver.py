import numpy
import numpy as np


class SystemSolver:
    @staticmethod
    def solve_system(matrix, coeff, terms):
        matrix = np.array(matrix).astype(numpy.float32)
        coeff = np.array(coeff).astype(numpy.float32)

        det = np.linalg.det(matrix)

        if det == 0:
            print("The determinant of the system is 0.")
            return None

        tramspose = matrix.transpose()
        adjoint = SystemSolver.get_adjoint(tramspose)

        inverse = adjoint / det
        result = inverse.dot(coeff)

        final_result = {}
        for i, term in enumerate(terms):
            final_result[term] = result[i, 0]

        return final_result

    @staticmethod
    def get_adjoint(transpose):
        result = np.full_like(transpose, 0)

        for i in range(transpose.shape[0]):
            for j in range(transpose.shape[1]):
                sub_matrix = np.delete(np.delete(transpose, i, axis=0), j, axis=1)
                result[i, j] = ((-1) ** (i + j)) * np.linalg.det(sub_matrix)

        return result
