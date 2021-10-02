
class Matrix:
    # raw calculus
    @staticmethod
    def multiply(a, b):
        n_a = len(a)
        m_a = len(a[0])
        m_b = len(b[0])

        result = Matrix.create_matrix(n_a, m_b)

        for i in range(n_a):
            for j in range(m_b):
                result[i][j] = 0
                for k in range(m_a):
                    result[i][j] += (a[i][k] * b[k][j])

        return result

    # numpy calculus
    @staticmethod
    def determinant(matrix):
        if len(matrix) != len(matrix[0]):
            raise Exception("The determinant require a square matrix.")

        if len(matrix) == 1:
            return matrix[0][0]

        det = 0
        for i in range(len(matrix)):
            det += ((-1) ** (1 + i)) * matrix[1][i] * Matrix.determinant(Matrix.cut(matrix, 1, i))

        return det

    @staticmethod
    def adjunct_matrix(matrix):
        n = len(matrix)

        if len(matrix) != len(matrix[0]):
            raise Exception("Adjunct matrix require squared matrix.")

        adjunct = Matrix.create_matrix(n, n)

        matrix = Matrix.transpose(matrix)
        for i in range(n):
            for j in range(n):
                adjunct[i][j] = ((-1) ** (i + j)) * Matrix.determinant(Matrix.cut(matrix, i, j))

        return adjunct

    @staticmethod
    def inverse(matrix):
        det = Matrix.determinant(matrix)

        if det == 0:
            print("Inverse could not be calculated due to null determinant")
            return None

        adjunct = Matrix.adjunct_matrix(matrix)
        result = Matrix.map(adjunct, lambda x: x * 1 / det)

        return result

    @staticmethod
    def map(matrix, fun):
        n = len(matrix)
        m = len(matrix[0])

        result = Matrix.create_matrix(n, m)

        for i in range(n):
            for j in range(m):
                result[i][j] = fun(matrix[i][j])

        return result

    # helper methods
    @staticmethod
    def cut(matrix, line, column):
        return Matrix.remove_column(Matrix.remove_line(matrix, line), column)

    @staticmethod
    def remove_line(matrix, line):
        if line < 0 or line >= len(matrix):
            raise Exception("Matrix index exceeded")

        return matrix[0:line] + matrix[(line + 1):]

    @staticmethod
    def remove_column(matrix, column):
        transpose = Matrix.transpose(matrix)
        transpose = Matrix.remove_line(transpose, column)
        return Matrix.transpose(transpose)

    @staticmethod
    def transpose(matrix):
        n = len(matrix)
        m = len(matrix[0])

        result = Matrix.create_matrix(m, n)

        for i in range(n):
            for j in range(m):
                result[j][i] = matrix[i][j]

        return result

    @staticmethod
    def create_matrix(n, m, default_value=0):
        return [[default_value for i in range(0, m)] for x in range(n)]

    @staticmethod
    def print(matrix):
        for line in matrix:
            for e in line:
                print(e, end=" ")
            print()
        return ''

