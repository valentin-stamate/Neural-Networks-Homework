from matrix import Matrix

def main():
    A = [[1, 3, 5, 9],
        [1, 3, 1, 7],
        [4, 3, 9, 7],
         [5, 2, 0, 9]]

    B = [[2, 5],
         [6, 7],
         [1, 8]]

    # Matrix.print(Matrix.multiply(A, B))
    Matrix.print(Matrix.multiply(Matrix.inverse(A), A))


if __name__ == '__main__':
    main()
