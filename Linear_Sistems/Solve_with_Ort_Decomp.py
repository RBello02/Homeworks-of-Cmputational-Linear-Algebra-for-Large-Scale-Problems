import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Factorizations.Gram_Schmidt_Decomposition import Gramschmidt_Decomposition
from Factorizations.Householder_Decomposition import Householder_Decomposition
from Factorizations.Givens_Decomposition import Givens_Decomposition

def Solve_with_Ort_Decomp(b,A=None, method='GS', tol = 1e-8):


    """ 
    This function solves a linear system Ax=b using an orthogonal decomposition A=QR
    The method parameter allows to choose between 'GS' (Gram-Schmidt), 'HH' (Householder) and 'Givens'
    """

    if A is None:
        raise ValueError("A is required.")

    if method == 'GS':
        Q, R, Qqual, QRqual = Gramschmidt_Decomposition(A)
    elif method == 'HH':
        Q, R, Qqual, QRqual = Householder_Decomposition(A)
    elif method == 'Givens':
        Q, R, Qqual, QRqual = Givens_Decomposition(A)
    else:
        raise ValueError("Unknown method. Select between 'GS', 'HH' o 'Givens'.")

    # Compute Q.T @ b
    if method in ["GS","Givens", "HH"]:
        Q_b = Q.T @ b

    # Backward substitution to solve Rx = Q.T @ b
    x = np.zeros_like(b, dtype=np.float64)
    n = len(b)
    for i in range(n-1, -1, -1):
        if abs(R[i,i]) < tol:
            raise ValueError("Matrix is singular to tolerance.")
        x[i] = (Q_b[i] - R[i, i+1:] @ x[i+1:]) / R[i,i]

    return x

def main_test():
    print("")
    print("")
    A = np.random.rand(4,4)
    b = np.random.rand(4)
    x = Solve_with_Ort_Decomp(b,A=A, method='GS')
    x_exact = np.linalg.solve(A,b)
    print("*************************************")
    print("Testing Solve_with_Ort_Decomp with Gram-Schmidt")
    print("*************************************")
    print("")
    print("Matrix 4x4 and vector b of size 4")
    print("Difference ||x - x_exact|| =",np.linalg.norm(x - x_exact))

    A = np.random.rand(30,30)
    b = np.random.rand(30)
    x = Solve_with_Ort_Decomp(b,A=A, method='GS')
    x_exact = np.linalg.solve(A,b)
    print("")
    print("Matrix 30x30 and vector b of size 30")
    print("Difference ||x - x_exact|| =",np.linalg.norm(x - x_exact))


    A = np.random.rand(130,130)
    b = np.random.rand(130)
    x = Solve_with_Ort_Decomp(b,A=A, method='GS')
    x_exact = np.linalg.solve(A,b)
    print("")
    print("Matrix 130x130 and vector b of size 130")
    print("Difference ||x - x_exact|| =",np.linalg.norm(x - x_exact))



    print("")
    print("")
    A = np.random.rand(4,4)
    b = np.random.rand(4)
    x = Solve_with_Ort_Decomp(b,A=A, method='Givens')
    x_exact = np.linalg.solve(A,b)
    print("*************************************")
    print("Testing Solve_with_Ort_Decomp with Givens")
    print("*************************************")
    print("")
    print("Matrix 4x4 and vector b of size 4")
    print("Difference ||x - x_exact|| =",np.linalg.norm(x - x_exact))

    A = np.random.rand(30,30)
    b = np.random.rand(30)
    x = Solve_with_Ort_Decomp(b,A=A, method='Givens')
    x_exact = np.linalg.solve(A,b)
    print("")
    print("Matrix 30x30 and vector b of size 30")
    print("Difference ||x - x_exact|| =",np.linalg.norm(x - x_exact))


    A = np.random.rand(130,130)
    b = np.random.rand(130)
    x = Solve_with_Ort_Decomp(b,A=A, method='Givens')
    x_exact = np.linalg.solve(A,b)
    print("")
    print("Matrix 130x130 and vector b of size 130")
    print("Difference ||x - x_exact|| =",np.linalg.norm(x - x_exact))

    print("")
    print("")
    A = np.random.rand(4,4)
    b = np.random.rand(4)
    x = Solve_with_Ort_Decomp(b,A=A, method='HH')
    x_exact = np.linalg.solve(A,b)
    print("*************************************")
    print("Testing Solve_with_Ort_Decomp with Householder")
    print("*************************************")
    print("")
    print("Matrix 4x4 and vector b of size 4")
    print("Difference ||x - x_exact|| =",np.linalg.norm(x - x_exact))

    A = np.random.rand(30,30)
    b = np.random.rand(30)
    x = Solve_with_Ort_Decomp(b,A=A, method='HH')
    x_exact = np.linalg.solve(A,b)
    print("")
    print("Matrix 30x30 and vector b of size 30")
    print("Difference ||x - x_exact|| =",np.linalg.norm(x - x_exact))


    A = np.random.rand(130,130)
    b = np.random.rand(130)
    x = Solve_with_Ort_Decomp(b,A=A, method='HH')
    x_exact = np.linalg.solve(A,b)
    print("")
    print("Matrix 130x130 and vector b of size 130")
    print("Difference ||x - x_exact|| =",np.linalg.norm(x - x_exact))

if __name__ == "__main__":
    main_test()