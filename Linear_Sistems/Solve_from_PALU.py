import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Factorizations.PLU_Decomposition import PLU_Decomposition

def Solve_from_PALU(b,A=None,PLU=None, tol = 1e-8):


    """ 
    This function solves a linear system Ax=b using the PA=LU decomposition
    """

    if PLU is None:
        if A is None:
            raise ValueError("Devi fornire almeno A o PLU")
        P, L, U, _ = PLU_Decomposition(A, tol=tol)
    else:
        P, L, U = PLU

    Pb = P @ b   # we have to permute the vector b

    # Forward substitution to solve Ly = Pb
    y = np.zeros_like(b, dtype=np.float64)
    for i in range(len(b)):
        y[i] = Pb[i] - L[i, :i] @ y[:i]       # do not divide by L[i,i] because L has 1s on the diag

    # Backward substitution to solve Ux = y
    x = np.zeros_like(b, dtype=np.float64)
    for i in range(len(b)-1, -1, -1):
        if abs(U[i,i]) < tol:
            raise ValueError("Matrix is singular to tolerance.")
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i,i]

    return x   


def main_test():

    print("Testing Solve_from_PALU with predefined matrix A and vector b")
    print("-------------------------------------------------------------- ")
    print("Test 1: 3x3 system")
    A = np.array([[0,2,1],[1,1,0],[2,1,1]],dtype=float)
    b = np.array([1,2,3],dtype=float)
    x = Solve_from_PALU(b,A=A)
    print("PALU Solution x =",x)
    x_exact = np.linalg.solve(A,b)
    print("Exact Solution x =",x_exact)
    print("Difference ||x - x_exact|| =",np.linalg.norm(x - x_exact))
    print("-------------------------------------------------------------- ")
    print("Test 2: 100x100 system")
    A = np.random.rand(100,100)
    b = np.random.rand(100)
    x = Solve_from_PALU(b,A=A)
    x_exact = np.linalg.solve(A,b)
    print("Difference ||x - x_exact|| =",np.linalg.norm(x - x_exact))
    print("-------------------------------------------------------------- ")
    print("Test 3: 1000x1000 system")
    A = np.random.rand(1000,1000)
    b = np.random.rand(1000)
    x = Solve_from_PALU(b,A=A)
    x_exact = np.linalg.solve(A,b)
    print("Difference ||x - x_exact|| =",np.linalg.norm(x - x_exact))

    

if __name__ == "__main__":
    main_test()