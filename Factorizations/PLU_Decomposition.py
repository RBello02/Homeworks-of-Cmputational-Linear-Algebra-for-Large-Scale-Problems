import numpy as np

def PLU_Decomposition(A, tol = 1e-8):


    """
    This function performs the PA = LU decomposition for rectangular matrices. Where P is a permutation matrix, L is lower triag and U is upper triag.
    tol is the tollerance 
    """

    A = A.copy()
    m,n = A.shape

    if m != n:
        raise ValueError("Number of rows must be greater than or equal to number of columns")
    
    P = np.eye(m)
    L = np.zeros((m,n))
    U = A.copy()

    for k in range(min(m,n)):
        # pivot
        i_max = np.argmax(np.abs(U[k:,k])) + k
        if abs(U[i_max,k]) < tol:
            continue

        # swap rows in U e P and L
        U[[k,i_max],:] = U[[i_max,k],:]
        P[[k,i_max],:] = P[[i_max,k],:]
        L[[k,i_max],:k] = L[[i_max,k],:k]  

        # elimination
        for i in range(k+1, m):
            L[i,k] = U[i,k] / U[k,k]
            U[i,k:] -= L[i,k] * U[k,k:]
    
    np.fill_diagonal(L,1)
    norm_result = np.linalg.norm(P @ A - L @ U)

    return P,L,U, norm_result

def main_test():

    A = np.array([[0,2,1],[1,1,0],[2,1,1]],dtype=float)
    P,L,U,norm = PLU_Decomposition(A)
    print("A =\n",A)
    print("P =\n",P)
    print("L =\n",L)
    print("U =\n",U)
    print("Difference ||PA - LU|| =\n",norm)

    A = np.array([[0,2,1],[1,1,0],[2,1,1],[0,1,4]],dtype=float)
    P,L,U,norm = PLU_Decomposition(A)
    print("A =\n",A)
    print("P =\n",P)
    print("L =\n",L)
    print("U =\n",U)
    print("Difference ||PA - LU|| =\n",norm)


if __name__ == "__main__":
    main_test()


