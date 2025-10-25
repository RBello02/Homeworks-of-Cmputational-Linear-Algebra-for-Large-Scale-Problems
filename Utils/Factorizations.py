import numpy as np

def plu_decomposition(A, tol = 1e-8):


    """
    This function performs the PA = LU decomposition. Where P is a permutation matrix, L is lower triag and U is upper triag.
    tol is the tollerance 
    """

    A = A.copy()
    m,n = A.shape
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

    return P,L,U

def main_test():

    A = np.array([[0,2,1],[1,1,0],[2,1,1]],dtype=float)
    P,L,U = plu_decomposition(A)
    print("A =\n",A)
    print("P =\n",P)
    print("L =\n",L)
    print("U =\n",U)
    print("PA =\n",np.dot(P,A))
    print("LU =\n",np.dot(L,U))
    print("Difference ||PA - LU|| =\n",np.linalg.norm(P@A - L@U))

if __name__ == "__main__":
    main_test()


