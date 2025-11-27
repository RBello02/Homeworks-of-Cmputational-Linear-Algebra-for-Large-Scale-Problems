import numpy as np

def transform(A, tol = 1e-8):
    # for visualization 
    A[abs(A)<tol] = 0.0
    return A

def householder_mat(x):
    """
    Function that compute the reflection matrix of the Householder method for a given vector x.
    :param x: vector that can be given both as 1D-array object and 2D-array column/row object (numpy ndarray);
    :return Px: Householder reflection matrix as 2D-array object (numpy ndarray).
    """
    
    # Reshaping the input as a column vector (if it is 1D-array or row 2D-array)... actually it works even if it is a non-vectot matrix...
    v = x.reshape(x.size, 1)

    sigma = -np.sign(x[0]) * np.linalg.norm(x)

    # Computation of u (versor)
    u = v + np.eye(v.size)[:,[0]]*sigma
    u = u/np.linalg.norm(u)

    # Computation of the reflection matrix
    Px = np.eye(v.size) - 2*u*u.T

    return Px

def left_householder_mat(A,k, tol = 1e-8):
    """
    This function is similar to householder_mat but it returns only the product HA and the vector u, this is optimize because we don't need to save H
    but only u and when we need it is possible to compute HA 

    From the theory we know that HA = A -  u*(u^t*A)     

    """
    A = A.copy()
    x = A[k:,k].reshape(-1,1)

    if x.size == 0:
        return None
    if np.linalg.norm(x) < tol or x.size == 1:
        return np.eye(1)
    
    sigma = -np.copysign(np.linalg.norm(x), x[0,0])

    u = x.copy()
    u[0,0] += sigma
    u = u/np.linalg.norm(u)

    Plx  = np.eye(u.size)- 2 * u @ (u.T)

    return Plx

def right_householder_mat(A, k, tol = 1e-8):
    """
    Apply Householder from the right to zero elements after the diagonal in row k.
    Returns updated A and the Householder vector u (compact form).
    """
    A = A.copy()
    x = A[k, k+1:].reshape(-1,1)

    if x.size == 0 or np.linalg.norm(x) < tol:
        return None
    if x.size == 1:
        return np.eye(1)
    
    sigma = -np.copysign(np.linalg.norm(x), x[0,0])

    u = x.copy()
    u[0,0] += sigma
    u = u / np.linalg.norm(u)
    
    # Apply Householder on the submatrix (columns k+1:n)
    Prx = np.eye(u.size) - 2 * (u) @ u.T
    
    return Prx
    

def Householder_Decomposition(X):
    """
    Function that performs the Householder method for a given square matrix X.
    :param X: square matrix represented as 2D-array object (numpy ndarray);
    :return Q: 2D-array (orthogonal matrix) 
    :return R: 2D-array (upper triangle).
    :return Qqual: norm ||In - Q @ Q.T||
    :return QRqual: norm ||Q @ X - R||
    """
    
    m, n = X.shape
    
    if m != n:
        print('MATRIX IS NOT SQUARE!')
        return None, None, None, None
    
    # Initialization of the matrices
    R = X.copy()
    Q = np.eye(n)

    for j in range(n-1):
        r = R[j:,j].reshape(-1, 1)
        H_small = householder_mat(r)
        H = np.eye(m)
        #print("*************")
        #print(H_small)
        H[j:, j:] = H_small
        R = H@R
        #print(H)
        #print(R)
        #print("*************")

        Q = Q@H.T
    
    Qqual = np.linalg.norm(np.eye(n)-Q@Q.T)    
    QRqual = np.linalg.norm(Q.T@X-R)

    return Q, R, Qqual, QRqual


def Householder_Bidiag_Decomposition(X):

    """
    This function is mostly used in SVD naive decompositions, The idea is to "left" apply householder decomposition to have a column with zeros under the diag 
    and also apply "right" householder decomposition for having a The element of the row equal to zero after the diag. It will generate a bidiagonal matrix. 
    We must consider, in this case, X also not squared.
    """

    m, n = X.shape

    Ul = np.eye(m)
    Ur = np.eye(n)

    for k in range(min(n,m)): #iterate over the columns

        # step 1 compute the householder dec for the columns

        HlSmall = left_householder_mat(X,k)  

        if HlSmall is not None:
            Hl = np.eye(m)
            Hl[k:,k:] = HlSmall

            X = Hl@X
            Ul = Ul@Hl


        # --- Step 2: Householder on right 

        HrSmall = right_householder_mat(X, k)

        if HrSmall is not None:
            Hr = np.eye(n)
            Hr[k+1:, k+1:] = HrSmall

            X = X @ Hr        
            Ur = Ur @ Hr 

    return X, Ul, Ur
    


def main_test():

    A = np.random.rand(4,4)
    Q, R, Qqual, QRqual = Householder_Decomposition(A)
    print("***************** TESTS HOUSEHOLDER NAIVE *********************")
    print("*************************************")
    print("Test random 4x4 matrix")
    print("||I - Q @ Q.T|| =", Qqual)
    print("||Q @ R - A|| =", QRqual)
    print("Q", Q)
    print("R", R)
    print("*************************************")

    A = np.random.rand(50,50)
    Q, R, Qqual, QRqual = Householder_Decomposition(A)
    print("*************************************")
    print("Test random 50x50 matrix")
    print("||I - Q @ Q.T|| =", Qqual)
    print("||Q @ R - A|| =", QRqual)
    print("*************************************")

    A = np.random.rand(200,200)
    Q, R, Qqual, QRqual = Householder_Decomposition(A)
    print("*************************************")
    print("Test random 200x200 matrix")
    print("||I - Q @ Q.T|| =", Qqual)
    print("||Q @ R - A|| =", QRqual)
    print("*************************************")

    print("\n")
    print("\n")
    print("***************** TESTS HOUSEHOLDER BIDIAG *********************")
    A = np.random.rand(5,4)
    A_n,U_left, U_right = Householder_Bidiag_Decomposition(A)
    print("*************************************")
    print("Test random 5x4 matrix")
    #print("A", transform(A_n))
    #print("U left", U_left)
    #print("U right", U_right)
    print("norm of the difference ||A-UBV||:", np.linalg.norm(A-U_left@A_n@U_right.T))
    print("*************************************")
    A = np.random.rand(4,5)
    A_n,U_left, U_right = Householder_Bidiag_Decomposition(A)
    print("*************************************")
    print("Test random 4x5 matrix")
    #print("A", transform(A_n))
    #print("U left", U_left)
    #print("U right", U_right)
    print("norm of the difference ||A-UBV||:", np.linalg.norm(A-U_left@A_n@U_right.T))
    print("*************************************")
    A = np.random.rand(100,50)
    A_n,U_left, U_right = Householder_Bidiag_Decomposition(A)
    print("*************************************")
    print("Test random 100x50 matrix")
    #print("A", transform(A_n))
    #print("U left", U_left)
    #print("U right", U_right)
    print("norm of the difference ||A-UBV||:", np.linalg.norm(A-U_left@A_n@U_right.T))
    print("*************************************")
    A = np.random.rand(100,500)
    A_n,U_left, U_right = Householder_Bidiag_Decomposition(A)
    print("*************************************")
    print("Test random 100x500 matrix")
    #print("A", transform(A_n))
    #print("U left", U_left)
    #print("U right", U_right)
    print("norm of the difference ||A-UBV||:", np.linalg.norm(A-U_left@A_n@U_right.T))
    print("*************************************")
    print(transform(A_n))

if __name__ == "__main__":
    main_test()