import numpy as np

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

def main_test():

    A = np.random.rand(4,4)
    Q, R, Qqual, QRqual = Householder_Decomposition(A)
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

if __name__ == "__main__":
    main_test()