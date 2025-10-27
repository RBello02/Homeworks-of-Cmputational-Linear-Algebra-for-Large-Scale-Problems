import numpy as np

def givens_mat(X, h, k):
    """
    Function that compute the Givens matrix for a given square matrix X and with respect to row h and column k
    :param X: square matrix represented as 2D-array object (numpy ndarray);
    :param h: integer value in the range of the number of X's rows;
    :param k: integer value in the range of the number of X's columns;
    :return G: the Givens matrix as 2D-array object (numpy ndarray).
    """
    
    m, n = X.shape
    
    if m != n:
        print('MATRIX IS NOT SQUARE!')
        return None
    
    # d (denominator of both c and s) can be written as:
    # d = np.sqrt(X[k, k]**2 + X[h, k]**2)
    # But is better (due to numerical problems) to use the hypot function.
    d =np.sqrt(X[k,k]**2 + X[h,k]**2)
    
    c = X[k,k]/d
    s = X[h,k]/d 

    G = np.eye(n)
    G[h,k] = -s
    G[k,h] = s
    G[h,h] = c
    G[k,k] = c

    return G

def Givens_Decomposition(X):
    """
    Function that performs the Givens method for a given square matrix X.
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

    for j in range(n):
        for i in range(n-1,j,-1):
            G = givens_mat(R,i,j)
            R = G@R
            Q = Q@G.T

    Qqual = np.linalg.norm(np.eye(n)-Q@Q.T)    
    QRqual = np.linalg.norm(Q.T@X-R) 

    return Q, R, Qqual, QRqual

def main_test():

    A = np.random.rand(4,4)
    Q, R, Qqual, QRqual = Givens_Decomposition(A)
    print("*************************************")
    print("Test random 4x4 matrix")
    print("||I - Q @ Q.T|| =", Qqual)
    print("||Q @ R - A|| =", QRqual)
    print("Q", Q)
    print("R", R)
    print("*************************************")

    A = np.random.rand(50,50)
    Q, R, Qqual, QRqual = Givens_Decomposition(A)
    print("*************************************")
    print("Test random 50x50 matrix")
    print("||I - Q @ Q.T|| =", Qqual)
    print("||Q @ R - A|| =", QRqual)
    print("*************************************")

    A = np.random.rand(200,200)
    Q, R, Qqual, QRqual = Givens_Decomposition(A)
    print("*************************************")
    print("Test random 200x200 matrix")
    print("||I - Q @ Q.T|| =", Qqual)
    print("||Q @ R - A|| =", QRqual)
    print("*************************************")

if __name__ == "__main__":
    main_test()