import numpy as np

def Gramschmidt_Decomposition(X):
    """
    Function that performs the Modified Gram-Schmidt method for a given square Matrix (changing the code is
    generalizable to rectangular matrices)
    The matrix must have full rank.
    :param X: square matrix represented as 2D-array object (numpy ndarray);
    :return Q: 2D-array (orthogonal matrix) 
    :return R: 2D-array (upper triangle).
    :return Qqual: norm ||In - Q @ Q.T||
    :return QRqual: norm ||Q.T @ X - R||
    """

    m, n = X.shape
    
    if m != n:
        print('MATRIX IS NOT SQUARE!')
        return None, None, None, None
    
    # Initialization of the matrices
    R = np.zeros((m,n))
    Q = np.zeros((m,n))

    R[0, 0] = np.linalg.norm(X[:,0])
    Q[:, 0] = X[:,0]/R[0,0]

    for j in range(1, n):
        Q[:,j] = X[:,j].copy()       # all'inizio Qj è inizializzato come Xj
        for i in range(0,j):
            R[i,j] = np.dot(Q[:,j],Q[:,i])
            Q[:,j] = Q[:,j] - R[i,j]*Q[:,i]
        R[j,j] = np.linalg.norm(Q[:,j])
        Q[:,j] = Q[:,j]/R[j,j]
        
    Qqual = np.linalg.norm(np.eye(n)-Q@Q.T)
    
    QRqual = np.linalg.norm(Q@R-X)

    return Q, R, Qqual, QRqual

def main_test():

    A = np.random.rand(4,4)
    Q, R, Qqual, QRqual = Gramschmidt_Decomposition(A)
    print("*************************************")
    print("Test random 4x4 matrix")
    print("||I - Q @ Q.T|| =", Qqual)
    print("||Q @ R - A|| =", QRqual)
    print("Q", Q)
    print("R", R)
    print("*************************************")

    A = np.random.rand(50,50)
    Q, R, Qqual, QRqual = Gramschmidt_Decomposition(A)
    print("*************************************")
    print("Test random 50x50 matrix")
    print("||I - Q @ Q.T|| =", Qqual)
    print("||Q @ R - A|| =", QRqual)
    print("*************************************")

    A = np.random.rand(200,200)
    Q, R, Qqual, QRqual = Gramschmidt_Decomposition(A)
    print("*************************************")
    print("Test random 200x200 matrix")
    print("||I - Q @ Q.T|| =", Qqual)
    print("||Q @ R - A|| =", QRqual)
    print("*************************************")

if __name__ == "__main__":
    main_test()
