import numpy as np
import warnings

def l1_norm(x):
    return np.sum(np.abs(x))

def PWM(M, x_0 ,k=1000, tol = 10**-8,real_x = None):


    ## inputs ##
    # M = the matrix where is applied the power method (in general must be column stochastic)
    # x_0 = initial guess of the eigenvector
    # k = number of matrix multiplications
    # real_x = is possible to insert the real value of x to compute the distance at each iteration 
    # tol = tolerance


    ## outputs ##
    # x = the eigenvector obtained with power method                        ok
    # x_seq = the sequence of the norm ||M^k*x_0-real_x||                   ok
    # lam = the approximation of the eigenvalue of convergence              ok
    # c = max_{j}|1-2*min_{i}M_{ij}|                                        ok
    # ratio = |M^kx_o-x_real|/|M^k-1x_0-x_real|                             ok       



    # checks over M
    if M.shape[0] != M.shape[1]:
        raise ValueError ("The matrix is not squared")
    
    n = M.shape[0]
    

    # checks if the matrix is positive and computing the factor c
    positive = True
    c = 0
    for j in range(n):
        min_M_ij = np.sum(M[:,j]) 
        for i in range(n):
            if M[i,j] < tol:
                positive = False
            if min_M_ij>M[i,j]:
                min_M_ij = M[i,j]
        shrinking = np.abs(1-2*min_M_ij)
        if shrinking > c:
            c = shrinking

    
    if not positive:
        warnings.warn("The matrix is not positive, the convergence of the method is not guaranteed\n ")
        print("\n")

    # POWER METHOD

    x_0 = x_0.copy().reshape(-1,1)   # column vectors

    x_seq = [None]*k
    ratio = None
    x = x_0.copy()

    if real_x is not None:    # normalization of the eigenvector
        real_x = real_x/l1_norm(real_x)

    for p in range(k):
        y = M@x
        normalization = l1_norm(y) 
        if normalization < tol:
            raise ValueError ("obtained an null eigenvector")
        y = y/normalization   # computing the new approximation
        y = y.reshape(-1,1)  # column vector
        if real_x is not None:
            x_seq[p] = l1_norm(y-real_x)
            ratio = x_seq[p]/l1_norm(x-real_x)
        x = y   # adjusting x


    # computation of lam

    z = M@x
    for k in range(len(x)):
        if x[k] > tol:
            lam = float((z[k]/x[k]).item())
            break

    if real_x is None:
        x_seq = None

    return x,lam,c,x_seq,ratio

def main_test():

    print("\n")
    print("******** Test 1 *********")
    print("\n")
    M= np.array([
    [0,1,1,0,1],
    [1,0,1,1,0],
    [1,1,0,1,1],
    [0,1,1,0,1],
    [1,0,1,1,0]
              ], dtype=float)
    
    eigvals, eigvecs = np.linalg.eig(M)
    idx = np.argmax(np.abs(eigvals))
    lambda_dom = eigvals[idx]
    v_dom = eigvecs[:, idx]
    x_0 = np.random.rand(M.shape[0],1)

    x,lam,c,x_seq,ratio = PWM(M,x_0,1000,10**-10,v_dom)
    v1 = x / l1_norm(x)
    v2 = v_dom / l1_norm(v_dom)
    if np.sign(v1[0]) != np.sign(v2[0]):
        v2 = -v2 
    diff_l1 = l1_norm(v1 - v2)
    print("the difference in norm l1: " , diff_l1)
    print("\n")
    print("the c value: ", c)
    print("\n")
    print("the difference of the eigenvalues: ",abs(lambda_dom-lam))
    print("\n")
    print("the ratio: ", ratio)

    print("\n")
    print("******** Test 2 *********")
    print("\n")
    M= np.random.rand(10,10)
    eigvals, eigvecs = np.linalg.eig(M)
    idx = np.argmax(np.abs(eigvals))
    lambda_dom = eigvals[idx]
    v_dom = eigvecs[:, idx]
    x_0 = np.random.rand(M.shape[0],1)

    x,lam,c,x_seq,ratio = PWM(M,x_0,1000,10**-10,v_dom)
    v1 = x / l1_norm(x)
    v2 = v_dom / l1_norm(v_dom)
    if np.sign(v1[0]) != np.sign(v2[0]):
        v2 = -v2 
    diff_l1 = l1_norm(v1 - v2)   
    print("the difference in norm l1: " , diff_l1)
    print("\n")
    print("the c value: ", c)
    print("\n")
    print("the difference of the eigenvalues: ",abs(lambda_dom-lam))
    print("\n")
    print("the ratio: ", ratio)


if __name__ == "__main__":
    main_test()