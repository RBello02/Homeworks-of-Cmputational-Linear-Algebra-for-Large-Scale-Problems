import numpy as np
import warnings
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix

def l1_norm(x):
    x = np.real(np.ravel(x))
    return np.sum(np.abs(x))
    #return np.linalg.norm(x,1)

def SX_sinkNodes(x,n,m,sink_nodes):
    s = m/n*np.ones((n,1))
    s[sink_nodes] = 1/n     # this factor will make the M matrix stochastic
    prod = np.dot(x.T,s)[0][0]
    res = np.ones((n,1))*prod
    return res


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
        real_x = np.real(real_x)
        real_x = real_x/l1_norm(real_x)
        real_x = real_x.copy().reshape(-1,1)
        #print(real_x)

    for p in range(k):
        y = M@x
        normalization = l1_norm(y) 
        if normalization < tol:
            raise ValueError ("obtained an null eigenvector")
        y = y/normalization   # computing the new approximation
        y = y.reshape(-1,1)  # column vector
        if real_x is not None and np.dot(y.ravel(), real_x.ravel()) < tol:
            y = -y
        if real_x is not None:
            diff_y = y.reshape(-1,1)-real_x.reshape(-1,1)
            '''
            if p == k-1:
                print(np.linalg.norm(diff_y, 1))
                print(np.sum(np.abs(diff_y)))
                print(np.sum(np.abs(diff_y.ravel())))
                print(np.sum(np.abs(np.real(diff_y))))
                print(l1_norm(diff_y.reshape(-1,1)))
                print("\n")
            '''
            x_seq[p] = l1_norm(diff_y.reshape(-1,1))
            if p > 0:
                ratio = x_seq[p]/x_seq[p-1]
        x = y.copy()   # adjusting x


    # computation of lam

    z = M@x
    for k in range(len(x)):
        if abs(x[k]) > tol:
            lam = float((z[k]/x[k]).item())
            break

    if real_x is None:
        x_seq = None

    return x,lam,c,x_seq,ratio

def Opt_PWM(A,m,x_0,k=1000, tol = 10**-8, sink_nodes = None):

    # checks over m
    if not (m>=0 and m <=1):
        raise ValueError ("m is not between 0 and 1")

    # checks over A
    if not issparse(A):
        raise ValueError("The matrix is not in sparse format, use PWM")
    if A.shape[0] != A.shape[1]:
        raise ValueError ("The matrix is not squared")
    
    n = A.shape[0]

    # computation of the factor c

    # computing M in a sparse way

    M = A.copy().tocsc()    
    M = M*(1-m)
    M.data += m/n     # do that only on the numbers not zero. Use this to preserve the sparsity (changes the computation of c)

    # computing the factor c
    c=0
    s = np.array(M.sum(axis = 0)).flatten()   # array that contains the sums over the columns
    for j,el in enumerate(s):   # cycle over s
        if abs(el-1)<tol: # it means that all the elements sums to 1 so the other are 0, so is not positive , so we must search the min over M
            col = M.getcol(j)
            M_ij_min_i = col.data.min()
        else:
            M_ij_min_i = m/n
        val = abs(1-2*M_ij_min_i)
        if val > c:   # save the max
            c = val

    x_0 = x_0.copy().reshape(-1,1)   # column vectors
    x = x_0.copy()
    x = x / np.sum(np.abs(x))

    one_vec = (np.ones(n)/n).reshape(-1,1)
    difference_in_norm_l1 = []

    for p in range(k):
        if sink_nodes is None:
            y = (1-m)*A@x + m*one_vec
        else:
            y  = (1-m)*A@x + SX_sinkNodes(x,n,m,sink_nodes).reshape(-1,1)
        y = y.reshape(-1,1)
        y = y / np.sum(np.abs(y))
        diff = l1_norm(y-x)
        difference_in_norm_l1.append(diff)
        x = y.copy()
        if abs(diff)<tol:
            print("The method stopped before k=",k)
            break
        
    if sink_nodes is None:
        Mx = (1-m)*A@x + m*one_vec
    else:
        Mx  = (1-m)*A@x + SX_sinkNodes(x,n,m,sink_nodes).reshape(-1,1)
    lam = float((Mx.T @ x) / (x.T @ x))


    return x,lam,c,p+1,difference_in_norm_l1

def page_rank(A,m,x_0,sink_nodes, no_backlink_nodes, k=1000, tol = 10**-8):
    x,lam,c,p,difference_in_norm_l1 = Opt_PWM(A,m,x_0,k,tol,sink_nodes=sink_nodes)

    n = len(x)
    result = np.zeros((n,1))
    residual_mass = 1-len(no_backlink_nodes)*m/n
    precedent_mass = 1-sum(x[no_backlink_nodes])
    for j in range(n):
        if j not in no_backlink_nodes:
            result[j] = x[j]*residual_mass/precedent_mass
        else:
            result[j] = m/n

    return result,lam,c,p, difference_in_norm_l1

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
    diff = v1.reshape(-1,1)-v2.reshape(-1,1)
    diff_l1 = l1_norm(diff)
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
    diff = v1.reshape(-1,1)-v2.reshape(-1,1)
    diff_l1 = l1_norm(diff)   
    print("the difference in norm l1: " , diff_l1)
    print("\n")
    print("the c value: ", c)
    print("\n")
    print("the difference of the eigenvalues: ",abs(lambda_dom-lam))
    print("\n")
    print("the ratio: ", ratio)

    print("\n")
    print("******** Test 3 *********")
    print("\n")
    A = np.array([[0,0,1/2,1/2,0],
              [1/3,0,0,0,0],
              [1/3,1/2,0,1/2,1],
              [1/3,1/2,0,0,0],
              [0,0,1/2,0,0]])
    m = 0.15
    n = 5
    S = 1/n*np.ones((n,n))
    M = (1-m)*A+m*S
    values,vectors = np.linalg.eig(M)
    idx = np.argsort(values)[::-1]
    values = values[idx]
    vectors = vectors[:, idx]
    score = vectors[:,0]/sum(vectors[:,0])

    x_50,lam_50,c,x_seq_50,ratio_50 = PWM(M=M,x_0=np.ones((M.shape[0],1)),k=50,tol=10**-8,real_x=score)
    print(x_seq[49])
    print("\n")
    print("factor c: ", c)

    print("\n")
    print("******** Test 1 OPTIMIZED *********")
    print("\n")
    A = np.array([[  0,   0, 1/2, 1/2,  0],
                  [1/3,   0,   0,   0,  0],
                  [1/3, 1/2,   0, 1/2,  1],
                  [1/3, 1/2,   0,   0,  0],
                  [  0,   0, 1/2,   0,  0]])
    A_sparse = csr_matrix(A)
    m = 0.15
    n = 5
    S = 1/n*np.ones((n,n))
    M = (1-m)*A_sparse+m*S
    values,vectors = np.linalg.eig(M)
    idx = np.argsort(values)[::-1]
    values = values[idx]
    vectors = vectors[:, idx]
    score = vectors[:,0]/sum(vectors[:,0])

    x,c= Opt_PWM(A_sparse,m,np.ones((n,1)), 50)
    print(l1_norm(x.reshape(-1,1)-score.reshape(-1,1)))
    print("\n the factor c:", c)

    print("\n")
    print("******** Test 2 OPTIMIZED *********")
    print("\n")
    A = np.array([
                  [1/3,   0,   0],
                  [1/3,   0,   1],
                  [1/3,   1,   0]
                  ])

    m = 0.3
    A_sparse = csr_matrix(A)
    x,c= Opt_PWM(A_sparse,m,np.ones((3,1)), 50)
    print(c)


if __name__ == "__main__":
    main_test()