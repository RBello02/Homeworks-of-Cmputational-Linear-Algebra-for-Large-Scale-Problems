import numpy as np

def l1_norm(x):
    x = np.real(np.ravel(x))
    return np.sum(np.abs(x))
    #return np.linalg.norm(x,1)

def Opt_PWM(A,m,x_0,k):

    # checks over m
    if not (m>=0 and m <=1):
        raise ValueError ("m is not between 0 and 1")

    # checks over M
    if A.shape[0] != A.shape[1]:
        raise ValueError ("The matrix is not squared")
    
    n = A.shape[0]

    x_0 = x_0.copy().reshape(-1,1)   # column vectors
    x = x_0.copy()
    x = x / np.sum(np.abs(x))

    one_vec = (np.ones(n)/n).reshape(-1,1)

    for p in range(k):
        y = (1-m)*A@x + m*one_vec
        x = y.reshape(-1,1)
        x = x / np.sum(np.abs(x))

    return x


def main_test():

    print("\n")
    print("******** Test 1 *********")
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

    x= Opt_PWM(A,m,np.ones((n,1)), 1000)
    print(l1_norm(x.reshape(-1,1)-score.reshape(-1,1)))



if __name__ == "__main__":
    main_test()

