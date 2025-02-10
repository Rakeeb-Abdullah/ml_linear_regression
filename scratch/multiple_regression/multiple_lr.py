import numpy as np
np.set_printoptions(precision=2)  

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 0
w_init = np.zeros(X_train.shape[1])

def predict(w,b,X):
    p=np.dot(w,X)+b
    return p

def compute_cost(X,y,w,b):
    m=X.shape[0]
    cost=0.0
    for i in range(m):
        p=predict(w,b,X[i])
        cost+=(p-y[i])**2
    return cost/(2*m)

def compute_gradient(X,y,w,b):
    m,n=X.shape
    dj_db=0.
    dj_dw=np.zeros((n,))
    for i in range(m):
        err=predict(w,b,X[i])-y[i]
        for j in range(n):
            dj_dw[j]+=err*X[i,j]
        dj_db+=err
    
    return dj_dw/m,dj_db/m

def gradient_descent(X,y,w,b,iterations,a):
    
    for i in range(iterations):
        dj_dw,dj_db=compute_gradient(X,y,w,b)
        w-=a*dj_dw
        b-=a*dj_db

        if i%10==0:
            print(compute_cost(X,y,w,b))
    return w,b
    

w_f,b_f=gradient_descent(X_train,y_train,w_init,b_init,1000,5.0e-7)
print(w_f,b_f)