import numpy as np
import matplotlib.pyplot as plt # type: ignore
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

def sigmoid(z):
    s=1/(1+np.exp(-z))
    
    return s

def calculate_z(X,w,b):
    z=np.dot(X,w)+b
    return z

def calculate_cost(X,y,w,b):
    cost=0
    m=X.shape[0]

    for i in range(m):
        f_wb_i=sigmoid(calculate_z(X[i],w,b))
        cost+=-1*(y[i]*(np.log(f_wb_i))+((1-y[i])*(np.log(1-f_wb_i))))
    cost=cost/m
    return cost

def calculate_gradient(X,y,w,b):
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0

    for i in range(m):
        f_wb_i=sigmoid(calculate_z(X[i],w,b))
        err_i=f_wb_i-y[i]
        for j in range(n):
            dj_dw[j]+=err_i*X[i,j]
        dj_db+=err_i
    
    return dj_dw/m,dj_db/m

def gradient_descent(X,y,w_init,b_init,a,iterations):
    w=w_init
    b=b_init
    for i in range(iterations):
        dj_dw,dj_db=calculate_gradient(X,y,w,b)
        w-=a*dj_dw
        b-=a*dj_db

        if i%1000==0:
            c=calculate_cost(X,y,w,b)
            print(c)
    return w,b
w_init=np.zeros((X_train.shape[1],))
b_init=0

w_f,b_f=gradient_descent(X_train,y_train,w_init,b_init,0.1,10000)

print(w_f,b_f)


def plot(X,y,w,b):
    m,n=X.shape
    for i in range(m):
        
        plt.scatter(X[i,0],X[i,1],marker="o" if y[i]==0 else "x",color="blue" if y[i]==0 else "red")

    x0=np.arange(-1,3)
    x1=(-b-x0*w[1])/w[0]
    plt.plot(x0,x1)
    plt.show()

plot(X_train,y_train,w_f,b_f)

def predict(X,w,b):
    m=X_train.shape[0]
    prediction=[]
    for i in range(m):
        f_wb_i=sigmoid(calculate_z(X[i],w,b))
        prediction.append([f_wb_i,"->",1 if f_wb_i>=0.5 else 0])
    
    return prediction

prediction=predict(X_train,w_f,b_f)
print(prediction)


