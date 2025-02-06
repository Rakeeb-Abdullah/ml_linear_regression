import numpy as np
import matplotlib.pyplot as plt # type: ignore


def calculate_gradient(w,b,x,y):
    m=len(x)
    dj_dw=0
    dj_db=0
    for i in range(m):
        f_wb=w*x[i]+b
        dj_dw+=(f_wb-y[i])*x[i]
        dj_db+=(f_wb-y[i])
    return dj_dw/m, dj_db/m

def calculate_cost(w,b,x,y):
    cost=0
    for i in range(len(x)):
        f_wb=w*x[i]+b
        cost+=(f_wb-y[i])**2
    return cost/(2*len(x))

def gradient_descent(w_ini,b_ini,iterations,x,y,a):
    w=w_ini
    b=b_ini

    for i in range(iterations):
        dj_dw,dj_db=calculate_gradient(w,b,x,y)

        w=w-a*dj_dw
        b=b-a*dj_db

        if i%1000==0:
            print(f"Iteration {i}: Cost = {calculate_cost(w, b, x, y):.4f}") 

    return w,b

