from process_data import data_columns
import numpy as np
from lr_ import lr

x_train,y_train=data_columns()
# print(x_train,y_train)

print("Training started...")
w_f,b_f=lr.gradient_descent(0,0,1000000,x_train,y_train,0.00005)

print("Training finished")

print(f"w={w_f}\nb={b_f}")

np.save("weights.npy", np.array([w_f, b_f]))

print("Training data saved")