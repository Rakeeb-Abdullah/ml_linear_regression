import numpy as np

w_f,b_f=np.load("weights.npy")


print("Training data loaded")

hour=float(input("How many hours did you study?"))
marks=w_f*hour+b_f

print(f"You will get {marks}")