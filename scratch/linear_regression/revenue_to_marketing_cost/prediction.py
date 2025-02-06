import numpy as np
money_spent=float(input("How much did you spend on adds?"))

w_f,b_f=np.load("weights.npy")
expected_revenue=money_spent*w_f+b_f

print(f"You can expect a revenue of {expected_revenue}")