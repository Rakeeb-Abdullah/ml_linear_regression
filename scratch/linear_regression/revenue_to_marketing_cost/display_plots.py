import matplotlib.pyplot as plt # type: ignore
import numpy as np
from process_data import data_columns

x_train,y_train=data_columns()

w_f,b_f=np.load("weights.npy")



x_range = np.linspace(0, 300, 100)
y_line = w_f * x_range + b_f
plt.plot(x_range, y_line, color='blue', label=f'y = {w_f}x + {b_f}')

plt.scatter(x_train,y_train,marker='x',c='r')
plt.title("Money spent on TV ads VS Revenue")
plt.ylabel("Revenue")
plt.xlabel("Money Spent")

plt.show()