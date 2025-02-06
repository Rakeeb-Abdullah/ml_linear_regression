import matplotlib.pyplot as plt # type: ignore
import numpy as np

x_train=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train=np.array([23, 45, 50, 68, 77, 75,87, 88, 97, 95])

w_f,b_f=np.load("weights.npy")



x_range = np.linspace(0, 11, 100)
y_line = w_f * x_range + b_f
plt.plot(x_range, y_line, color='red', label=f'y = {w_f}x + {b_f}')

plt.scatter(x_train,y_train,marker='x',c='r')
plt.title("Hours studied ans marks achieved")
plt.ylabel("Marks")
plt.xlabel("Hours studied")

plt.show()