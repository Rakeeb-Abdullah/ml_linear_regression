import numpy as np
from sklearn.linear_model import LinearRegression # type: ignore

x_train=np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y_train=np.array([23, 45, 50, 68, 77, 75,87, 88, 97, 95])

model=LinearRegression()
model.fit(x_train,y_train)

print(model.coef_,model.intercept_)

prediction=model.predict([[4.5],[1.3],[6.7],[9.9]])
print(prediction)