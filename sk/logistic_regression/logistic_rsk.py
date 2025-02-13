import numpy as np
from sklearn.linear_model import LogisticRegression # type: ignore


X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

lr_model=LogisticRegression()
lr_model.fit(X_train,y_train)

y_pred=lr_model.predict(X_train)
print(y_pred)
print(lr_model.score(X_train,y_train))

w,f=lr_model.coef_,lr_model.intercept_

print(w,f)