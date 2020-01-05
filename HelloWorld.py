import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

USAhousing = pd.read_csv('USA_Housing.csv')
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
predict= "Price"



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LinearRegression

lm=LinearRegression()

lm.fit(X_train,y_train)
acc=lm.score(X_test,y_test)

print (acc)
print('Coefficients :\n',lm.coef_)
print('Intercept :\n',lm.intercept_)
predictions = lm.predict(X_test)
