import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Load the data
df = pd.read_csv('homeprices.csv')

# Prepare the data
X = df['area'].values.reshape(-1,1)
y = df['price'].values.reshape(-1,1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()  
model.fit(X_train, y_train)

# Make predictions using the test set
y_pred = model.predict(X_test)

# Print the model coefficients
print('Intercept:', model.intercept_)
print('Coefficient:', model.coef_)

# Evaluate the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))