- 👋 Hi, I’m @bothafrik
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
bothafrik/bothafrik is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
# Import the necessary libraries

import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

 

# Load the data for the teams and games

data = pd.read_csv("data.csv")

 

# Split the data into training and test sets

train_data = data[:800]

test_data = data[800:]

 

# Extract the features and target variable

X_train = train_data.drop("outcome", axis=1)

y_train = train_data["outcome"]

X_test = test_data.drop("outcome", axis=1)

y_test = test_data["outcome"]

 

# Create a linear regression model

model = LinearRegression()

 

# Train the model on the training data

model.fit(X_train, y_train)

 

# Make predictions on the test data

predictions = model.predict(X_test)

 

# Evaluate the model's performance

mse = np.mean((predictions - y_test)**2)

print("Mean Squared Error:", mse)
