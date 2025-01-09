# Step 1: Data Exploration
# We'll begin by exploring the dataset. Let's load the Boston Housing dataset for this project:
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
boston_data = load_boston()
boston_df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston_df['Price'] = boston_data.target

# Display basic info and the first few rows
print(boston_df.info())
print(boston_df.head())
#This code loads the Boston Housing dataset, creates a DataFrame with the features, and adds the target variable Price.

# Step 2: Data Preprocessing
# Before proceeding with the analysis, we must check for missing data, outliers, or any other data inconsistencies.
# Check for missing values
print(boston_df.isnull().sum())

# Check for outliers using boxplots
sns.boxplot(data=boston_df, orient='h')
plt.show()

# If any feature has outliers, we may need to remove or transform them.
# In this case, the Boston Housing dataset doesn't have missing values or major outliers, but we would perform more rigorous checks on larger datasets.

# Step 3: Exploratory Data Analysis (EDA)
# Next, we perform some basic EDA to understand the relationships between features and the target variable Price. This step helps in identifying any trends, correlations, or insights that could guide model development.
# Correlation heatmap to understand relationships between features
corr_matrix = boston_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# Scatter plot to analyze the relationship between key features and price
sns.scatterplot(x='RM', y='Price', data=boston_df)
plt.title('Relationship between RM (Average Rooms) and Price')
plt.xlabel('Average Rooms')
plt.ylabel('Price')
plt.show()
# This step helps us identify the most relevant features for predicting house prices. For instance, RM (average number of rooms per dwelling) often shows a high correlation with the target variable Price.

"""
Step 4: Feature Engineering
We might decide to create new features or modify existing ones based on the analysis. For example, if we observe that the number of rooms (RM) is highly correlated with the price, we may create a new feature combining RM with other relevant features.
In this case, we’ll proceed with the features provided, but for larger datasets, we might create new ones, such as interaction terms or binning certain variables (e.g., categorizing the number of rooms into groups).
"""

# Step 5: Model Building
# Now, we can build a predictive model. We’ll use Linear Regression as a starting point, as it’s simple and interpretable.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Split the data into training and testing sets
X = boston_df.drop('Price', axis=1)  # Features
y = boston_df['Price']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
"""
In this code:
•	We split the data into training and testing sets.
•	We train a Linear Regression model.
•	We evaluate the model using the Mean Squared Error (MSE) and R-squared metrics.
"""

"""
Step 6: Model Evaluation
We’ll check how well our model is performing using metrics like R-squared, which shows how well the model explains the variance in the target variable. A value closer to 1 indicates a good fit.
If the performance is not satisfactory, we can try more complex models such as Random Forest, XGBoost, or Lasso Regression.
"""

"""
Step 7: Final Conclusion
Based on the results from the model evaluation, we can:
•	Interpret the performance of the linear regression model.
•	Suggest areas for model improvement, such as trying different algorithms, adding polynomial features, or performing hyperparameter tuning.
•	If the results are good, we can summarize the most important features contributing to price prediction.
"""
