import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

# Read the data from the CSV file
data_path = 'FAU_Bank_Employee_Wellbeing.csv'
data = pd.read_csv(data_path)

# Map categorical data to numeric for analysis
age_categories = {'Less than 20': 1, '21 to 35': 2, '36 to 50': 3, '51 or more': 4}
data['AGE'] = data['AGE'].map(age_categories)
gender_categories = {'Male': 0, 'Female': 1}
data['GENDER'] = data['GENDER'].map(gender_categories)
job_role_categories = {
    'Bank Teller': 1, 'Business Analyst': 2, 'Credit Analyst': 3, 'Customer Service': 4, 
    'Finance Analyst': 5, 'Human Resources': 6, 'Investment Banker': 7, 'Loan Processor': 8, 
    'Mortgage Consultant': 9, 'Risk Analyst': 10
}
data['JOB_ROLE'] = data['JOB_ROLE'].map(job_role_categories)

# Impute missing values with the mean for numerical stability
features = data.drop(columns=['WORK_LIFE_BALANCE_SCORE'])
target = data['WORK_LIFE_BALANCE_SCORE']
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.2, random_state=42)

# Fit a Linear Regression model to the training data
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict using the trained model on the test data
predictions = linear_model.predict(X_test)

# Calculate the R² score to evaluate the model
r2_value = r2_score(y_test, predictions)
print(f"R² score: {r2_value}")

# Display the actual vs predicted values for the test set
comparison = pd.DataFrame({
    'Actual': y_test.reset_index(drop=True),
    'Predicted': predictions,
    'Difference': y_test.reset_index(drop=True) - predictions
})
print("\nComparison of actual vs. predicted values:\n", comparison)

# Plot the actual vs. predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values Scatter Plot')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.show()

# Predicting the WLB score for a hypothetical new employee
new_employee_data = {
    'Employee_ID': 2222,
    'JOB_ROLE': 1,  # Bank Teller
    'DAILY_STRESS': 2,
    'WORK_TRAVELS': 2,
    'TEAM_SIZE': 5,
    'DAYS_ABSENT': 0,
    'WEEKLY_EXTRA_HOURS': 5,
    'ACHIEVED_BIG_PROJECTS': 2,
    'EXTRA_HOLIDAYS': 0,
    'BMI_RANGE': 1,
    'TODO_COMPLETED': 6,
    'DAILY_STEPS_IN_THOUSAND': 5,
    'SLEEP_HOURS': 7,
    'LOST_VACATION': 5,
    'SUFFICIENT_INCOME': 1,
    'PERSONAL_AWARDS': 4,
    'TIME_FOR_HOBBY': 0,
    'AGE': 2,  # 21 to 35
    'GENDER': 0  # Male
}

# Convert the new employee data into a DataFrame
new_employee_df = pd.DataFrame([new_employee_data])

# Impute any missing values in the new employee data
new_employee_imputed = imputer.transform(new_employee_df)

# Predict the WLB score for the new employee
predicted_wlb = linear_model.predict(new_employee_imputed)
print(f"\nPredicted Work-Life Balance score for the new employee: {predicted_wlb[0]}")
