import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset from a CSV file
data_path = 'FAU_Bank_Employee_Wellbeing.csv'  # Update this path with the correct file location
wellbeing_data = pd.read_csv(data_path)


# Check for missing data in the dataset
missing_data_count = wellbeing_data.isnull().sum()
print("Missing data count:\n", missing_data_count)

# Remove unnecessary columns, e.g., 'Employee_ID' since it might only be an identifier
wellbeing_data.drop(columns=['Employee_ID'], inplace=True)

# Map categorical data to numerical values
# Convert age ranges to numeric categories
age_map = {'Less than 20': 1, '21 to 35': 2, '36 to 50': 3, '51 or more': 4}
wellbeing_data['AGE'] = wellbeing_data['AGE'].map(age_map)

# Convert gender to numeric values
gender_map = {'Male': 0, 'Female': 1}
wellbeing_data['GENDER'] = wellbeing_data['GENDER'].map(gender_map)

# Visualize daily stress levels by gender using a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='GENDER', y='DAILY_STRESS', data=wellbeing_data, ci=None)
plt.xlabel('Gender')
plt.ylabel('Daily Stress')
plt.title('Daily Stress by Gender')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])
plt.show()

# Visualize daily stress levels by job role
plt.figure(figsize=(14, 8))
sns.barplot(x='JOB_ROLE', y='DAILY_STRESS', data=wellbeing_data, ci=None)
plt.xlabel('Job Role')
plt.ylabel('Daily Stress')
plt.title('Daily Stress by Job Role')
plt.xticks(rotation=45)
plt.show()

# Calculate average time spent on hobbies by gender
avg_hobby_time = wellbeing_data.groupby('GENDER')['TIME_FOR_HOBBY'].mean()
# print("Average time for hobby by gender:\n", avg_hobby_time)

# Visualize the average time dedicated to hobbies by gender
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=avg_hobby_time.index, y=avg_hobby_time.values)
plt.xlabel('Gender')
plt.ylabel('Average Time for Hobbies')
plt.title('Average Time for Hobbies by Gender')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])

# Annotate the bar chart with the average values
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')

plt.show()

# Build a correlation heatmap to identify factors related to work-life balance
# Convert job roles to numerical values
job_role_map = {
    'Bank Teller': 1, 'Business Analyst': 2, 'Credit Analyst': 3, 'Customer Service': 4,
    'Finance Analyst': 5, 'Human Resources': 6, 'Investment Banker': 7, 'Loan Processor': 8,
    'Mortgage Consultant': 9, 'Risk Analyst': 10
}
wellbeing_data['JOB_ROLE'] = wellbeing_data['JOB_ROLE'].map(job_role_map)

# Calculate the correlation matrix
corr_matrix = wellbeing_data.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Identify the variables most correlated with the work-life balance score
wlb_correlation = corr_matrix['WORK_LIFE_BALANCE_SCORE'].sort_values(ascending=False)
print("Attributes most correlated with Work-Life Balance score:\n", wlb_correlation)
