import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import the dataset from the CSV file
data = pd.read_csv('FAU_Bank_turnover.csv')

# Define mappings for job roles and salary levels
role_mapping = {
    'bank_teller': 1, 'business_analyst': 2, 'credit_analyst': 3, 'customer_service': 4,
    'finance_analyst': 5, 'hr': 6, 'investment_banker': 7, 'IT': 8, 'loan_analyst': 9, 'mortgage_consultant': 10
}
salary_mapping = {'low': 1, 'medium': 2, 'high': 3}

# Apply mappings to the dataset
data['job_role'] = data['job_role'].map(role_mapping)
data['salary'] = data['salary'].map(salary_mapping)

# Bin job satisfaction and last performance evaluation into categories
data['job_satisfaction_category'] = pd.cut(data['job_satisfaction_level'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])
data['performance_category'] = pd.cut(data['last_performance_evaluation'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])

# Create a new feature representing work intensity
data['work_intensity'] = data['completed_projects'] * data['average_working_hours_monthly']

# Display the processed dataset
print("\nProcessed Data Preview:")
print(data.head())

# Calculate the mean job satisfaction level for employees who have left the company
mean_satisfaction_left = data[data['left'] == 1]['job_satisfaction_level'].mean()
print(f'\nMean job satisfaction level of employees who left: {mean_satisfaction_left}')

# Determine the most common salary level among employees who left
common_salary_left = data[data['left'] == 1]['salary'].map({1: 'low', 2: 'medium', 3: 'high'}).mode()[0]
print(f'Most common salary level among employees who left: {common_salary_left}')

# Calculate the average tenure of employees who left the organization
average_tenure_left = data[data['left'] == 1]['years_spent_with_company'].mean()
print(f'Average tenure for employees who left: {average_tenure_left} years')

# Analyze the relationship between salary level and turnover
turnover_by_salary = data.groupby('salary')['left'].mean()
# Revert the salary mapping for interpretability
salary_labels = {v: k for k, v in salary_mapping.items()}
turnover_by_salary.index = turnover_by_salary.index.map(salary_labels)

print(f'\nTurnover rate by salary level:\n{turnover_by_salary}')

# Generate the correlation matrix for the dataset
correlation_data = data.corr()

# Display the correlation matrix
print('\nCorrelation Matrix:')
print(correlation_data)

# Show correlations with the 'left' column
print('\nCorrelations with "left" column:')
print(correlation_data['left'].sort_values(ascending=False))

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Correlation Matrix')
plt.show()
