import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('FAU_Bank_turnover.csv')

# Define mappings for job_role and salary
job_role_mapping = {
    'bank_teller': 1, 'business_analyst': 2, 'credit_analyst': 3, 'customer_service': 4,
    'finance_analyst': 5, 'hr': 6, 'investment_banker': 7, 'IT': 8, 'loan_analyst': 9, 'mortgage_consultant': 10
}
salary_mapping = {'low': 1, 'medium': 2, 'high': 3}

# Apply the mappings
df['job_role'] = df['job_role'].map(job_role_mapping)
df['salary'] = df['salary'].map(salary_mapping)

# Data binning for job satisfaction and last performance evaluation
df['job_satisfaction_bin'] = pd.cut(df['job_satisfaction_level'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])
df['performance_evaluation_bin'] = pd.cut(df['last_performance_evaluation'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])

# Create a new feature combining number of projects and working hours
df['projects_hours'] = df['completed_projects'] * df['average_working_hours_monthly']

# Prepare the feature matrix and target variable
X = df[['job_satisfaction_level', 'engagement_with_task', 'last_performance_evaluation',
        'completed_projects', 'average_working_hours_monthly', 'years_spent_with_company',
        'received_support', 'promotion_last_5years', 'job_role', 'salary', 'projects_hours']]

y = df['left']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# Initialize and train the Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train, y_train)

# Make predictions
y_pred = gb_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the performance metrics
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': gb_clf.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()
