import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
file_path = 'data\FAU_Bank_Employee_Performance.xls'
df = pd.read_excel(file_path)

# Display the first few rows of the dataset
print(df.head())

# Check for missing values in the dataset
missing_values = df.isnull().sum()
print(missing_values)

# Drop unnecessary columns
df.drop(columns=['EmpNumber'], inplace=True)

# Department-wise performance rating mean
dept_performance = df.groupby('EmpDepartment')['PerformanceRating'].mean().sort_values()

# Plot department-wise performance
plt.figure(figsize=(12, 6))
sns.barplot(x=dept_performance.index, y=dept_performance.values, palette='coolwarm')
plt.title('Department-wise Employee Performance Rating')
plt.xlabel('Department')
plt.ylabel('Average Performance Rating')
plt.xticks(rotation=45)
plt.show()

# List of categorical columns
categorical_cols = ['Gender', 'EducationBackground', 'MaritalStatus', 'EmpDepartment', 
                    'EmpJobRole', 'BusinessTravelFrequency', 'OverTime', 'Attrition']

# Initialize LabelEncoder
le = LabelEncoder()

# Apply LabelEncoder to each categorical column
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Display the first few rows of the dataset to verify changes
print(df.head())

# Define features and target variable
X = df.drop(columns=['PerformanceRating'])
y = df['PerformanceRating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions
y_pred = gb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# Get feature importances
feature_importances = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='coolwarm')
plt.title('Feature Importances in Predicting Employee Performance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Display the top important features
print(feature_importances.head(10))

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=gb_model.classes_, yticklabels=gb_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Calculate the correlation matrix
corr_matrix = df.corr()

# Plot the correlation matrix
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Sort the correlation values with respect to PerformanceRating
corr_with_target = corr_matrix['PerformanceRating'].sort_values(ascending=False)
print(corr_with_target)
