import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Step 1: Load the dataset
# Assuming you have a dataset 'credit_data.csv' with relevant features and a target column 'credit_score'
# Replace 'credit_data.csv' with your actual dataset file
# Here's a mock data creation for demonstration purposes
data = {
    'age': np.random.randint(20, 70, 1000),
    'income': np.random.randint(20000, 100000, 1000),
    'loan_amount': np.random.randint(1000, 20000, 1000),
    'credit_score': np.random.choice(['good', 'average', 'bad'], 1000)
}
df = pd.DataFrame(data)

# Step 2: Preprocess the data
# Convert categorical target to numerical
df['credit_score'] = df['credit_score'].map({'bad': 0, 'average': 1, 'good': 2})

# Split the data into features and target
X = df.drop('credit_score', axis=1)
y = df['credit_score']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a classification model
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Visualize the results
# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['bad', 'average', 'good'], yticklabels=['bad', 'average', 'good'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
