# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the Dataset
# Replace 'sales_data.csv' with your actual dataset file
data = pd.read_csv("C:\\Users\\USER\\Documents\\ML\\futuresale prediction.csv")

# Print the first few rows of the dataset to understand its structure
print(data.head())

# Step 3: Preprocess the Data
# Assuming 'sales' is the target variable and others are features
X = data.drop('sales', axis=1)
y = data['sales']

# Identify categorical and numerical columns
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 4: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 5: Train the Model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=0))
])

model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R² Score:', r2_score(y_test, y_pred))

# Step 7: Fine-Tune the Model (optional)
# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'regressor__n_estimators': [100, 200, 300],
#     'regressor__max_features': ['auto', 'sqrt', 'log2'],
# }

# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)

# print(grid_search.best_params_)

# Step 8: Make Predictions
# Let's make a prediction on a new sample (example features)
new_sample = pd.DataFrame({
    'feature1': [value1],
    'feature2': [value2],
    # Add all required features here
})

predictions = model.predict(new_sample)
print('Predicted Sales for new sample:', predictions)

# Optional: Visualize feature importances if using a tree-based model
if hasattr(model.named_steps['regressor'], 'feature_importances_'):
    importances = model.named_steps['regressor'].feature_importances_
    feature_names = numerical_cols + list(model.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_cols))
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importances')
    plt.show()
