import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load the final data
data = pd.read_csv('final-results/Finalni_Podaci.csv')

y = data['Maximum Installs']
X = data.drop('Maximum Installs', axis=1)

# Take 100 examples for testing
_, X_test, _, y_test = train_test_split(X, y, test_size=1000, random_state=42)

# Load the trained model
model = joblib.load('final-results/main_model.joblib')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Display the first 30 predictions and their actual values
comparison = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
print(comparison.head(100))

# Save the test results
comparison.to_csv('final-results/test_results.csv', index=False)
