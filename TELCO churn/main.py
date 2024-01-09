from libraries import feature_eng
from libraries import model

file_path = 'Telco-Customer-Churn.csv'

# Clean the data
churn_data = feature_eng.clean_the_data(file_path)

# Feature engineering
churn_data_f_eng = feature_eng.feature_eng(churn_data)

# Model evaluation
model_performance = model.evaluating_the_models(churn_data_f_eng.drop('Churn', axis=1), churn_data_f_eng['Churn'])

# Print model_performance:
for key, value in model_performance.items():
    print(f"{key}: {value}")

"""
Logistic Regression and 
Gradient Boosting gives the best results.
One of them can be used for the future data to make the best predictions!
"""
