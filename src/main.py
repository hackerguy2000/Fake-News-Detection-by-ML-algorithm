import pandas as pd
from data_preprocessing import preprocess_data
from model import train_model, predict
from evaluation import evaluate_model

# Load the dataset
data = pd.read_csv('data/train.csv')

# Preprocess the data
X, Y = preprocess_data(data)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the model
model = train_model(X_train, Y_train)

# Make predictions
Y_pred = predict(model, X_test)

# Evaluate the model
accuracy = evaluate_model(Y_test, Y_pred)

print(f'Accuracy: {accuracy}')
