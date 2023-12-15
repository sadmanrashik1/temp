import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Assuming 'your_dataset.csv' is the name of your dataset file
df = pd.read_csv('data.csv')


#dropping irrevelent data 
df=df.drop('Node_ID', axis=1)
df=df.drop('Timestamp', axis=1)
df=df.drop('IP_Address', axis=1)

# Extract features and target variable
X = df.drop('Is_Malicious', axis=1)
y = df['Is_Malicious']

df.head()


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Standardize the data 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and train the neural network model
model = MLPClassifier(hidden_layer_sizes=(2, 2), max_iter=500, random_state=30)
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")