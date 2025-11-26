# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
#Dataset: Date_Fruit_Dataset
#File Type:.csv
path = "/content/Date_Fruit_Dataset.csv"
fruit_df = pd.read_csv(path)

#Display some rows from the dataset
fruit_df.head()

# Inspect the dataset (The shape, list of columns and check for missing values)
# Display the shape of the dataset (row,column)
print(f"Dataset Shape: {fruit_df.shape}")
# Display all the coulumn in the dataset
print(f"Column Names: {fruit_df.columns.tolist()}")
# Display the number of missing values exist for each column
print(f"Missing Values:\n{fruit_df.isnull().sum()}")
print(fruit_df.isnull().sum())

# Separate the features and target (X - new features dataset, Y - new target dataset)
# features - all column except for 'Class'
X = fruit_df.drop(columns=["Class"])
# target - 'Class'
y = fruit_df["Class"]

# Convert text labels into numbers
label_encoder = LabelEncoder()
# Applies the encoder to the target dataset y and
# Converts the text labels into numbers
y_encoded = label_encoder.fit_transform(y)

#Split the data (Training Sets, Testing Sets)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
# X_train, y_train - features and target for training
# X_test, y_test - features and target for testing
# test_size=0.2 - Testing (20% ), Training (80%)
# random_state=42 - get the same split every time run the code

# Scale the features
# create scalar object to standadize the data
scaler = StandardScaler()
# fits the scaler to the training data and transforms it
X_train_scaled = scaler.fit_transform(X_train)
# transforms the test data using the same scaling rules
X_test_scaled = scaler.transform(X_test)

# Initialize and train the KNN Model
# set 5 as the number of neighbors
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
# train the model
knn.fit(X_train_scaled, y_train)

# Make prediction and evaluate the model
# make predictions on the test set
y_pred = knn.predict(X_test_scaled)
# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_output = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# print results
print(f"Accuracy with k-NN (k={k}): {accuracy * 100:.2f}%")
print("\nClassification Report for k-NN:\n")
print(classification_report_output)
