import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the training and testing datasets
training_data = pd.read_csv(r"C:\ECE 4982\Excursions\Excursiontwo\augmented_training_data.csv")  # New training dataset
testing_data = pd.read_csv(r"C:\ECE 4982\Excursions\Excursiontwo\shuffled_testing_data.csv")  # Your separate testing dataset

# Step 2: Split training and testing data into features (X) and target (y)
X_train = training_data[['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']]
y_train = training_data['Letter']

X_test = testing_data[['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']]
y_test = testing_data['Letter']

# Step 3: Encode target labels (Letters) to numerical values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)  # Fit on training data
y_test_encoded = label_encoder.transform(y_test)  # Transform testing data based on training labels

# Step 4: Initialize and train the Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train_encoded)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)

# Adjust the target names based on actual labels in y_test
unique_labels = sorted(set(y_test_encoded) | set(y_pred))  # Get all unique labels in y_test and y_pred
relevant_target_names = [label_encoder.classes_[i] for i in unique_labels]

# Generate the classification report
#accuracy = accuracy_score(y_test_encoded, y_pred)
#report = classification_report(y_test_encoded, y_pred, target_names=relevant_target_names)

#print(f"Accuracy: {accuracy}")
#print("Classification Report:")
#print(report)

# Step 6: Predict new data (Example)
# Add feature names to new input
#new_input = pd.DataFrame([[0.1, 0.9, 0.8, 1.0, 0.9]], columns=['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'])

# Predict the letter
#predicted_class = model.predict(new_input)
#predicted_letter = label_encoder.inverse_transform(predicted_class)

#print(f"Predicted Letter: {predicted_letter[0]}")


# Print all predictions from the testing set
predicted_letters = label_encoder.inverse_transform(y_pred)
print("Predicted Letters from Testing Set:")
print(predicted_letters)

# Manual input for testing
'''print("Enter the finger values separated by spaces: ")
input_values = input().strip().split() 
input_values = [float(value) for value in input_values]

manual_input = pd.DataFrame([input_values], columns=['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'])

predicted_class = model.predict(manual_input)
predicted_letter = label_encoder.inverse_transform(predicted_class)
print(f"Predicated Letter: {predicted_letter[0]}")'''

