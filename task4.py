# import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Define the path to the dataset
data_path = Path("C:/Users/amith/task4/leapGestRecog")  # Use the full path if necessary

# Check if the data_path exists and list its contents
if data_path.exists():
    print(f"Contents of '{data_path}':")
    for item in data_path.iterdir():
        print(item)
else:
    print(f"Data path '{data_path}' does not exist.")

# Initialize lists for storing data and labels
X = []
y = []

# Loop through the dataset folders
for i in range(10):  # Assuming 10 gesture categories
    gesture_folder = data_path / f"{i:02d}"  # Format as two digits
    print(f"Checking folder: {gesture_folder.resolve()}")  # Debugging output
    
    if not gesture_folder.exists():
        print(f"Folder '{gesture_folder}' not found. Please verify the path.")
        continue

    # Recursively load images from the current folder and subfolders
    for subfolder in gesture_folder.iterdir():
        if subfolder.is_dir():  # Check if it's a directory
            print(f"Loading images from subfolder: {subfolder}")  # Debugging output
            for img_path in subfolder.glob("*.png"):  # Adjust extension if needed
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (64, 64))  # Resize to 64x64 pixels
                    X.append(img)
                    y.append(i)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Ensure images were loaded
if len(X) == 0 or len(y) == 0:
    raise ValueError("No images were loaded. Please check the dataset path and structure.")

# Normalize pixel values
X = X / 255.0

# Reshape X to add a single channel for grayscale images
X = X.reshape(X.shape[0], 64, 64, 1)

# Convert labels to categorical
y = to_categorical(y, num_classes=10)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

# Build the CNN model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3rd Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and Fully Connected Layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output Layer (10 gesture classes)
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()

# Data augmentation to improve model generalization
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Fit the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=20, verbose=1)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training and validation accuracy/loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Save the model
model.save('gesture_recognition_model.h5')  # Save the model after training

# Real-time gesture recognition
# Load the trained model
model = load_model('gesture_recognition_model.h5')

# Define the labels (assuming you have 10 classes)
gesture_labels = [f"{i:02d}" for i in range(10)]  # ['00', '01', ..., '09']

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Preprocess the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray_frame = cv2.resize(gray_frame, (64, 64))  # Resize to match input size
    gray_frame = gray_frame / 255.0  # Normalize pixel values
    gray_frame = gray_frame.reshape(1, 64, 64, 1)  # Reshape for model input

    # Make prediction
    predictions = model.predict(gray_frame)
    predicted_class = np.argmax(predictions)
    gesture_name = gesture_labels[predicted_class]

    # Display the result
    cv2.putText(frame, f'Predicted Gesture: {gesture_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Gesture Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
