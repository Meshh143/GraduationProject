import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset Paths
path_train = r"C:\Users\arar6\Desktop\Data224\train"
path_val = r"C:\Users\arar6\Desktop\Data224\val"
path_test = r"C:\Users\arar6\Desktop\Data224\test"

# Define your categories (update these if necessary)
CATEGORIES = ["Alhlwah", "Berhi", "Khalas", "Meneifi", "Red Sukkari", "Ruthana", "Shishi", "Sullaj"]

# Image size
IMG_SIZE = 224

# Function to load data from a directory
def load_data(path):
    data = []
    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        category_path = os.path.join(path, category)
        
        if not os.path.exists(category_path):
            print(f"Warning: Directory {category_path} not found!")
            continue

        for img in os.listdir(category_path):
            img_path = os.path.join(category_path, img)
            
            if not os.path.isfile(img_path):
                print(f"Warning: File {img_path} not found!")
                continue
            
            img_array = cv2.imread(img_path)
            
            if img_array is None:
                print(f"Warning: Unable to read {img_path}")
                continue
            
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append([new_array, class_num])
    
    random.shuffle(data)
    return data

# Load training, validation, and test data
train_data = load_data(path_train)
# If you have validation data you can use it; otherwise, you can ignore it.
val_data = load_data(path_val) if os.path.exists(path_val) else []
test_data = load_data(path_test)

# If no separate validation data is provided, the training data remains unchanged.
if not val_data:
    train_data += val_data

# Function to split data into features and labels
def split_data(data):
    X = []
    y = []
    for features, label in data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype('float32') / 255
    y = np.array(y)
    return X, y

X_train, y_train = split_data(train_data)
X_test, y_test = split_data(test_data)

# One-hot encode labels
Y_train = to_categorical(y_train, num_classes=len(CATEGORIES))
Y_test = to_categorical(y_test, num_classes=len(CATEGORIES))

# Model parameters
batch_size = 16
nb_epochs = 5

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(CATEGORIES), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# Note: We are using validation_data here for evaluation during training.
# However, for plotting, we will only use the training metrics.
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_test, Y_test))

# Evaluate the model on the test set
score = model.evaluate(X_test, Y_test, verbose=0)
print("Test Score: ", score[0])
print("Test Accuracy: ", score[1])

# Predict the values from the test dataset for further evaluation (optional)
Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

# Confusion Matrix (optional)
cm = confusion_matrix(y_test, y_pred, normalize='true')
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.show()

# Classification Report (optional)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

# -----------------------------------
# Plot only the training accuracy and training loss
# -----------------------------------

# Plot Training Accuracy over Epochs
plt.figure()
plt.plot(history.history['accuracy'], marker='o')
plt.title('Training Accuracy Over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.grid(True)
plt.show()

# Plot Training Loss over Epochs
plt.figure()
plt.plot(history.history['loss'], marker='o', color='red')
plt.title('Training Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.grid(True)
plt.show()
