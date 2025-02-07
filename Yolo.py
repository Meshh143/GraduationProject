import torch
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Define dataset path
dataset_path = r"C:\Users\arar6\Desktop\Data224"

# Load a YOLO model for classification
model = YOLO("yolov8n-cls.pt")  # Using YOLOv8 classification model

# Train the model and capture training results
train_results = model.train(data=dataset_path, epochs=10, imgsz=224, batch=16)

# Retrieve accuracy and loss from saved logs
accuracy_per_epoch = train_results.metrics.get("accuracy_top1", [])  # Get accuracy per epoch
loss_per_epoch = train_results.metrics.get("loss", [])  # Get loss per epoch

# Ensure accuracy/loss exist before plotting
if accuracy_per_epoch and loss_per_epoch:
    epochs = range(1, len(accuracy_per_epoch) + 1)

    # Plot Accuracy Per Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracy_per_epoch, marker='o', linestyle='-', label="Top-1 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Top-1 Accuracy Per Epoch")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Loss Per Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_per_epoch, marker='o', linestyle='-', color='red', label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Per Epoch")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("Could not retrieve accuracy/loss per epoch.")

# Evaluate the model on validation data
metrics = model.val()
print("Validation Metrics:", metrics)

# Extract available metrics
top1_accuracy = metrics.top1
top5_accuracy = metrics.top5
print(f"Top-1 Accuracy: {top1_accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}")

# Path to the test directory
test_path = Path(dataset_path) / "test"
all_labels = []
all_predictions = []

# Iterate through each class folder in the test directory
for folder in test_path.iterdir():
    if folder.is_dir():
        true_label = folder.name
        # Process each image in the class folder
        for image_path in folder.glob("*.*"):
            # Perform prediction on the image
            results = model.predict(str(image_path))
            
            # Get the probabilities from the prediction results
            pred_probs = results[0].probs.data
            
            # Get predicted class index
            pred_class_idx = torch.argmax(pred_probs).item()
            
            # Map the index to the corresponding label
            pred_label = model.names[pred_class_idx]
            
            all_labels.append(true_label)
            all_predictions.append(pred_label)

# Generate confusion matrix
class_names = list(model.names.values())
cm = confusion_matrix(all_labels, all_predictions, labels=class_names)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
