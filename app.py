import streamlit as st
import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import tempfile

# Define models and their associated fabric types
fabric_models = {
    "1": ("model_knitted_Gray.h5", "knitted_Gray"),
    "2": ("model_knitted_printed.h5", "knitted_printed"),
    "3": ("model_defect_detection.h5", "knitted_Dyed"),
    "4": ("model_woven_Gray.h5", "woven_Gray"),
    "5": ("model_woven_Printed.h5", "woven_Printed"),
    "6": ("model_woven_Dyed.h5", "woven_Dyed")
}

# Streamlit interface
st.title("Fabric Defect Detection")

# Step 1: User selects the fabric defect model
fabric_choice = st.selectbox(
    "Select the fabric defect type you want to detect:",
    list(fabric_models.keys()),
    format_func=lambda x: fabric_models[x][1]
)

# Load the selected model based on user choice
model_path, selected_fabric_type = fabric_models[fabric_choice]
model = load_model(model_path)
st.write(f"Model for '{selected_fabric_type}' loaded successfully.")

# Step 2: Upload the test dataset directory
test_dir = st.text_input("Enter the path to the test dataset:")
st.write("Example path: /path/to/your/test/dataset")

# Function to preprocess and predict on a single image
def predict_defect(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (28, 28))  # Resize to 28x28 pixels
    img = img.astype('float32') / 255.0  # Normalize
    img = img.reshape(1, -1, img.shape[2])  # Reshape for LSTM input
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

# Step 3: Run predictions if dataset path is valid
if test_dir and os.path.isdir(test_dir):
    categories = ['stain', 'damage', 'broken thread', 'holes', 'non defective']
    true_labels = []
    predicted_labels = []

    for category_index, category in enumerate(categories):
        category_path = os.path.join(test_dir, category)
        if not os.path.exists(category_path):
            st.warning(f"Directory {category_path} does not exist. Skipping.")
            continue

        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            if img_file.endswith('.jpg'):
                pred_label = predict_defect(img_path)
                true_labels.append(category_index)
                predicted_labels.append(pred_label)

    # Generate classification report and display it
    unique_labels = np.unique(np.concatenate((true_labels, predicted_labels)))
    class_report = classification_report(
        true_labels, 
        predicted_labels, 
        labels=unique_labels,
        target_names=categories,
        zero_division=0
    )
    st.text("Classification Report:")
    st.text(class_report)

    # Display confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    st.text("Confusion Matrix:")
    st.write(conf_matrix)

    # Plot Confusion Matrix
    fig, ax = plt.subplots()
    ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=45)
    ax.set_yticklabels(categories)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    st.pyplot(fig)

else:
    st.warning("Please provide a valid path to the test dataset.")
