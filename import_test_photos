# Anthony Greene's code
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the directory where the images are located
image_dir = '/content/drive/MyDrive/asl_alphabet_test/'

# List all image files in the directory
image_files = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

# Initialize lists to hold image arrays and labels
image_arrays = []
image_labels = []

# Load images and labels, and associate labels with filenames
for filename in image_files:
    try:
        img = Image.open(filename)
        img_array = np.array(img)
        image_arrays.append(img_array)

        label = os.path.basename(filename).split('_')[0]
        image_labels.append(label)
    except Exception as e:
        print(f"An error occurred while processing {filename}: {e}")

# Create a dictionary to associate filenames with labels
filename_to_label = {filename: label for filename, label in zip(image_files, image_labels)}

# Convert to NumPy arrays
image_arrays = np.array(image_arrays)
image_labels = np.array(image_labels)

# Label Encoding
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(image_labels)

# Data Splitting
X_train, X_temp, y_train, y_temp = train_test_split(image_arrays, y_encoded, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Image Normalization
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Check the shape and data type of the resulting sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Training set data type:", X_train.dtype, y_train.dtype)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Validation set data type:", X_val.dtype, y_val.dtype)
print("Test set shape:", X_test.shape, y_test.shape)
print("Test set data type:", X_test.dtype, y_test.dtype)

# Uncomment below if you want to visualize the images
for img_array in image_arrays:
     plt.imshow(img_array)
     plt.axis('off')
     plt.show()

print("Unique labels:", np.unique(image_labels))
