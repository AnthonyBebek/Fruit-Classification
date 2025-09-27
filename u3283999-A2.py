import kagglehub
import pandas as pd
import pandas as pd
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import os
from sklearn.model_selection import train_test_split
path = str(kagglehub.dataset_download("icebearogo/fruit-classification-dataset")) + "/Fruit_dataset"
print("Path to dataset files:", path)
csv_path = f"{path}/train.csv"
df = pd.read_csv(csv_path)

print(df.head())

# load csv
train_df = pd.read_csv(f"{path}/train.csv")

# image size (small for classical ML)
size = (32, 32)

images, labels = [], []

print(path)

for _, row in train_df.iterrows():
    fname = row["image:FILE"].replace("train/", "train1/").lstrip("/\\")
    file  = os.path.join(path, fname)
    img = load_img(file, target_size=size)
    arr = img_to_array(img) / 255.0
    images.append(arr.flatten())
    labels.append(row["category"])

X = np.array(images)
y = np.array(labels)

# split train/val if you want
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_val.shape)