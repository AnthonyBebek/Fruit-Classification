import kagglehub
import pandas as pd
import numpy as np
import joblib
import os

# Get the path to the dataset
path = str(kagglehub.dataset_download("moltean/fruits"))
print("Path to dataset files:", path)


train_csv = os.path.join(os.getcwd(), 'fruits_training.csv')
test_csv = os.path.join(os.getcwd(), 'fruits_test.csv')

# Load them into DataFrames
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Check the first few rows
print(train_df.head())
print(test_df.head())

from sklearn.metrics import accuracy_score, classification_report
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import LabelEncoder
from functools import partial
from PIL import Image
import os

size = (64, 64)

def process_image(row, base_path, crop_size):
    image_path = os.path.join(base_path, row['image'])
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize(crop_size)  # use the tuple directly
    img_array = np.array(img) / 255.0

    label = row['label']
    return img_array, label



def load_parallel(csv_df, base_path, crop_size=size, n_jobs=30):
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        func = partial(process_image, base_path=base_path, crop_size=crop_size)
        results = list(executor.map(func, [row for _, row in csv_df.iterrows()]))

    features, labels = zip(*results)
    return np.array(features), np.array(labels)


# usage
base_path_train = os.path.join(path, "fruits-360_100x100/fruits-360/Training")
X_train, y_train = load_parallel(train_df, base_path_train)

base_path_val = os.path.join(path, "fruits-360_100x100/fruits-360/Test")
X_val, y_val = load_parallel(test_df, base_path_val)

print("Train:", X_train.shape, "Val:", X_val.shape)
print("Unique train labels:", np.unique(y_train))
print("Unique val labels:", np.unique(y_val))

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat   = X_val.reshape(X_val.shape[0], -1)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)

print(y_train_enc)
print(y_val_enc)


import joblib

xgb_clf = joblib.load("xgb_model.joblib")
y_pred = xgb_clf.predict(X_val_flat)
acc = accuracy_score(y_val_enc, y_pred)
print(f"Validation accuracy: {acc:.4f}")
print(classification_report(
    y_val_enc,
    y_pred,
    target_names=[str(c) for c in le.classes_]
))

dt_clf = joblib.load("dt_model.joblib")
y_pred = dt_clf.predict(X_val_flat)
acc = accuracy_score(y_val_enc, y_pred)
print(f"Validation accuracy: {acc:.4f}")
print(classification_report(
    y_val_enc,
    y_pred,
    target_names=[str(c) for c in le.classes_]
))

lr_clf = joblib.load("lr_model.joblib")
y_pred = lr_clf.predict(X_val_flat)
acc = accuracy_score(y_val_enc, y_pred)
print(f"Validation accuracy: {acc:.4f}")
print(classification_report(
    y_val_enc,
    y_pred,
    target_names=[str(c) for c in le.classes_]
))

knn_clf = joblib.load("knn_model.joblib")
y_pred = knn_clf.predict(X_val_flat)
acc = accuracy_score(y_val_enc, y_pred)
print(f"Validation accuracy: {acc:.4f}")
print(classification_report(
    y_val_enc,
    y_pred,
    target_names=[str(c) for c in le.classes_]
))

rf_clf = joblib.load("rf_model.joblib")
y_pred = rf_clf.predict(X_val_flat)
acc = accuracy_score(y_val_enc, y_pred)
print(f"Validation accuracy: {acc:.4f}")
print(classification_report(
    y_val_enc,
    y_pred,
    target_names=[str(c) for c in le.classes_]
))


