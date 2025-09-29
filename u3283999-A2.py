import kagglehub
import pandas as pd
import numpy as np
import joblib
import os

# Get the path to the dataset
path = str(kagglehub.dataset_download("moltean/fruits"))
print("Path to dataset files:", path)

def create_csv(dataset_type):
    dataset_path = f'{path}/fruits-360_100x100/fruits-360/{dataset_type}'
    image_paths = []
    labels = []

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                if image_name.endswith('.jpg'):
                    image_paths.append(os.path.join(label, image_name))
                    labels.append(label)

    df = pd.DataFrame({
        'image': image_paths,
        'label': labels
    })

    csv_filename = f'fruits_{dataset_type.lower()}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"{dataset_type} CSV saved at: {os.path.join(os.getcwd(), csv_filename)}")

# Create CSVs for training and testing datasets
create_csv('Training')
create_csv('Test')

train_csv = os.path.join(os.getcwd(), 'fruits_training.csv')
test_csv = os.path.join(os.getcwd(), 'fruits_test.csv')

# Load them into DataFrames
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Check the first few rows
print(train_df.head())
print(test_df.head())

def limit_classes(df, num_classes=50, samples_per_class=None, random_state=42):
    # Pick a subset of classes
    unique_classes = df["label"].unique()
    np.random.seed(random_state)
    selected_classes = np.random.choice(unique_classes, num_classes, replace=False)

    # Filter down to those classes
    df = df[df["label"].isin(selected_classes)]

    # Optionally, sample equal images per class
    if samples_per_class:
        df = df.groupby("label").apply(
            lambda x: x.sample(n=min(samples_per_class, len(x)), random_state=random_state)
        ).reset_index(drop=True)

    return df




imgPerClass = 50

#train_df = limit_classes(train_df, samples_per_class=imgPerClass)
#test_df = limit_classes(test_df, samples_per_class=imgPerClass)

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

import matplotlib.pyplot as plt
import random
from PIL import Image
import os

# Pick 5 random rows
sample_rows = train_df.sample(5, random_state=42)

# Set base path
base_path_train = os.path.join(path, "fruits-360_100x100/fruits-360/Training")

# Create figure
plt.figure(figsize=(15, 5))

for i, (_, row) in enumerate(sample_rows.iterrows(), 1):
    img_path = os.path.join(base_path_train, row['image'])
    img = Image.open(img_path).convert('RGB')
    
    plt.subplot(1, 5, i)
    plt.imshow(img)
    plt.title(row['label'])
    plt.axis('off')

plt.show()


import xgboost as xgb

xgb_clf = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=len(np.unique(y_train_enc)),
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=1,
    eval_metric="mlogloss",
)

xgb_clf.fit(
    X_train_flat, y_train_enc,
    eval_set=[(X_val_flat, y_val_enc)],
    verbose=True
)

joblib.dump(xgb_clf, "xgb_model.joblib")

xgb_clf = joblib.load("xgb_model.joblib")
y_pred = xgb_clf.predict(X_val_flat)
acc = accuracy_score(y_val_enc, y_pred)
print(f"Validation accuracy: {acc:.4f}")
print(classification_report(
    y_val_enc,
    y_pred,
    target_names=[str(c) for c in le.classes_]
))

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(max_depth=20, 
                                min_samples_leaf=2,
                                random_state=42)
dt_clf.fit(X_train_flat, y_train_enc)

y_pred_dt = dt_clf.predict(X_val_flat)
joblib.dump(dt_clf, "dt_model.joblib")
print("Decision Tree Accuracy:", accuracy_score(y_val_enc, y_pred_dt))
print(classification_report(y_val_enc, y_pred_dt, target_names=le.classes_))

from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=500, multi_class='multinomial', solver='saga', n_jobs=-1)
lr_clf.fit(X_train_flat, y_train_enc)

y_pred_lr = lr_clf.predict(X_val_flat)
joblib.dump(lr_clf, "lr_model.joblib")
print("Logistic Regression Accuracy:", accuracy_score(y_val_enc, y_pred_lr))
print(classification_report(y_val_enc, y_pred_lr, target_names=le.classes_))


from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_clf.fit(X_train_flat, y_train_enc)

y_pred_knn = knn_clf.predict(X_val_flat)
joblib.dump(knn_clf, "knn_model.joblib")
print("kNN Accuracy:", accuracy_score(y_val_enc, y_pred_knn))
print(classification_report(y_val_enc, y_pred_knn, target_names=le.classes_))


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)
rf_clf.fit(X_train_flat, y_train_enc)

y_pred_rf = rf_clf.predict(X_val_flat)
joblib.dump(rf_clf, "rf_model.joblib")
print("Random Forest Accuracy:", accuracy_score(y_val_enc, y_pred_rf))
print(classification_report(y_val_enc, y_pred_rf, target_names=le.classes_))


from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# Flatten images: (n_samples, 64, 64, 3) -> (n_samples, 12288)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# Reduce dimensionality with PCA for speed and performance
pca = PCA(n_components=200, random_state=42)
X_train_pca = pca.fit_transform(X_train_flat)
X_val_pca = pca.transform(X_val_flat)

# Train KNN on reduced features
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)

y_pred = knn.predict(X_val_pca)

# Train kNN model
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)  # try k=5 first
knn.fit(X_train_pca, y_train_enc)
joblib.dump(knn, "knn.joblib")

# Predict
y_pred_knn = knn.predict(X_val_pca)

# Evaluate
acc = accuracy_score(y_val_enc, y_pred_knn)
print("kNN Accuracy:", acc)
#print(classification_report(y_val_enc, y_pred_knn, target_names=label_encoder.classes_))
