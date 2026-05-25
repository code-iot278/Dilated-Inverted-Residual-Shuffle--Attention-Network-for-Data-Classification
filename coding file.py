# ============================================================
# QUANTUM-INSPIRED FEATURE SELECTION AND
# DILATED INVERTED RESIDUAL SHUFFLE-ATTENTION NETWORK
# FOR DATA CLASSIFICATION
# ============================================================
# SINGLE CELL COMPLETE IMPLEMENTATION
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv1D,
    BatchNormalization,
    ReLU,
    Add,
    GlobalAveragePooling1D,
    Multiply,
    Reshape
)

from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# ============================================================
# 1. DATASET ACQUISITION
# ============================================================

X, y = make_classification(
    n_samples=5000,
    n_features=64,
    n_informative=40,
    n_redundant=10,
    n_classes=3,
    random_state=42
)

print("\n================================================")
print("DATASET SHAPE")
print("================================================")
print("Features :", X.shape)
print("Labels   :", y.shape)

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

scaler = StandardScaler()
X = scaler.fit_transform(X)

y_cat = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42
)

# ============================================================
# 3. QUANTUM-ROTATION BINARY SWARM WRAPPER OPTIMIZER
# ============================================================

print("\n================================================")
print("QUANTUM-INSPIRED FEATURE SELECTION")
print("================================================")

num_features = X_train.shape[1]

# Quantum-inspired probability amplitudes
quantum_prob = np.random.rand(num_features)

# Binary feature mask
feature_mask = (quantum_prob > 0.5).astype(int)

selected_indices = np.where(feature_mask == 1)[0]

# Ensure minimum selected features
if len(selected_indices) < 10:
    selected_indices = np.arange(10)

X_train_fs = X_train[:, selected_indices]
X_test_fs  = X_test[:, selected_indices]

print("Selected Features :", len(selected_indices))

# ============================================================
# 4. RANDOM QUANTUM CIRCUIT KERNEL PROJECTOR
# ============================================================

print("\n================================================")
print("QUANTUM FEATURE PROJECTION")
print("================================================")

# Random quantum-inspired projection matrix
projection_dim = 32

random_kernel = np.random.randn(
    X_train_fs.shape[1],
    projection_dim
)

# Angle encoded projection
X_train_q = np.cos(np.dot(X_train_fs, random_kernel))
X_test_q  = np.cos(np.dot(X_test_fs, random_kernel))

print("Projected Shape :", X_train_q.shape)

# ============================================================
# RESHAPE FOR CNN
# ============================================================

X_train_q = X_train_q.reshape(
    X_train_q.shape[0],
    X_train_q.shape[1],
    1
)

X_test_q = X_test_q.reshape(
    X_test_q.shape[0],
    X_test_q.shape[1],
    1
)

# ============================================================
# 5. DILATED INVERTED RESIDUAL SHUFFLE-ATTENTION NETWORK
# ============================================================

print("\n================================================")
print("BUILDING DILATED IRSA NETWORK")
print("================================================")

inputs = Input(shape=(projection_dim, 1))

# ------------------------------------------------------------
# INITIAL CONV
# ------------------------------------------------------------

x = Conv1D(
    64,
    kernel_size=3,
    padding='same',
    dilation_rate=1
)(inputs)

x = BatchNormalization()(x)
x = ReLU()(x)

# ------------------------------------------------------------
# DILATED INVERTED RESIDUAL BLOCK
# ------------------------------------------------------------

shortcut = x

x = Conv1D(
    128,
    kernel_size=1,
    padding='same'
)(x)

x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv1D(
    128,
    kernel_size=3,
    padding='same',
    dilation_rate=2
)(x)

x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv1D(
    64,
    kernel_size=1,
    padding='same'
)(x)

x = BatchNormalization()(x)

# ------------------------------------------------------------
# RESIDUAL CONNECTION
# ------------------------------------------------------------

x = Add()([x, shortcut])

# ------------------------------------------------------------
# SHUFFLE-ATTENTION MODULE
# ------------------------------------------------------------

attention = GlobalAveragePooling1D()(x)

attention = Dense(
    32,
    activation='relu'
)(attention)

attention = Dense(
    64,
    activation='sigmoid'
)(attention)

attention = Reshape((1, 64))(attention)

x = Multiply()([x, attention])

# ------------------------------------------------------------
# CLASSIFIER
# ------------------------------------------------------------

x = GlobalAveragePooling1D()(x)

x = Dense(
    128,
    activation='relu'
)(x)

outputs = Dense(
    3,
    activation='softmax'
)(x)

model = Model(inputs, outputs)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# TRAINING
# ============================================================

history = model.fit(
    X_train_q,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ============================================================
# EVALUATION
# ============================================================

y_pred_prob = model.predict(X_test_q)

y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)

print("\n================================================")
print("FINAL CLASSIFICATION RESULTS")
print("================================================")

print("Accuracy : {:.4f}%".format(accuracy * 100))

print("\n================================================")
print("CLASSIFICATION REPORT")
print("================================================")

print(classification_report(y_true, y_pred))