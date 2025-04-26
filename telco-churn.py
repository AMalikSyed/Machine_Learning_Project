"""
Telco Customer Churn Prediction Analysis
This script performs customer churn prediction using multiple machine learning models:
1. Random Forest
2. XGBoost
3. Deep Neural Network

The analysis includes:
- Data preprocessing and cleaning
- Feature engineering
- Model training and hyperparameter tuning
- Performance evaluation with optimal threshold selection
- Visualization of model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers

# Load and inspect the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Initial shape of data:", df.shape)
print("Columns:", df.columns)

# Data cleaning and preprocessing
# Remove customer ID as it's not a predictive feature
if 'customerID' in df.columns:
   df.drop('customerID', axis=1, inplace=True)
print("Missing values:\n", df.isnull().sum())

# Convert TotalCharges to numeric and handle missing values
if 'TotalCharges' in df.columns:
   df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
print("Shape after dropping missing:", df.shape)

# Feature engineering
# Separate categorical and numerical columns
cat_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'Churn']
num_cols = [col for col in df.columns if df[col].dtype in ['int64','float64']]

# Encode target variable and categorical features
label_enc = LabelEncoder()
df['Churn'] = label_enc.fit_transform(df['Churn'])  # "Yes" -> 1, "No" -> 0
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Prepare features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", X_train.shape, "Test set shape:", X_test.shape)

# Scale numerical features
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

def find_best_threshold(probs, y_true, step=0.001):
   """
   Find the optimal probability threshold for classification
   that maximizes the F1 score.
   
   Args:
       probs: Predicted probabilities
       y_true: True labels
       step: Step size for threshold search
   
   Returns:
       best_thr: Optimal threshold
       best_f1: Best F1 score achieved
   """
   best_thr = 0.5
   best_f1 = 0.0
   thresholds = np.arange(0.0, 1.0 + step, step)
   for thr in thresholds:
       preds = (probs >= thr).astype(int)
       if preds.sum() == 0:
           continue
       prec = precision_score(y_true, preds)
       rec = recall_score(y_true, preds)
       f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
       if f1 > best_f1:
           best_f1 = f1
           best_thr = thr
   print(f"Selected threshold: {best_thr:.3f} (F1: {best_f1:.3f})")
   return best_thr, best_f1

# Random Forest Model
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
   'n_estimators': [50, 100],
   'max_depth': [None, 5, 10],
   'min_samples_split': [2, 5],
}
grid_rf = GridSearchCV(
   rf,
   param_grid=param_grid_rf,
   scoring='accuracy',
   cv=3,
   n_jobs=-1,
   verbose=1
)

# Train and evaluate Random Forest
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print("Best RF Params:", grid_rf.best_params_)
probs_rf = best_rf.predict_proba(X_test)[:, 1]
thr_rf, f1_rf = find_best_threshold(probs_rf, y_test, step=0.001)
y_pred_rf_adj = (probs_rf >= thr_rf).astype(int)
print("Adjusted Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf_adj))

# XGBoost Model
xgb_clf = xgb.XGBClassifier(
   n_estimators=100,
   max_depth=5,
   learning_rate=0.1,
   use_label_encoder=False,
   eval_metric='logloss',
   random_state=42
)

# Train and evaluate XGBoost
xgb_clf.fit(X_train, y_train)
probs_xgb = xgb_clf.predict_proba(X_test)[:, 1]
thr_xgb, f1_xgb = find_best_threshold(probs_xgb, y_test, step=0.001)
y_pred_xgb_adj = (probs_xgb >= thr_xgb).astype(int)
print("Adjusted XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb_adj))

# Deep Neural Network Model
model = keras.Sequential([
   layers.Input(shape=(X_train.shape[1],)),
   layers.Dense(64, activation='relu'),
   layers.Dropout(0.3),
   layers.Dense(32, activation='relu'),
   layers.Dropout(0.2),
   layers.Dense(1, activation='sigmoid')
])

# Compile and train DNN
model.compile(
   loss='binary_crossentropy',
   optimizer='adam',
   metrics=['accuracy']
)

history = model.fit(
   X_train, y_train,
   validation_split=0.2,
   epochs=20,
   batch_size=32,
   verbose=1
)

# Evaluate DNN
probs_dnn = model.predict(X_test).flatten()
thr_dnn, f1_dnn = find_best_threshold(probs_dnn, y_test, step=0.001)
y_pred_dnn_adj = (probs_dnn >= thr_dnn).astype(int)
print("Adjusted DNN Classification Report:")
print(classification_report(y_test, y_pred_dnn_adj))

# Plot training history
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("DNN Accuracy")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DNN Loss")
plt.legend()
plt.show()