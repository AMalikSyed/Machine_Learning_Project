# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, precision_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
df = pd.read_csv("Churn Modeling.csv")

# Print initial data information
print("Initial shape of data:", df.shape)
print("Columns:", df.columns.tolist())

# Remove unnecessary columns that don't contribute to prediction
drop_columns = ["RowNumber", "CustomerId", "Surname"]
df.drop(columns=drop_columns, inplace=True, errors='ignore')

# Check and handle missing values
print("Missing values:\n", df.isnull().sum())
df.dropna(inplace=True)
print("Shape after dropping missing values:", df.shape)

# Verify target column exists
if 'Exited' not in df.columns:
    raise ValueError("Target column 'Exited' not found in the dataset.")

# Identify and one-hot encode categorical columns
cat_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'Exited']
print("Categorical columns:", cat_cols)
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Identify numerical columns for scaling
num_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'Exited']
print("Numerical columns:", num_cols)

# Split data into features (X) and target (y)
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training set shape:", X_train.shape, "Test set shape:", X_test.shape)

# Scale numerical features
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Function to find optimal threshold for classification based on target precision
def find_threshold(probs, y_true, target_precision=0.70):
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    
    # Find thresholds that meet target precision
    candidate_thresholds = []
    for prec, rec, thr in zip(precisions[:-1], recalls[:-1], thresholds):
        preds = (probs >= thr).astype(int)
        if preds.sum() > 0 and prec >= target_precision:
            f1 = 2 * (prec * rec) / (prec + rec)
            candidate_thresholds.append((thr, prec, rec, f1))
    
    # Select best threshold based on F1 score if candidates exist
    if candidate_thresholds:
        best_threshold, best_prec, best_rec, best_f1 = max(candidate_thresholds, key=lambda x: x[3])
        print(f"Selected threshold: {best_threshold:.3f} (Precision: {best_prec:.3f}, Recall: {best_rec:.3f}, F1: {best_f1:.3f})")
        return best_threshold
    else:
        # Fallback to best precision if no threshold meets target
        all_candidates = []
        for prec, rec, thr in zip(precisions[:-1], recalls[:-1], thresholds):
            preds = (probs >= thr).astype(int)
            if preds.sum() > 0:
                all_candidates.append((thr, prec, rec))
        
        if all_candidates:
            best_threshold, best_prec, best_rec = max(all_candidates, key=lambda x: x[1])
            print(f"No threshold met target precision; selected threshold: {best_threshold:.3f} (Precision: {best_prec:.3f}, Recall: {best_rec:.3f})")
            return best_threshold
        else:
            print("No positive predictions found at any threshold; defaulting to 0.5")
            return 0.5

# Train and evaluate Random Forest model
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

# Perform grid search for best hyperparameters
grid_rf = GridSearchCV(
    rf,
    param_grid=param_grid_rf,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print("Best RF Params:", grid_rf.best_params_)

# Make predictions and adjust threshold
probs_rf = best_rf.predict_proba(X_test)[:, 1]
thr_rf = find_threshold(probs_rf, y_test, target_precision=0.70)
y_pred_rf_adj = (probs_rf >= thr_rf).astype(int)
print("Adjusted Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf_adj))

# Train and evaluate XGBoost model
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_clf.fit(X_train, y_train)

# Make predictions and adjust threshold
probs_xgb = xgb_clf.predict_proba(X_test)[:, 1]
thr_xgb = find_threshold(probs_xgb, y_test, target_precision=0.70)
y_pred_xgb_adj = (probs_xgb >= thr_xgb).astype(int)
print("Adjusted XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb_adj))

# Build and train Deep Neural Network
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=1
)

# Make predictions and adjust threshold
probs_dnn = model.predict(X_test).flatten()
thr_dnn = find_threshold(probs_dnn, y_test, target_precision=0.70)
y_pred_dnn_adj = (probs_dnn >= thr_dnn).astype(int)
print("Adjusted DNN Classification Report:")
print(classification_report(y_test, y_pred_dnn_adj))

# Plot training history
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('DNN Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('DNN Loss')
plt.legend()
plt.show()