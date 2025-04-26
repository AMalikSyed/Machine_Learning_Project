import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load the Dataset


df = pd.read_csv("BankChurners.csv")
print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Drop Non-Predictive & Leakage Columns


# Drop customer ID and any potential leakage columns like 'Naive_Bayes_Classification'

for col in ['CLIENTNUM', 'Naive_Bayes_Classification']:

    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# 3. Data Cleaning & Missing Values


print("Missing values:\n", df.isnull().sum())

df.dropna(inplace=True)

print("Shape after dropping missing values:", df.shape)

# 4. Encode Target and Categorical Features


# Convert target 'Attrition_Flag' to binary: 0 for "Existing Customer", 1 for "Attrited Customer"

df['Attrition_Flag'] = df['Attrition_Flag'].apply(lambda x: 1 if x.strip() == 'Attrited Customer' else 0)

# Identify categorical (excluding target) and numerical columns

categorical_columns = [col for col in df.columns if df[col].dtype == 'object' and col != 'Attrition_Flag']

numerical_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'Attrition_Flag']

# One-hot encode categorical features

df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# 5. Train/Test Split and Scaling


X = df.drop('Attrition_Flag', axis=1)

y = df['Attrition_Flag']

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42, stratify=y

)

print("Training set shape:", X_train.shape, "Test set shape:", X_test.shape)

# Scale only the numerical columns that remain after encoding

scaler = StandardScaler()

num_cols = [col for col in numerical_columns if col in X_train.columns]

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

X_test[num_cols] = scaler.transform(X_test[num_cols])


# 6. Helper Function for Threshold Selection


def find_threshold(probs, y_true, target_precision=0.70):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)

    differences = np.abs(precisions[:-1] - target_precision)

    best_idx = np.argmin(differences)

    best_threshold = thresholds[best_idx]

    print(
        f"For target precision {target_precision}, selected threshold {best_threshold:.3f} (precision: {precisions[best_idx]:.3f})")

    return best_threshold


# 7. Model 1 - Random Forest with GridSearchCV


rf = RandomForestClassifier(random_state=42)

param_grid_rf = {

    'n_estimators': [50, 100],

    'max_depth': [None, 5, 10],

    'min_samples_split': [2, 5]

}

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

# Get probability predictions and adjust threshold

probs_rf = best_rf.predict_proba(X_test)[:, 1]

thr_rf = find_threshold(probs_rf, y_test, target_precision=0.70)

y_pred_rf_adj = (probs_rf >= thr_rf).astype(int)

print("Adjusted Random Forest Classification Report:")

print(classification_report(y_test, y_pred_rf_adj))

# 8. Model 2 - XGBoost Classifier


xgb_clf = xgb.XGBClassifier(

    n_estimators=100,

    max_depth=5,
    learning_rate=0.1,

    use_label_encoder=False,

    eval_metric='logloss',

    random_state=42

)

xgb_clf.fit(X_train, y_train)

probs_xgb = xgb_clf.predict_proba(X_test)[:, 1]

thr_xgb = find_threshold(probs_xgb, y_test, target_precision=0.70)

y_pred_xgb_adj = (probs_xgb >= thr_xgb).astype(int)

print("Adjusted XGBoost Classification Report:")

print(classification_report(y_test, y_pred_xgb_adj))

# 9. Model 3 - Deep Neural Network with Keras

# Create a more robust DNN model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    layers.Dense(1, activation='sigmoid')
])

# Compile with appropriate metrics and optimizer
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# Add early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model with validation split
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Get probability predictions
probs_dnn = model.predict(X_test).flatten()

# Find optimal threshold
thr_dnn = find_threshold(probs_dnn, y_test, target_precision=0.70)
y_pred_dnn_adj = (probs_dnn >= thr_dnn).astype(int)

print("Adjusted DNN Classification Report:")
print(classification_report(y_test, y_pred_dnn_adj))

# 10. Plot Training History for DNN


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
