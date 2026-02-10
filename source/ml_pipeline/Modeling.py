import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib
import json
from datetime import datetime

df = pd.read_csv('D:/AIjourney/projects/Pride Ads Project/CSV/pride_ads_engineered_1.csv')
df = df.drop('Unnamed: 0', axis=1)

# ------------------------------- Pre-Processing -------------------------------
inputs = df[['name','trim','mileage','fuel','transmission','body_status','age']]
outputs = df[['price']]

print('Inputs shape:', inputs.shape)
print('Outputs shape:', outputs.shape)

x = inputs.values
y = outputs.values

inputScaler = StandardScaler()
inputScaler.fit(x)
x = inputScaler.transform(x)
print('Input data has been scaled.')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print('\nTrain/Test split took effect:')
print('X train shape:', x_train.shape)
print('X test shape:', x_test.shape)
print('Y train shape:', y_train.shape)
print('Y test shape:', y_test.shape)

# ------------------------------- Multiple Linear Regression -------------------------------
model_LR = LinearRegression()
model_LR.fit(x_train, y_train)

# Predict
pred_LR_train = model_LR.predict(x_train)
pred_LR = model_LR.predict(x_test)
# Evaluate
mae_LR = mean_absolute_error(y_test, pred_LR)
rmse_LR = np.sqrt(mean_squared_error(y_test, pred_LR))
r2_LR = r2_score(y_test, pred_LR)

print(f"MAE:  {mae_LR:,.0f} Toman")
print(f"RMSE: {rmse_LR:,.0f} Toman")
print(f"R²:   {r2_LR*100:,.2f}")

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_test/1e6, pred_LR/1e6, alpha=0.6)
plt.plot([y_test.min()/1e6, y_test.max()/1e6], 
         [y_test.min()/1e6, y_test.max()/1e6], 'r--', linewidth=2)
plt.xlabel('Actual Price (Million Toman)')
plt.ylabel('Predicted Price (Million Toman)')
plt.title('Linear Regression: Actual vs Predicted')
plt.tight_layout()
plt.show()

# ------------------------------- Random Forest -------------------------------
model_RF = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_RF.fit(x_train, y_train)

# Predict
pred_RF_train = model_RF.predict(x_train)
pred_RF = model_RF.predict(x_test)
# Evaluate
mae_RF = mean_absolute_error(y_test, pred_RF)
rmse_RF = np.sqrt(mean_squared_error(y_test, pred_RF))
r2_RF = r2_score(y_test, pred_RF)

print(f"MAE:  {mae_RF:,.0f} Toman")
print(f"RMSE: {rmse_RF:,.0f} Toman")
print(f"R²:   {r2_RF*100:.2f}")

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_test/1e6, pred_RF/1e6, alpha=0.6)
plt.plot([y_test.min()/1e6, y_test.max()/1e6], 
         [y_test.min()/1e6, y_test.max()/1e6], 'r--', linewidth=2)
plt.xlabel('Actual Price (Million Toman)')
plt.ylabel('Predicted Price (Million Toman)')
plt.title('Random Forest: Actual vs Predicted')
plt.tight_layout()
plt.show()

# Feature importance for Random Forest model
feature_importance = pd.DataFrame({
    'feature': inputs.columns,
    'importance': model_RF.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 important features (Random Forest):")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# ------------------------------- XGBoost -------------------------------
model_XGB = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
model_XGB.fit(x_train, y_train)

# Predict
pred_XGB_train = model_XGB.predict(x_train)
pred_XGB = model_XGB.predict(x_test)
# Evaluate
mae_XGB = mean_absolute_error(y_test, pred_XGB)
rmse_XGB = np.sqrt(mean_squared_error(y_test, pred_XGB))
r2_XGB = r2_score(y_test, pred_XGB)

print(f"MAE:  {mae_XGB:,.0f} Toman")
print(f"RMSE: {rmse_XGB:,.0f} Toman")
print(f"R²:   {r2_XGB*100:.2f}")

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_test/1e6, pred_XGB/1e6, alpha=0.6)
plt.plot([y_test.min()/1e6, y_test.max()/1e6], 
         [y_test.min()/1e6, y_test.max()/1e6], 'r--', linewidth=2)
plt.xlabel('Actual Price (Million Toman)')
plt.ylabel('Predicted Price (Million Toman)')
plt.title('XGBoost: Actual vs Predicted')
plt.tight_layout()
plt.show()

# ------------------------------- Neural Network -------------------------------
x = inputs.values
y = outputs.values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

inputScaler_2 = StandardScaler()
outputScaler = StandardScaler()

inputScaler_2.fit(x_train)
x_train_scaled = inputScaler_2.transform(x_train)
x_test_scaled = inputScaler_2.transform(x_test)

outputScaler.fit(y_train)
y_train_scaled = outputScaler.transform(y_train)
y_test_scaled = outputScaler.transform(y_test)

model_NN = keras.Sequential([
    layers.Input(shape=(x_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model_NN.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Smaller LR
    loss='mse',
    metrics=['mae']
)

early_stopping = callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True,
    monitor='val_loss'
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001
)

model_NN.summary()

print("Training neural network...")
history = model_NN.fit(
    x_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Predict
pred_NN_train = model_NN.predict(x_train_scaled, verbose=0).flatten()
pred_NN_scaled = model_NN.predict(x_test_scaled, verbose=0).flatten()
pred_NN = outputScaler.inverse_transform(pred_NN_scaled.reshape(-1, 1))

# Evaluate
mae_NN = mean_absolute_error(y_test, pred_NN)
rmse_NN = np.sqrt(mean_squared_error(y_test, pred_NN))
r2_NN = r2_score(y_test, pred_NN)

print(f"MAE:  {mae_NN:,.0f} Toman")
print(f"RMSE: {rmse_NN:,.0f} Toman")
print(f"R²:   {r2_NN*100:.2f}")

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_test/1e6, pred_NN/1e6, alpha=0.6)
plt.plot([y_test.min()/1e6, y_test.max()/1e6], 
         [y_test.min()/1e6, y_test.max()/1e6], 'r--', linewidth=2)
plt.xlabel('Actual Price (Million Toman)')
plt.ylabel('Predicted Price (Million Toman)')
plt.title('Neural Network: Actual vs Predicted')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training History - Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



# ------------------------------- Performance Comparison -------------------------------
comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'Neural Network'],
    'MAE (Million Toman)': [mae_LR/1e6, mae_RF/1e6, mae_XGB/1e6, mae_NN/1e6],
    'RMSE (Million Toman)': [rmse_LR/1e6, rmse_RF/1e6, rmse_XGB/1e6, rmse_NN/1e6],
    'R² Score': [r2_LR, r2_RF, r2_XGB, r2_NN],
    'Training R²': [
        r2_score(y_train, pred_LR_train),
        r2_score(y_train, pred_RF_train),
        r2_score(y_train, pred_XGB_train),
        r2_score(y_train_scaled, pred_NN_train)
    ]
})

comparison_df = comparison_df.sort_values('R² Score', ascending=False)

print(comparison_df.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# R2 score comparison
models = comparison_df['Model']
r2_scores = comparison_df['R² Score']
bars1 = axes[0].bar(models, r2_scores)
axes[0].set_ylabel('R² Score')
axes[0].set_title('Model Performance (R²)')
axes[0].tick_params(axis='x', rotation=45)
for bar, score in zip(bars1, r2_scores):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')

# MAE Comparison
mae_scores = comparison_df['MAE (Million Toman)']
bars2 = axes[1].bar(models, mae_scores, color='orange')
axes[1].set_ylabel('MAE (Million Toman)')
axes[1].set_title('Mean Absolute Error')
axes[1].tick_params(axis='x', rotation=45)
for bar, mae in zip(bars2, mae_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mae:.1f}', ha='center', va='bottom')

# Overfitting Check
train_r2 = comparison_df['Training R²']
test_r2 = comparison_df['R² Score']
x = np.arange(len(models))
width = 0.35
axes[2].bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8)
axes[2].bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(models, rotation=45, ha='right')
axes[2].set_ylabel('R² Score')
axes[2].set_title('Overfitting Check (Train vs Test)')
axes[2].legend()
axes[2].axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.show()

# Select best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = {
    'Linear Regression': model_LR,
    'Random Forest': model_RF,
    'XGBoost': model_XGB,
    'Neural Network': model_NN
}[best_model_name]

best_r2 = comparison_df.iloc[0]['R² Score']
best_mae = comparison_df.iloc[0]['MAE (Million Toman)']

print("\n" + "=" * 60)
print(f"🏆 BEST MODEL: {best_model_name}")
print("=" * 60)
print(f"R² Score: {best_r2:.4f}")
print(f"MAE: {best_mae:.1f} Million Toman")
print(f"Average error: {best_mae/(y_test.mean()/1e6)*100:.1f}% of average car price")

print("\n✅ All models trained and evaluated successfully!")
print(f"   Best model saved as variable: best_model")
print(f"   Predictions saved as: pred_LR, pred_RF, pred_XGB, pred_NN")

# ------------------------------- Saving models and scalers -------------------------------
joblib.dump(inputScaler, 'D:/AIjourney/projects/Pride Ads Project/Models/v1/input_scaler.pkl')
print("Inputs scaler saved as 'input_scaler.pkl'")

joblib.dump(model_RF, 'D:/AIjourney/projects/Pride Ads Project/Models/v1/random_forest_model.pkl')
print("Random Forest model saved as 'random_forest_model.pkl'")

metadata = {
    'model_type': 'RandomForestRegressor',
    'model_name': 'Random Forest',
    'performance': {
        'r2_score': float(r2_RF),
        'mae': float(mae_RF),
        'mae_million_toman': float(mae_RF / 1e6),
        'rmse': float(rmse_RF)
    },
    'training_info': {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_samples': x_train.shape[0],
        'test_samples': x_test.shape[0],
        'feature_count': x_train.shape[1]
    },
    'features': {
        'column_names': list(inputs.columns.tolist()),
        'column_order': list(inputs.columns.tolist())
    },
    'target': {
        'name': 'price',
        'average_value': float(y_train.mean()),
        'range': [float(y_train.min()), float(y_train.max())]
    },
    'model_parameters': {
        'n_estimators': model_RF.n_estimators,
        'random_state': model_RF.random_state
    }
}

with open('D:/AIjourney/projects/Pride Ads Project/Models/v1/model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print("Metadata saved as 'model_metadata.json'")

feature_importance_df = pd.DataFrame({
    'feature': inputs.columns,
    'importance': model_RF.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance_df.to_csv('D:/AIjourney/projects/Pride Ads Project/Models/v1/feature_importance.csv',
 index=False,
 encoding='utf-8-sig')
print("Feature importance saved as 'feature_importance.csv'")
