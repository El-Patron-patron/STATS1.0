import pandas as pd
import numpy as np

# Read file
df = pd.read_csv(r"C:\Users\Hanz\Downloads\pinakapinaka bago.csv")

# Actual target variable
y = df["Sleep Duration"].values

# Predictors
X = df[
    [
        "Physical Activity Level",
        "Age",
        "Stress Level",
        "Daily Steps",
        "Disorder_Insomnia",
        "Disorder_SleepApnea"
    ]
].values

# Add intercept
X_int = np.column_stack([np.ones(len(X)), X])

# -----------------------------------------------------
# Train-test split (80% train, 20% test)
# -----------------------------------------------------
n = len(df)
cut = int(n * 0.8)   # 80% cutoff index

X_train = X_int[:cut]
y_train = y[:cut]

X_test = X_int[cut:]
y_test = y[cut:]

print("Training size:", len(X_train))
print("Testing size:", len(X_test))
print("Training %:", round(len(X_train) / n * 100, 2))
print("Testing %:", round(len(X_test) / n * 100, 2))

# Fit linear regression
beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

# Predictions for test set
y_pred = X_test @ beta

# -----------------------------------------
# OVERALL TEST SET PERFORMANCE
# -----------------------------------------
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
nrmse = (rmse / (y_test.max() - y_test.min())) * 100

print("\n=== OVERALL TEST SET PERFORMANCE ===")
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("NRMSE (%):", round(nrmse, 2))

# -----------------------------------------
# BOTTOM 20% OF TEST SET
# -----------------------------------------
bottom_count = int(len(y_test) * 0.20)

sorted_idx = np.argsort(y_test)
idx_bottom20 = sorted_idx[:bottom_count]

y_test_bottom = y_test[idx_bottom20]
y_pred_bottom = y_pred[idx_bottom20]

mse_b20 = np.mean((y_test_bottom - y_pred_bottom) ** 2)
rmse_b20 = np.sqrt(mse_b20)
nrmse_b20 = (rmse_b20 / y_test_bottom.mean()) * 100

print("\n=== BOTTOM 20% TEST PERFORMANCE ===")
print("Bottom 20% MSE:", round(mse_b20, 4))
print("Bottom 20% RMSE:", round(rmse_b20, 4))
print("Bottom 20% NRMSE (%):", round(nrmse_b20, 2))
