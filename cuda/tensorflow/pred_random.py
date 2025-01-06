import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. LSMのスパイクデータをCSVから読み込む
spike_data = pd.read_csv('../lsm/spike_train_random.csv', header=None)  # スパイクデータ (時間 vs ニューロン)
target_data = pd.read_csv('../NARMA10/narma10_data.csv', header=0)  # NARMAモデルの出力（目標ラベル）

# 2. 目標データからOutput列を取り出す
y = target_data['Output'].to_numpy()

# スパイクデータを転置（ニューロン×時間の形式に変更）
spike_data = spike_data.T

# 特徴量とターゲットを分割
X_train, X_test, y_train, y_test = train_test_split(spike_data, y, test_size=0.2, random_state=42)

# 特徴量（X）と目標ラベル（y）の標準化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)  # 訓練データを基準にスケーリング
X_test_scaled = scaler_X.transform(X_test)  # テストデータも同じスケーラーでスケーリング

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))  # 訓練データの目標ラベルをスケーリング
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))  # テストデータの目標ラベルをスケーリング

# リッジ回帰モデルの訓練
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train_scaled)

# テストデータで予測
y_pred_scaled = ridge_model.predict(X_test_scaled)

# MSEを表示
mse = mean_squared_error(y_test_scaled, y_pred_scaled)
print(f"Mean Squared Error (MSE): {mse}")

# 実測値と予測値の比較（スケールを戻してから表示）
y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))  # 元のスケールに戻す
y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))  # 元のスケールに戻す

# 実測値と予測値を比較
plt.figure(figsize=(10, 6))
plt.plot(y_test_original[:100], label='Actual', marker='o', linestyle='--')
plt.plot(y_pred_original[:100], label='Predicted', marker='x', linestyle='-')
plt.legend()
plt.title("Actual vs Predicted (Sample of 100 points)")
plt.xlabel("Sample Index")
plt.ylabel("Output Value")
plt.grid()
plt.show()

# 回帰係数の可視化
plt.figure(figsize=(10, 6))
plt.bar(range(len(ridge_model.coef_)), ridge_model.coef_)
plt.title("Ridge Regression Coefficients")
plt.xlabel("Neuron Index")
plt.ylabel("Coefficient Value")
plt.show()