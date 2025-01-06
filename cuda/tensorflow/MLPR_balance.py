import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. LSMのスパイクデータをCSVから読み込む
spike_data = pd.read_csv('../lsm/spikes_balance_50.csv', header=None)  # スパイクデータ (時間 vs ニューロン)
target_data = pd.read_csv('../NARMA10/narma10_data.csv', header=0)  # NARMAモデルの出力（目標ラベル）

# 2. 目標データからOutput列を取り出す
y = target_data['Output'].to_numpy()

# スパイクデータ (特徴量) の転置
spike_data = spike_data.T

# 3. データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(spike_data, y, test_size=0.2, random_state=42)

# 4. 特徴量を標準化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 目標データを標準化
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# 5. MLP回帰モデルの訓練
mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp_model.fit(X_train_scaled, y_train_scaled.ravel())

# 6. テストデータで予測
y_pred_scaled = mlp_model.predict(X_test_scaled)

# 7. 予測を元のスケールに戻す
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# 8. モデルの性能評価 (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# 9. 実測値と予測値の比較（100点サンプルをプロット）
plt.figure(figsize=(10, 6))
plt.plot(y_test[:100], label='Actual', marker='o', linestyle='--')
plt.plot(y_pred[:100], label='Predicted (MLP)', marker='x', linestyle='-')
plt.legend()
plt.title("Actual vs Predicted (MLP) (Sample of 100 points)")
plt.xlabel("Sample Index")
plt.ylabel("Output Value")
plt.grid()
plt.show()

# 10. 回帰係数の可視化 (ニューラルネットワークでは直接の係数はないので、重みを可視化)
# ここでは隠れ層の重みを示します
plt.figure(figsize=(10, 6))
plt.imshow(mlp_model.coefs_[0], cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("MLP Weights (Layer 1)")
plt.xlabel("Neuron Index")
plt.ylabel("Input Feature Index")
plt.show()
