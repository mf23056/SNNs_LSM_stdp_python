import numpy as np
import matplotlib.pyplot as plt

# NARMA10モデルを定義
def narma10(y, u, t):
    if t < 10:
        return 0  # t < 10 の場合は定義できないので 0
    return (0.3 * y[t-1] +
            0.05 * y[t-1] * np.sum(y[t-10:t]) +
            1.5 * u[t-10] * u[t-1] +
            0.1)

# シミュレーション関数
def simulate_narma10(u, T):
    y = np.zeros(T)
    for t in range(T):
        y[t] = narma10(y, u, t)
    return y

# Lyapunov指数計算用
def calculate_lyapunov_narma10(u, T, delta=1e-6):
    # 初期条件を少しずつずらした2つのシステムを計算
    y1 = np.zeros(T)
    y2 = np.zeros(T)
    y1[0] = 0.5
    y2[0] = 0.5 + delta
    
    lyapunov_sum = 0
    for t in range(1, T):
        y1[t] = narma10(y1, u, t)
        y2[t] = narma10(y2, u, t)
        
        # 差分とログを計算
        diff = abs(y2[t] - y1[t])
        if diff > 0:  # ゼロ除算を回避
            lyapunov_sum += np.log(diff / delta)
        y2[t] = y1[t] + delta * (y2[t] - y1[t]) / diff  # 再スケーリング
    
    return lyapunov_sum / (T - 1)

# シミュレーションパラメータ
T = 1000  # 時間ステップ数
u = np.random.uniform(0, 1, T)  # 入力信号

# NARMA10のシミュレーション
y = simulate_narma10(u, T)

# Lyapunov指数の計算
lyapunov = calculate_lyapunov_narma10(u, T)

# 結果表示
print(f"Lyapunov指数: {lyapunov}")

# 時系列データのプロット
plt.figure(figsize=(10, 6))
plt.plot(y, label="NARMA10 Output")
plt.title("NARMA10 Time Series")
plt.xlabel("Time Step")
plt.ylabel("Output")
plt.grid()
plt.legend()
plt.show()
