from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# ローレンツ方程式の定義
def lorenz(t, state, sigma, beta, rho):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# パラメータと初期条件
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0
initial_state = [1.0, 1.0, 1.0]  # 初期条件
t_span = (0, 50)  # 時間範囲
t_eval = np.linspace(t_span[0], t_span[1], 10000)  # 評価時間点

# 微分方程式を解く
sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, args=(sigma, beta, rho))

# データを取得
x_data = sol.y[0]

# シャノンエントロピーを計算
def compute_entropy(data, bins=50):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    probabilities = hist / np.sum(hist)
    probabilities = probabilities[probabilities > 0]  # 0を除外
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

entropy = compute_entropy(x_data)

# 結果をプロット
plt.figure(figsize=(10, 6))
plt.plot(t_eval, x_data, label="x (Lorenz System)", color="blue")
plt.title(f"Lorenz System and Entropy (H = {entropy:.4f} bits)", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel("x", fontsize=12)
plt.grid(alpha=0.5)
plt.legend()
plt.show()

# エントロピーの結果を出力
print(f"Calculated Shannon Entropy: {entropy:.4f} bits")