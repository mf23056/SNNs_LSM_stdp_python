import numpy as np
import pandas as pd

# パラメータ設定
dt = 0.01
steps = 30000
input_range = [0, 0.5]  # 入力の範囲

# 入力信号を生成
np.random.seed(42)  # 再現性のため
u = np.random.uniform(input_range[0], input_range[1], steps)

# NARMA10を計算
def narma10(input_signal):
    n = len(input_signal)
    y = np.zeros(n)
    for t in range(10, n):
        y[t] = (
            0.3 * y[t-1] +
            0.05 * y[t-1] * np.sum(y[t-10:t]) +
            1.5 * input_signal[t-10] * input_signal[t-1] +
            0.1
        )
    return y

# NARMA10データを生成
narma_output = narma10(u)

# 時間軸を作成
time = np.arange(0, steps * dt, dt)

# データを保存
data = pd.DataFrame({'Time': time, 'Input': u, 'Output': narma_output})
data.to_csv('narma10_data.csv', index=False)

print("NARMA10データを'narma10_data.csv'に保存しました。")
