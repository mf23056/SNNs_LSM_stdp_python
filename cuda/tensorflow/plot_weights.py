import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# スパイクトレインのCSVファイルを読み込む
# ヘッダーなし、1列でスパイク時間が記録されていると仮定
spike_train = pd.read_csv('../lsm/spike_bin_blance.csv', header=None)


# 行数と列数を表示
print(f'行数: {spike_train.shape[0]}, 列数: {spike_train.shape[1]}')

# 最初の数行を表示して形を確認
print(spike_train.head()) 

# ラスタープロットを作成
plt.figure(figsize=(10, 6))

# 各ニューロンごとにスパイクのタイミングをプロット
for neuron in range(spike_train.shape[0]):
    spike_times = spike_train.iloc[neuron].to_numpy().nonzero()[0]  # スパイクのタイムステップのインデックス
    plt.plot(spike_times, np.ones_like(spike_times) * neuron, 'o', color='black', markersize=2)

# 軸ラベル
plt.xlabel('Time step')
plt.ylabel('Neuron index')
plt.title('Raster Plot of Spike Train')

# グラフの表示
plt.tight_layout()
plt.show()