import matplotlib.pyplot as plt
import pandas as pd

# CSVファイルの読み込み
data = pd.read_csv('narma10_data.csv')

# 時間、入力、出力を取得
time = data['Time']
input_signal = data['Input']
output_signal = data['Output']

# プロット
plt.figure(figsize=(12, 6))

# 入力信号のプロット
plt.subplot(2, 1, 1)
plt.plot(time, input_signal, label='Input Signal', color='blue')
plt.title('Input Signal')
plt.xlabel('Time')
plt.ylabel('Input')
plt.grid(True)
plt.legend()

# 出力信号（NARMA10）のプロット
plt.subplot(2, 1, 2)
plt.plot(time, output_signal, label='NARMA10 Output', color='green')
plt.title('NARMA10 Output')
plt.xlabel('Time')
plt.ylabel('Output')
plt.grid(True)
plt.legend()

# プロットの表示
plt.tight_layout()
plt.show()
