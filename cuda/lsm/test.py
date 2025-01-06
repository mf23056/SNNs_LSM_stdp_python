import numpy as np
import pandas as pd

# CSVファイルのパスを指定
csv_file_path = '../random_netwrok/weights.csv'

# CSVファイルを読み込む
df = pd.read_csv(csv_file_path, header=None)

# CSVの全ての値を合計
weight_sum = df.to_numpy().sum()

# 合計値を表示
print(f'The sum of the weights is: {weight_sum}')