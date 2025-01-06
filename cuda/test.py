import torch

# サンプルデータ (4x4 行列)
delta_w = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0]])

# スパイク状態 (4 要素)
spike_state = torch.tensor([1, 0, 1, 0])  # 各行・列のスパイク状態

# 行と列のマスクを作成
row_mask = spike_state != 1
col_mask = spike_state != 1

# 行列マスクを作成 (4x4 マスク)
matrix_mask = row_mask.unsqueeze(1) & col_mask.unsqueeze(0)

# delta_w をマスクでフィルタリング
delta_w[matrix_mask] = 0

# 結果を表示
print(delta_w)