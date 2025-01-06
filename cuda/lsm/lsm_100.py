import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class LIF:
    def __init__(self, dt=0.01, tau_m=20, V_rest=-65.0, R=10.0, I_back=0.0, V_reset=-65.0, V_th=-50.0, V_ref=3):
        '''
        param dt: 刻み幅 [ms]
        param tau_m: 膜時間定数 [ms]
        param V_rest: 静止電位 [mV]
        param R: 抵抗 [MΩ]
        param I_back: バックグラウンド電流 [nA]
        param V_reset: リセット電位 [mV]
        param V_th: 閾値電位 [mV]
        param V_ref: 不応期 [ms]
        '''
        self.dt = dt
        self.tau_m = tau_m
        self.V_rest = V_rest
        self.R = R
        self.I_back = I_back
        self.V_reset = V_reset
        self.V_th = V_th
        self.V_ref_steps = int(V_ref / dt)

    def __call__(self, I_syn, before_V, ref_time):
        # 不応期のカウントダウン
        ref_time = torch.clamp(ref_time - 1, min=0)

        # 膜電位の更新
        V = torch.where(ref_time > 0, torch.full_like(before_V, self.V_reset),
                        before_V + self.dt * ((1 / self.tau_m) * (-(before_V - self.V_rest) + self.R * (I_syn + self.I_back))))

        # スパイク判定
        spiked = (V >= self.V_th).float()

        # スパイクが発生したニューロンの処理
        ref_time = torch.where(spiked > 0, self.V_ref_steps, ref_time)
        V = torch.where(spiked > 0, torch.full_like(V, self.V_reset), V)
        return spiked, V, ref_time


class StaticSynapse:
    def __init__(self, dt=0.01, tau_syn=25):
        self.dt = dt
        self.tau_syn = tau_syn

    def __call__(self, bin_spike, W, before_I):
        # bin_spike は1か0のバイナリスパイク列
        spikes = bin_spike.unsqueeze(1)  # 各スパイクをテンソルに合わせて繰り返す

        return before_I + self.dt * (-before_I / self.tau_syn) + W * spikes
    

class SNN:
    def __init__(self, n_exc, n_inh, dt=0.01, device='cuda', weight_file=None):
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh
        self.dt = dt
        self.device = device
        self.C = {"EE": 0.121, "EI": 0.169, "IE": 0.127, "II": 0.097}
        
        self.neuron = LIF()
        self.synapse = StaticSynapse()

        self._initialize_neurons(n_exc, n_inh)
        self._initialize_synapses(weight_file)

    def _initialize_neurons(self, n_exc, n_inh):
        self.sum_I_syn = torch.zeros(self.n_total, device=self.device)
        self.before_V = torch.full((self.n_total,), -65.0, device=self.device)
        self.ref_time = torch.zeros(self.n_total, device=self.device)
        self.spike_state = torch.zeros(self.n_total, device=self.device)

    def _initialize_synapses(self, weight_file):
        self.before_I = torch.zeros((self.n_total, self.n_total), device=self.device)
         # weights.csv を読み込む
        if weight_file:
            # CSVファイルの読み込み (n_total x n_totalの行列)
            weights_df = pd.read_csv(weight_file, header=None)  # ヘッダーなしで読み込む
            print(f"Shape of weights: {weights_df.shape}")  # データの形状を表示
            weights_matrix = torch.tensor(weights_df.values, dtype=torch.float32, device=self.device)
            # CSVの全ての値を合計
            weight_sum = weights_df.to_numpy().sum()

            # 合計値を表示
            print(f'The sum of the weights is: {weight_sum}')
            
            self.weights = weights_matrix
        else:
            # もしファイルがない場合はランダムな初期化
            self.weights = torch.randn(self.n_total, self.n_total, device=self.device)

    def run_simulation(self, inputs):
        T = inputs.size(0)
        self.spike_record = torch.zeros((self.n_total, T), device=self.device)
        
        # ログ用テンソルを初期化
        num_steps = T
        self.exc_input_log = torch.zeros(num_steps, device=self.device)
        self.inh_input_log = torch.zeros(num_steps, device=self.device)
        self.ei_diff_log = torch.zeros(num_steps, device=self.device)

        for t in range(1, T):
            # スパイク状態を使ってシナプスの重みを更新
            self.before_I = self.synapse(self.spike_state, self.weights, self.before_I)
            self.sum_I_syn = torch.sum(self.before_I, dim=0)

            # inputからの入力
            self.sum_I_syn[:200] += inputs[t]
            self.sum_I_syn[self.n_exc:50] += inputs[t]
            self.spike_state, self.before_V, self.ref_time = self.neuron(self.sum_I_syn, self.before_V, self.ref_time)

            # Excitatory (興奮性) と Inhibitory (抑制性) の入力を分けてログ
            EI_input = torch.sum(self.before_I, dim=1)
            exc_input = torch.sum(EI_input[:self.n_exc])
            inh_input = torch.sum(EI_input[self.n_exc:])
            ei_diff = exc_input + inh_input  # E - I の差を計算

            # GPU 上でログに保存
            self.exc_input_log[t] = exc_input
            self.inh_input_log[t] = inh_input
            self.ei_diff_log[t] = ei_diff
            self.spike_record[:, t] = self.spike_state

        return self.spike_record
    
    
    def save_spikes_to_csv(self, filename="spikes_random_50.csv"):
        # スパイクデータをそのままバイナリ形式で保存
        spike_train = self.spike_record.cpu().numpy()  # GPU上のテンソルをCPU上のNumPy配列に変換
        np.savetxt(filename, spike_train, delimiter=",")
        print(f"Spikes saved to {filename}")
    
    def plot_raster(self):
        spike_times = torch.nonzero(self.spike_record, as_tuple=False)
        plt.figure(figsize=(12, 8))
        plt.scatter(spike_times[:, 1].cpu() * self.dt, spike_times[:, 0].cpu(), marker="|", color="black", s=10)
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron Index")
        plt.title("Spike Raster Plot")
        plt.savefig("spike_raster_lsm.png", dpi=300)
        plt.show()



if __name__ == '__main__':

    scale_factor = 50

    # CSVファイルの読み込み
    data = pd.read_csv('../NARMA10/narma10_data.csv')

    # Pandasデータフレームをテンソルに変換
    time_tensor = torch.tensor(data['Time'].values, dtype=torch.float32)
    input_tensor = torch.tensor(data['Input'].values, dtype=torch.float32)
    output_tensor = torch.tensor(data['Output'].values, dtype=torch.float32)

    # 重みファイルを指定 (例: 'weights.csv' のパス)
    weight_file = '../random_netwrok/weights.csv'
    weight_file = 0

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    input_tensor = input_tensor.to(device)
    input_tensor *= scale_factor
    
    # SNNのインスタンス化、重みファイルを渡す
    network = SNN(n_exc=1000, n_inh=250, device=device, weight_file=weight_file)
    spike_record = network.run_simulation(input_tensor)
    network.plot_raster()
    network.save_spikes_to_csv()