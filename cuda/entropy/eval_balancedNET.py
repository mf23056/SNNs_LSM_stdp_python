import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class LIF:
    def __init__(self, dt=0.01, tau_m=20, V_rest=-65.0, R=10.0, I_back=10.0, V_reset=-65.0, V_th=-50.0, V_ref=3):
        self.dt = dt
        self.tau_m = tau_m
        self.V_rest = V_rest
        self.R = R
        self.I_back = I_back
        self.V_reset = V_reset
        self.V_th = V_th
        self.V_ref_steps = int(V_ref / dt)

    def __call__(self, I_syn, before_V, ref_time):
        ref_time = torch.clamp(ref_time - 1, min=0)
        V = torch.where(ref_time > 0, torch.full_like(before_V, self.V_reset),
                        before_V + self.dt * ((1 / self.tau_m) * (-(before_V - self.V_rest) + self.R * (I_syn + self.I_back))))
        spiked = (V >= self.V_th).float()
        ref_time = torch.where(spiked > 0, self.V_ref_steps, ref_time)
        V = torch.where(spiked > 0, torch.full_like(V, self.V_reset), V)
        return spiked, V, ref_time

class StaticSynapse:
    def __init__(self, dt=0.01, tau_syn=25):
        self.dt = dt
        self.tau_syn = tau_syn

    def __call__(self, bin_spike, W, before_I):
        spikes = bin_spike.unsqueeze(1)
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
        if weight_file:
            weights_df = pd.read_csv(weight_file, header=None)
            weights_matrix = torch.tensor(weights_df.values, dtype=torch.float32, device=self.device)
            self.weights = weights_matrix
        else:
            self.weights = torch.randn(self.n_total, self.n_total, device=self.device)

    def run_simulation(self, inputs):
        T = inputs.size(0)
        self.spike_record = torch.zeros((self.n_total, T), device=self.device)
        for t in range(1, T):
            self.before_I = self.synapse(self.spike_state, self.weights, self.before_I)
            self.sum_I_syn = torch.sum(self.before_I, dim=0)
            self.sum_I_syn[:200] += inputs[t]
            self.spike_state, self.before_V, self.ref_time = self.neuron(self.sum_I_syn, self.before_V, self.ref_time)
            self.spike_record[:, t] = self.spike_state
        return self.spike_record

    def compute_shannon_entropy(self, spike_record):
        entropy_list = []
        n_neurons, T = spike_record.shape

        window_size = int(50 / self.dt)  # 100msのウィンドウ
        for t in range(0, T - window_size, window_size):
            window_data = spike_record[:, t:t + window_size]
            flattened_data = window_data.reshape(-1).cpu().numpy()

            unique, counts = np.unique(flattened_data, return_counts=True)
            probabilities = counts / counts.sum()

            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
            entropy_list.append(entropy)

            # デバッグ用にエントロピーの値を表示
            print(f"Window {t}-{t + window_size}, Entropy: {entropy}")

        return entropy_list


    def plot_entropy(self, entropy_list):
        """
        エントロピーを時間軸でプロット
        :param entropy_list: エントロピーの時間変化
        """
        if len(entropy_list) == 0:
            print("No entropy data to plot.")
            return

        time = np.arange(len(entropy_list)) * 100  # 100msのウィンドウ
        plt.figure(figsize=(10, 6))
        plt.plot(time, entropy_list, label="Entropy over Time", color="blue")
        plt.xlabel("Time (ms)")
        plt.ylabel("Entropy (bits)")
        plt.title("Shannon Entropy over Time")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_spike_raster(self, spike_record, title="Spike Raster Plot"):
        spike_times = torch.nonzero(spike_record, as_tuple=False)
        plt.figure(figsize=(12, 8))
        plt.scatter(spike_times[:, 1].cpu() * self.dt, spike_times[:, 0].cpu(), marker="|", color="black", s=10)
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron Index")
        plt.title(title)
        plt.grid()
        plt.show()

if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = SNN(n_exc=1000, n_inh=250, device=device, weight_file='../random_netwrok/weights.csv')

    # スパイクデータの収集
    inputs = torch.zeros((10000, 1), device=device)  # 任意の入力
    spike_record = network.run_simulation(inputs)

    # エントロピー計算
    entropy_list = network.compute_shannon_entropy(spike_record)
    network.plot_entropy(entropy_list)

    # スパイクラスタープロット
    network.plot_spike_raster(spike_record, title="Spike Raster Plot")
