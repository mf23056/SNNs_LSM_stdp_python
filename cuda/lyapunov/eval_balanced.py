import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class LIF:
    def __init__(self, dt=0.01, tau_m=20, V_rest=-65.0, R=10.0, I_back=6.0, V_reset=-65.0, V_th=-50.0, V_ref=3):
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
            self.sum_I_syn[self.n_exc:50] += inputs[t]
            self.spike_state, self.before_V, self.ref_time = self.neuron(self.sum_I_syn, self.before_V, self.ref_time)
            self.spike_record[:, t] = self.spike_state
        return self.spike_record

    def calculate_lyapunov_exponent(self, epsilon=1e-6, num_steps=1000):
        self._initialize_neurons(self.n_exc, self.n_inh)
        original_state = {
            "V": self.before_V.clone(),
            "spike_state": self.spike_state.clone(),
            "ref_time": self.ref_time.clone(),
        }
        perturbed_V = original_state["V"] + epsilon * torch.randn_like(original_state["V"])
        perturbed_state = {
            "V": perturbed_V,
            "spike_state": original_state["spike_state"].clone(),
            "ref_time": original_state["ref_time"].clone(),
        }
        distances = []
        log_lyapunov_exponents = []
        for t in range(num_steps):
            self.before_V, self.spike_state, self.ref_time = (
                original_state["V"],
                original_state["spike_state"],
                original_state["ref_time"],
            )
            self.before_I = self.synapse(self.spike_state, self.weights, self.before_I)
            sum_I_syn = torch.sum(self.before_I, dim=0)
            spiked, V, ref_time = self.neuron(sum_I_syn, self.before_V, self.ref_time)
            original_state = {"V": V, "spike_state": spiked, "ref_time": ref_time}

            self.before_V, self.spike_state, self.ref_time = (
                perturbed_state["V"],
                perturbed_state["spike_state"],
                perturbed_state["ref_time"],
            )
            self.before_I = self.synapse(self.spike_state, self.weights, self.before_I)
            sum_I_syn = torch.sum(self.before_I, dim=0)
            spiked, V, ref_time = self.neuron(sum_I_syn, self.before_V, self.ref_time)
            perturbed_state = {"V": V, "spike_state": spiked, "ref_time": ref_time}

            distance = torch.norm(original_state["V"] - perturbed_state["V"], p=2)
            distances.append(distance.item())

            if t > 0:
                # 距離がゼロでないことを確認
                if distances[-2] > 0:
                    log_lyapunov_exponent = torch.log(torch.tensor(distance.item() / distances[-2]))
                    log_lyapunov_exponents.append(log_lyapunov_exponent.item())
                else:
                    log_lyapunov_exponents.append(float('nan'))  # ゼロの場合はNaNを追加

            if distance > epsilon * 10:
                scaling_factor = epsilon / distance
                perturbed_state["V"] = original_state["V"] + scaling_factor * (
                    perturbed_state["V"] - original_state["V"]
                )

        distances = torch.tensor(distances)
        lyapunov_exponent = torch.mean(torch.tensor(log_lyapunov_exponents)).item()
        return lyapunov_exponent, log_lyapunov_exponents

    def plot_lyapunov_exponents(self, log_lyapunov_exponents, dt):
        time = torch.arange(len(log_lyapunov_exponents)) * dt
        plt.figure(figsize=(10, 6))
        plt.plot(time.cpu(), log_lyapunov_exponents, label="Lyapunov Exponent")
        plt.xlabel("Time (ms)")
        plt.ylabel("LE Value")
        plt.title("Lyapunov Exponent Over Time")
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

if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = SNN(n_exc=1000, n_inh=250, device=device, weight_file='../random_netwrok/weights.csv')

    # スパイクデータの収集
    inputs = torch.zeros((10000, 1), device=device)  # 任意の入力
    # spike_record_no_perturb = network.run_simulation(inputs)

    # 摂動ありでのシミュレーション
    le, log_lyapunov_exponents = network.calculate_lyapunov_exponent(epsilon=1e-6, num_steps=10000)

    # 摂動なしスパイクラスタープロット
    # network.plot_spike_raster(spike_record_no_perturb, title="Spike Raster Plot (No Perturbation)")
    
    # 摂動ありスパイクラスタープロット
    spike_record_with_perturb = network.run_simulation(inputs)  # 摂動ありのスパイク
    network.plot_spike_raster(spike_record_with_perturb, title="Spike Raster Plot (With Perturbation)")
    
    # plt.tight_layout()
    # plt.show()

    # LEsのプロット
    network.plot_lyapunov_exponents(log_lyapunov_exponents, network.dt)
