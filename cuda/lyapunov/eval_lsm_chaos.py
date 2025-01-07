import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class LIF:
    def __init__(self, dt=0.01, tau_m=20, V_rest=-65.0, R=10.0, I_back=0.0, V_reset=-65.0, V_th=-50.0, V_ref=3):
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

class LSM:
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

    def calculate_jacobian(self):
        J = torch.zeros((self.n_total, self.n_total), device=self.device)
        dV_dV = -1 / self.neuron.tau_m
        dV_dI = 1 / self.neuron.tau_m
        dI_dI = -1 / self.synapse.tau_syn
        dI_dV = self.weights / self.synapse.tau_syn
        J += torch.diag(torch.full((self.n_total,), dV_dV, device=self.device))
        J += torch.diag(torch.full((self.n_total,), dV_dI, device=self.device), offset=0)
        J += dI_dV
        J += torch.diag(torch.full((self.n_total,), dI_dI, device=self.device))
        return J

    def run_simulation(self, inputs):
        T = inputs.size(0)
        self.spike_record = torch.zeros((self.n_total, T), device=self.device)
        self.Phi = torch.eye(self.n_total, device=self.device)

        for t in range(1, 100):
            self.before_I = self.synapse(self.spike_state, self.weights, self.before_I)
            self.sum_I_syn = torch.sum(self.before_I, dim=0)
            self.sum_I_syn[:200] += inputs[t]
            self.spike_state, self.before_V, self.ref_time = self.neuron(self.sum_I_syn, self.before_V, self.ref_time)
            J = self.calculate_jacobian()
            exp_J = torch.matrix_exp(J * self.dt)
            spiked_indices = (self.spike_state > 0).nonzero(as_tuple=True)
            S = torch.eye(self.n_total, device=self.device)
            S[spiked_indices, spiked_indices] = 0
            self.Phi = S @ exp_J @ self.Phi
            self.spike_record[:, t] = self.spike_state
        return self.spike_record

    def calculate_lyapunov_exponents(self):
        eigenvalues, _ = torch.linalg.eig(self.Phi)
        lyapunov_exponents = torch.log(torch.abs(eigenvalues)).real / self.Phi.size(0)
        lambda_max = lyapunov_exponents.max()
        return lambda_max

    def save_spikes_to_csv(self, filename="spike_train.csv"):
        spike_train = self.spike_record.cpu().numpy()
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
    scale_factor = 10
    data = pd.read_csv('../NARMA10/narma10_data.csv')
    time_tensor = torch.tensor(data['Time'].values, dtype=torch.float32)
    input_tensor = torch.tensor(data['Input'].values, dtype=torch.float32)
    output_tensor = torch.tensor(data['Output'].values, dtype=torch.float32)

    weight_file = '../random_netwrok/weights.csv'
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_tensor = input_tensor.to(device)
    input_tensor *= scale_factor

    network = LSM(n_exc=1000, n_inh=250, device=device, weight_file=weight_file)
    spike_record = network.run_simulation(input_tensor)
    lambda_max = network.calculate_lyapunov_exponents()
    print(f"Maximum Lyapunov Exponent: {lambda_max}")
    network.plot_raster()
    network.save_spikes_to_csv()
