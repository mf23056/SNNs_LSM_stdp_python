import torch
import numpy as np
import matplotlib.pyplot as plt

# Guetig_STDPクラス
class Guetig_STDP:
    def __init__(self, dt=0.01, A_plus=0.01, A_minus=0.01, tau_plus=10.0, tau_minus=20.0, alpha=0.95, device='cuda'):
        self.dt = dt
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.alpha = alpha
        self.device = device

    def __call__(self, delta_t):
        delta_t_tensor = torch.tensor(delta_t, dtype=torch.float32, device=self.device)
        delta_w = torch.where(
            delta_t_tensor > 0,
            self.A_plus
            * (torch.exp(-delta_t_tensor / self.tau_plus) - self.alpha * torch.exp(-delta_t_tensor / (self.alpha * self.tau_plus))),
            -self.A_minus
            * (torch.exp(delta_t_tensor / self.tau_minus) - self.alpha * torch.exp(delta_t_tensor / (self.alpha * self.tau_minus))),
        )
        return delta_w.cpu().numpy()  # NumPy形式に変換

# Δt範囲の設定
delta_t = np.linspace(-100, 100, 500)  # -100msから100msの範囲

# Guetig_STDPインスタンスの作成
stdp = Guetig_STDP(device='cpu')

# Δwを計算
delta_w = stdp(delta_t)

# プロット
plt.figure(figsize=(10, 6))
plt.plot(delta_t, delta_w, label="Change in Synaptic Weight", color="blue")
plt.axhline(0, color="red", linestyle="--", label="Zero Change")
plt.title("Guetig STDP: Change in Synaptic Weight vs. Delta t")
plt.xlabel("Delta t (ms)")
plt.ylabel("Change in Weight (Δw)")
plt.legend()
plt.grid()
plt.savefig("guetig_stdp.png", dpi=300)
plt.show()
