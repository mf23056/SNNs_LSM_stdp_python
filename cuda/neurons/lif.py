import torch
import matplotlib.pyplot as plt

class LIF:
    def __init__(self, dt=0.01, tau_m=20, V_rest=-65.0, R=10.0, I_back=0.0, V_reset=-65.0, V_th=-55.0, V_ref=5):
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
        ref_time = torch.maximum(ref_time - 1, torch.zeros_like(ref_time))
        V = torch.where(ref_time > 0, torch.full_like(before_V, self.V_reset),
                        before_V + self.dt * ((1 / self.tau_m) * (-(before_V - self.V_rest) + self.R * (I_syn + self.I_back))))
        spiked = (V >= self.V_th).int()
        ref_time = torch.where(spiked > 0, self.V_ref_steps, ref_time)
        V = torch.where(spiked > 0, torch.full_like(V, self.V_reset), V)
        return spiked, V, ref_time   
    

# 使用例
if __name__ == "__main__":
   # テストパラメータ
    dt = 0.01  # 時間刻み
    simulation_time = 300  # シミュレーション時間 (ms)
    time_steps = int(simulation_time / dt)  # 時間ステップ数

    # 入力シナプス電流（定数で試す）
    I_syn = torch.tensor([0.0])  # 入力電流
    V = torch.tensor([-65.0])  # 初期膜電位
    ref_time = torch.tensor([0.0])  # リフラクトリ時間

    # LIFニューロンの初期化
    lif_neuron = LIF(dt=dt)

    # 結果を記録するリスト
    voltages = []
    spikes = []
    currents = []  # 入力電流を記録するリスト


    # シミュレーションの実行
    for t in range(time_steps):
        # 最初の100msは電流0、それ以降は乱数で入力電流を設定
        if t < int(100/dt):
            I_syn = torch.tensor([0.0])  # 最初の100msは電流0
            
        elif t < int(200/dt):
            I_syn = torch.tensor([1.2])  # 0.0~3000pAの範囲
        
        else:
            I_syn = torch.tensor([torch.rand(1) * 1.2])  # 乱数で0.0〜200.0pAの範囲

        spiked, V, ref_time = lif_neuron(I_syn, V, ref_time)
        voltages.append(V.item())  # 膜電位を記録
        spikes.append(spiked.item())  # スパイクを記録
        currents.append(I_syn.item())  # 入力電流を記録

    # プロット
    time = [dt * i for i in range(time_steps)]  # 時間軸

    plt.figure(figsize=(12, 8))

    # 膜電位のプロット
    plt.subplot(3, 1, 1)
    plt.plot(time, voltages, label="Membrane Potential (V)")
    plt.axhline(y=lif_neuron.V_th, color='r', linestyle='--', label="Threshold Voltage")
    plt.axhline(y=lif_neuron.V_reset, color='b', linestyle='--', label="Reset Voltage")
    plt.title("LIF Neuron Membrane Potential")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.grid()

    # スパイクのプロット
    plt.subplot(3, 1, 2)
    plt.plot(time, spikes, label="Spike Output", color='g')
    plt.title("LIF Neuron Spike Output")
    plt.xlabel("Time (ms)")
    plt.ylabel("Spike")
    plt.yticks([0, 1], ["No Spike", "Spike"])
    plt.legend()
    plt.grid()

    # 入力電流のプロット
    plt.subplot(3, 1, 3)
    plt.plot(time, currents, label="Input Current (I_syn)", color='orange')
    plt.title("Input Current to LIF Neuron")
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (nA)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()