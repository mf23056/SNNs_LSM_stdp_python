import numpy as np
import matplotlib.pyplot as plt

class LIFNeuron:
    def __init__(self, R=10.0, membrane_time_constant=20.0, reset_potential=-65.0, threshold=-50.0, v_rest=-65.0):
        self.R = R  # 膜抵抗
        self.tau = membrane_time_constant  # 膜時間定数
        self.reset_potential = reset_potential  # リセット電位
        self.threshold = threshold  # 発火閾値
        self.v_rest = v_rest  # 静止電位
        self.v_m = v_rest  # 膜電位
        self.spike = False  # 発火フラグ

    def update(self, I, dt):
        """
        ニューロンの膜電位を更新します。

        :param I: 入力電流
        :param dt: 時間ステップ
        """
        # 膜電位の更新
        dv = (self.v_rest - self.v_m + self.R * I) / self.tau * dt
        self.v_m += dv
        
        # 発火条件をチェック
        if self.v_m >= self.threshold:
            self.spike = True
            self.v_m = self.reset_potential  # リセット電位に戻す
        else:
            self.spike = False

    def get_membrane_potential(self):
        """現在の膜電位を返します。"""
        return self.v_m

    def has_spiked(self):
        """発火したかどうかを返します。"""
        return self.spike

# 使用例
if __name__ == "__main__":
    dt = 1.0  # 時間ステップ
    I = 10.0  # 入力電流
    neuron = LIFNeuron(R=10.0)  # 抵抗を指定してニューロンを作成

    time = np.arange(0, 100, dt)  # 時間軸
    membrane_potential = []  # 膜電位のリスト
    spikes = []  # スパイクの時間を記録するリスト

    for t in time:
        neuron.update(I, dt)
        membrane_potential.append(neuron.get_membrane_potential())
        if neuron.has_spiked():
            spikes.append(t)  # スパイクの時間を追加

    # プロット
    plt.figure(figsize=(10, 5))
    plt.plot(time, membrane_potential, label='Membrane Potential (V)', color='blue')
    
    # スパイクを赤い線で示す
    for spike in spikes:
        plt.axvline(x=spike, color='red', linestyle='--', label='Spike' if spike == spikes[0] else "")

    plt.title('LIF Neuron Membrane Potential and Spikes')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.ylim(-80, 0)  # Y軸の範囲を設定
    plt.legend()
    plt.grid()
    plt.show()
