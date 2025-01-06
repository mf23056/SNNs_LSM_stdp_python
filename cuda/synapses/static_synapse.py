import numpy as np
import matplotlib.pyplot as plt

class StaticSynapse:
    def __init__(self, dt=0.01, tau_syn=25, before_I=0):
        '''
        param dt: 刻み幅 (ms)
        param tau_syn: シナプスの時定数 (ms)
        param before_I: 初期のシナプス電流
        '''
        self.tau_syn = tau_syn  # シナプスの時定数
        self.dt = dt            # 時間の刻み幅
        self.before_I = before_I # 初期のシナプス電流

    # postsynaptic currentを計算
    def __call__(self, bin_spike: int, W: float) -> float:
        '''
        param bin_spike: 入力データ (バイナリスパイク)
        param W: シナプスの重み
        return: postsynaptic current (latest state)
        '''
        # postsynaptic currentの計算
        I = self.before_I + self.dt * (-self.before_I / self.tau_syn) + W * bin_spike

        # 内部状態の更新
        self.before_I = I
        
        return I
    
    
if __name__ == "__main__":
    '''
    common static synapse model
    '''

    # ランダムスパイクデータを生成
    np.random.seed(42)  # 再現性のために乱数シードを設定
    T = 300  # 総シミュレーション時間(ms)
    dt = 0.01  # 刻み幅(ms)
    time = np.arange(0, T, dt)  # 時間軸
    spike_prob = 0.001  # 各時刻でスパイクが発生する確率
    spike_data = np.random.binomial(1, spike_prob, len(time))  # バイナリスパイクデータ

    # StaticSynapseクラスのインスタンスを作成
    tau_syn = 20  # シナプスの時定数
    weight = 1.0  # シナプスの重み
    synapse = StaticSynapse(dt=dt, tau_syn=tau_syn)

    # ポストシナプス電流の計算
    postsynaptic_current = []
    for spike in spike_data:
        I = synapse(bin_spike=spike, W=weight)
        postsynaptic_current.append(I)

    # プロット
    plt.figure(figsize=(10, 6))

    # スパイクデータのプロット
    plt.subplot(2, 1, 1)
    plt.plot(time, spike_data, drawstyle='steps-post', label='Spike Data')
    plt.ylabel('Spike (0 or 1)')
    plt.title('Random Spike Input and Postsynaptic Current')
    plt.grid(True)

    # ポストシナプス電流のプロット
    plt.subplot(2, 1, 2)
    plt.plot(time, postsynaptic_current, label='Postsynaptic Current')
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (I)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("static.png")

