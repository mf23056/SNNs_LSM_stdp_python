import numpy as np
import matplotlib.pyplot as plt

class LIF:
    def __init__(self, dt=0.01, tau_m=20, V_rest=-65.0, R=10.0, I_back=0.0, V_reset=-65.0, V_th=-50.0, V_ref=5, bef_V=0, r_timer=0):
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
        self.V_ref = V_ref / dt  # 不応期を刻み幅で割る
        self.bef_V = V_reset  # 初期膜電位
        self.r_timer = r_timer  # 不応期タイマー
        self.state = 0

    def __call__(self, I_syn):
        '''
        ニューロンの状態を更新し、スパイクをチェックします。
        param I_syn: 他ニューロンからのI_postの合計値 [nA]
        Returns:
            int: スパイクが発火した場合は1、そうでなければ0
        '''
        self.state = 0
        
        # 不応期の処理
        if self.r_timer <= 0:
            # 膜電位を計算
            V = self.bef_V + self.dt * ((1 / self.tau_m) * (-(self.bef_V - self.V_rest) + self.R * (I_syn + self.I_back)))

            if V >= self.V_th:
                V = self.V_reset  # スパイク発火
                self.r_timer = self.V_ref  # 不応期を設定
                self.state = 1
        else:
            V = self.V_reset  # 不応期中は膜電位をリセット
            self.r_timer -= 1  # 不応期タイマーを減少させる

        self.bef_V = V  # 膜電位を更新
        return self.state



    
    

# 使用例
if __name__ == "__main__":
    # シミュレーションパラメータ
    T = 300  # 総シミュレーション時間 (ms)
    dt = 0.01  # 時間刻み (ms)
    time = np.arange(0, T, dt)  # 時間軸

    # ランダムな入力電流を生成
    np.random.seed(42)  # 再現性のために乱数シードを設定
    input_current = np.random.normal(0.0, 0.0, len(time))  # 平均5.0nA、標準偏差2.0nAのノイズ

    # LIFニューロンのインスタンスを作成
    lif_neuron = LIF(dt=dt)

    # 各時刻における膜電位Vとスパイクの状態を保存するリスト
    V_vals = []
    spike_vals = []

    # シミュレーションループ
    for I in input_current:
        spike = lif_neuron(I)
        V_vals.append(lif_neuron.bef_V)
        spike_vals.append(spike)

    # プロットの準備
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # スパイクデータのプロット
    axs[0].plot(time, spike_vals, label="Spikes", color="black")
    axs[0].set_ylabel("Spike")
    axs[0].legend(loc="upper right")
    axs[0].set_title("Spike Data and Membrane Potential Over Time")

    # 膜電位Vのプロット
    axs[1].plot(time, V_vals, label="Membrane Potential (V)", color="blue")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Membrane Potential (V) [mV]")
    axs[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("lif_neuron.png", dpi=300)
        
    