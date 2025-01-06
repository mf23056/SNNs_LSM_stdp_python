import numpy as np
import matplotlib.pyplot as plt 

# Long-term plasticity
class Guetig_STDP:
    def __init__(self, dt=0.01, A_plus=0.01, A_minus=0.01, tau_plus=30.0, tau_minus=30.0, alpha=0.999):
        '''
        Parameters:
        delta_t (float): Time difference between post-synaptic spike and pre-synaptic spike (t_post - t_pre).
        A_plus (float): Amplitude of the weight change for potentiation (when delta_t > 0).
        A_minus (float): Amplitude of the weight change for depression (when delta_t < 0).
        tau_plus (float): Time constant for potentiation.
        tau_minus (float): Time constant for depression.
        '''
        self.dt = dt
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.alpha = alpha
        
    
    
    def __call__(self, delta_t) -> float:
        '''
        Returns:
        float: Change in synaptic weight.
        '''
        
        if delta_t > 0:
            # Long-term potentiation (LTP) with Gütig's modification
            delta_w = self.A_plus * (np.exp(-delta_t / self.tau_plus) - self.alpha * np.exp(-delta_t / (self.alpha * self.tau_plus)))
        else:
            # Long-term depression (LTD) with Gütig's modification
            delta_w = -self.A_minus * (np.exp(delta_t / self.tau_minus) - self.alpha * np.exp(delta_t / (self.alpha * self.tau_minus)))

        return delta_w


if __name__ == "__main__":
    '''
    guetig stdp - long term plasticity
    '''
    
    # Guetig STDPのインスタンスを作成
    stdp = Guetig_STDP()

    # delta_tの範囲を設定
    delta_ts = np.linspace(-100, 100, 400)
    changes = [stdp(dt) for dt in delta_ts]

    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(delta_ts, changes, label='Change in Synaptic Weight', color='blue')
    plt.title('Guetig STDP: Change in Synaptic Weight vs. Delta t')
    plt.xlabel('Delta t (ms)')
    plt.ylabel('Change in Weight (Δw)')
    plt.axhline(0, color='red', linestyle='--')  # y=0のラインを追加
    plt.grid()
    plt.legend()
    plt.savefig("stdp_guetig.png", dpi=300)