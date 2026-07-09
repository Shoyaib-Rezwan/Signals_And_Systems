import numpy as np
import matplotlib.pyplot as plt
import os
class SmartIrrigation:
    def __init__(self, a=0.5, b=1.0, t_max=20, dt=0.01):
        self.a=a
        self.b=b
        self.t_max=t_max
        self.dt=dt
        self.t=np.arange(0,t_max,dt)

    def u_step(self):
        return np.ones_like(self.t)
    def u_ramp(self):
        return 0.1*self.t
    def u_sin(self):
        return np.sin(0.5*self.t)
    def u_exponential(self): 
        return 1-np.exp(-0.3*self.t)
    def u_pulse(self):
        return np.where(self.t<5, 1,0)

    def laplace_transform(self, f, s):
        result=np.zeros_like(s,dtype=np.complex128)
        for i,s_val in enumerate(s):
            result[i]=np.trapezoid(f*np.exp(-s_val*self.t),self.t)
            # y = f * np.exp(-s_val * self.t)
            # result[i] = np.sum((y[:-1] + y[1:]) / 2.0) * self.dt
        return result

    def inverse_laplace(self, s_list, F_s_values):
        N=len(s_list)
        W=s_list[-1].imag
        del_w=2*W/N
        h=np.zeros_like(self.t)
        for i,t in enumerate(self.t):
            h[i]= del_w/(2*np.pi)*np.real(np.sum(F_s_values*np.exp(s_list*t)))
        return h

    def H_s(self, s, U_s):
        return (self.b/(s+self.a))*U_s
    
    def steady_state(self, h):
        """Mean of last 5% of signal."""
        return np.mean(h[int(len(h)*(1-0.05)):])

    def time_constant(self, h):
        """Time to first reach 63.2% of steady-state."""
        initial_height=h[0]
        final_height=self.steady_state(h)
        target_height=h[0]+0.632*(final_height-initial_height)
        indices=np.where(h>=target_height)[0]
        if(len(indices)>0):
            return self.t[indices[0]]
        return self.t[-1]

    def rise_time(self, h):
        """Time to go from 10% to 90% of steady-state."""
        initial_height=h[0]
        final_height=self.steady_state(h)
        target_height=h[0]+0.1*(final_height-initial_height)
        indices=np.where(h>=target_height)[0]
        if(len(indices)==0):
            return 0
        t1=self.t[indices[0]]
        target_height=h[0]+0.9*(final_height-initial_height)
        indices=np.where(h>=target_height)[0]
        if(len(indices)==0):
            return 0
        t2=self.t[indices[0]]
        return t2-t1

    def settling_time(self, h):
        """Time after which h(t) stays permanently within ±2% of h_ss."""
        h_ss=self.steady_state(h)
        h_diff=0.02*np.abs(h_ss)
        out_indices=np.where(np.abs(h-h_ss)>h_diff)[0]
        if len(out_indices)==0:
            return self.t[0]
        return self.t[out_indices[-1]]

    def overshoot(self, h):
        """Percentage overshoot: (h_max - h_ss) / h_ss * 100."""
        h_ss= self.steady_state(h)
        h_max=np.max(np.abs(h))
        if np.abs(h_ss)<1e-6:
            return 0.0
        return (h_max-h_ss)/h_ss*100

    def compute_metrics(self, h):
       
        return {
            "steady_state":  self.steady_state(h),
            "time_constant": self.time_constant(h),
            "rise_time":     self.rise_time(h),
            "settling_time": self.settling_time(h),
            "overshoot_%":   self.overshoot(h),
        }

    def euler_simulate(self, u):
        """
        Euler method for dh/dt = -a*h(t) + b*u(t)
        h[n+1] = h[n] + dt * (-a*h[n] + b*u[n])
        """
        h = np.zeros_like(self.t)
        for n in range(len(self.t) - 1):
            dhdt = -self.a * h[n] + self.b * u[n]
            h[n + 1] = h[n] + self.dt * dhdt
        return h

def simulate_system(a=0.5,b=1.0):
    #Change values of a, b to experiment with different system dynamics
    system = SmartIrrigation(a, b, t_max=20, dt=0.01)

    inputs = {
        "Step Input":        system.u_step(),
        "Ramp Input":        system.u_ramp(),
        "Sinusoidal Input":  system.u_sin(),
        "Exponential Input": system.u_exponential(),
        "Pulse Input":       system.u_pulse(),
    }

    # Bromwich contour parameters, set these values
    c = 0.5
    W = 100.0
    N = 4000
    print("-" * 30)
    print("BROMWICH CONTOUR JUSTIFICATION:")
    print(f"1. Pole Location: The system pole is at s = -a = -{system.a}")
    print(f"   -Stability: c ({c}) is > pole (-{system.a}), satisfying the Bromwich condition.")
    print(f"   -Convergence: exp(-c * t_max) = {np.exp(-c * system.t_max)}")
    print("    -This value is near zero, ensuring the integrand has sufficiently decayed.")
    print(f"2. Frequency (W={W}):")
    print(f"   - System bandwidth (a) is {system.a} rad/s.")
    print(f"   - W is set to {W} rad/s to capture high-frequency components from sharp inputs like Step and Pulse functions.")
    print(f"5. Number of samples (N={N}):")
    print(f"   - Sampling spacing: Δω = 2W/N = {2*W/N:.4f} rad/s")
    print(f"   - Time resolution: Δt = π/W = {np.pi/W:.4f} s")
    print(f"   - N={N} provides sufficient resolution for t_max={system.t_max}s")
    print("-" * 30)

    omega=np.linspace(-W,W,N, endpoint=False)
    s_list = c+1j*omega

    colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']

    for idx, (name, u) in enumerate(inputs.items()):
        print(f"Processing: {name}...")

        # --- Laplace --- set these values
        U_s_vals =system.laplace_transform(u,s_list)
        H_s_vals = system.H_s(s_list,U_s_vals)
        h_laplace = system.inverse_laplace(s_list, H_s_vals)
        print(f"\n  ► {name}")
        metrics = system.compute_metrics(h_laplace)
        for k, v in metrics.items():
            print(f"      {k.replace('_',' ').title():<22}: {v}")

        # --- Euler ---
        h_euler = system.euler_simulate(u)

        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        fig.suptitle(f"Smart Irrigation(a={a}, b={b}) — {name}", fontsize=13, fontweight='bold')

        # Laplace subplot
        axes[0].plot(system.t, u, 'b--', lw=1.8, label="Input u(t)")
        axes[0].plot(system.t, h_laplace, color=colors[idx], lw=2.2, label="Output h(t)")
        axes[0].set_title("Laplace Transform Simulation", fontweight='bold')
        axes[0].set_xlabel("Time (s)", fontsize=11)
        axes[0].set_ylabel("Water Level / Input", fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Euler subplot
        axes[1].plot(system.t, u, 'b--', lw=1.8, label="Input u(t)")
        axes[1].plot(system.t, h_euler, color='tomato', lw=2.2, label="Output h(t)")
        axes[1].set_title("Euler Method Simulation", fontweight='bold')
        axes[1].set_xlabel("Time (s)", fontsize=11)
        axes[1].set_ylabel("Water Level / Input", fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        folder='output_plots'
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(f'output_plots/{name}_a({a})_b({b}).png')
        # plt.show()

def main():
    # Bromwich contour parameters, set these values
    a=0.5
    b=1.0
    print(f"Simulating system with a= {a} and b={b}.....")
    simulate_system()
    for i in range(2):
        a,b=map(float,input("Enter 2 floating point values for a and b: ").split())
        print(f"Simulating system with a= {a} and b={b}.....")
        simulate_system(a,b)

if __name__=="__main__":
    main()