import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class SmartIrrigation:
    def __init__(self, a=0.5, b=1.0, t_max=20, dt=0.01):
        """
        Initialise system parameters and time array.
          dh/dt + a*h(t) = b*u(t)
        """
        self.a = a
        self.b = b
        self.t_max = t_max
        self.dt = dt
        self.t = np.arange(0, t_max, dt)   # time grid [0, t_max)

    # ------------------------------------------------------------------
    # Input functions
    # ------------------------------------------------------------------
    def u_step(self):
        """u(t) = 1  for all t >= 0"""
        return np.ones_like(self.t)

    def u_ramp(self):
        """u(t) = 0.1*t"""
        return 0.1 * self.t

    def u_sin(self):
        """u(t) = sin(0.5*t)"""
        return np.sin(0.5 * self.t)

    def u_exponential(self):
        """u(t) = 1 - exp(-0.3*t)"""
        return 1.0 - np.exp(-0.3 * self.t)

    def u_pulse(self):
        """u(t) = 1 for 0<=t<5, else 0"""
        u = np.zeros_like(self.t)
        u[self.t < 5.0] = 1.0
        return u

    # ------------------------------------------------------------------
    # Laplace Transform  (trapezoidal rule, no scipy)
    # ------------------------------------------------------------------
    def laplace_transform(self, f, s):
        """
        Numerically approximate:
            F(s) = integral_0^{T} f(t) * exp(-s*t) dt
        using the trapezoidal rule.
        """
        integrand = f * np.exp(-s * self.t)
        return np.trapezoid(integrand, self.t)

    # ------------------------------------------------------------------
    # Transfer function in s-domain
    # ------------------------------------------------------------------
    def H_s(self, s, U_s):
        """
        H(s) = b / (s + a) * U(s)
        Works element-wise when s and U_s are arrays.
        """
        return (self.b / (s + self.a)) * U_s

    # ------------------------------------------------------------------
    # Inverse Laplace via Bromwich contour
    # ------------------------------------------------------------------
    def inverse_laplace(self, s_list, F_s_values):
        """
        Recover h(t) from H(s) using the Bromwich (Fourier-Mellin) contour:

            h(t) ≈ (Δω / 2π) * Re( Σ_k  H(c + i*ω_k) * exp((c + i*ω_k)*t) )

        Parameters
        ----------
        s_list     : 1-D complex array  c + i*ω_k,  k = 0 … N-1
        F_s_values : 1-D complex array  H(s_list[k])
        """
        N = len(s_list)
        # Recover Δω = 2W/N from the imaginary part of the last sample
        W = s_list[-1].imag          # ≈ +W  (linspace endpoint)
        delta_omega = 2.0 * W / N

        # Vectorised outer product: shape (N, M)  where M = len(self.t)
        # exp_matrix[k, i] = exp(s_k * t_i)
        exp_matrix = np.exp(np.outer(s_list, self.t))

        # Sum over k: (N,) @ (N, M) → (M,)
        h = (delta_omega / (2.0 * np.pi)) * np.real(F_s_values @ exp_matrix)
        return h

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------
    def steady_state(self, h):
        """Mean of last 5% of signal."""
        n = len(h)
        return float(np.mean(h[int(0.95 * n):]))

    def time_constant(self, h):
        """Time to first reach 63.2% of steady-state."""
        h_ss = self.steady_state(h)
        h_peak = float(np.max(np.abs(h)))
        if abs(h_ss) < 1e-3 * max(h_peak, 1e-10) or abs(h_ss) < 1e-10:
            return float('nan')
        idx = np.where(h >= 0.632 * h_ss)[0]
        return float(self.t[idx[0]]) if len(idx) > 0 else float('nan')

    def rise_time(self, h):
        """Time to go from 10% to 90% of steady-state."""
        h_ss = self.steady_state(h)
        h_peak = float(np.max(np.abs(h)))
        if abs(h_ss) < 1e-3 * max(h_peak, 1e-10) or abs(h_ss) < 1e-10:
            return float('nan')
        idx10 = np.where(h >= 0.10 * h_ss)[0]
        idx90 = np.where(h >= 0.90 * h_ss)[0]
        if len(idx10) == 0 or len(idx90) == 0:
            return float('nan')
        return float(self.t[idx90[0]] - self.t[idx10[0]])

    def settling_time(self, h):
        """Time after which h(t) stays permanently within +-2% of h_ss."""
        h_ss = self.steady_state(h)
        h_peak = float(np.max(np.abs(h)))
        if abs(h_ss) < 1e-3 * max(h_peak, 1e-10) or abs(h_ss) < 1e-10:
            return float('nan')
        band = 0.02 * abs(h_ss)
        outside = np.where(np.abs(h - h_ss) > band)[0]
        if len(outside) == 0:
            return float(self.t[0])
        return float(self.t[outside[-1]])

    def overshoot(self, h):
        """Percentage overshoot: (h_max - h_ss) / |h_ss| * 100, or 0/nan."""
        h_ss = self.steady_state(h)
        h_peak = float(np.max(np.abs(h)))
        if abs(h_ss) < 1e-3 * max(h_peak, 1e-10) or abs(h_ss) < 1e-10:
            return float('nan')
        h_max = float(np.max(h))
        if h_max <= h_ss * 1.001:
            return 0.0
        return float((h_max - h_ss) / abs(h_ss) * 100.0)

    def compute_metrics(self, h):
        return {
            "steady_state":  self.steady_state(h),
            "time_constant": self.time_constant(h),
            "rise_time":     self.rise_time(h),
            "settling_time": self.settling_time(h),
            "overshoot_%":   self.overshoot(h),
        }

    # ------------------------------------------------------------------
    # Euler simulation (provided – do not modify)
    # ------------------------------------------------------------------
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


# ======================================================================
# Helper: run the full pipeline for one (a, b) configuration
# ======================================================================
def run_config(a, b, t_max=20, dt=0.01, save_prefix=""):
    """
    Build a SmartIrrigation system, simulate all 5 inputs,
    print metrics, and save plots.
    """
    system = SmartIrrigation(a=a, b=b, t_max=t_max, dt=dt)

    inputs = {
        "Step Input":        system.u_step(),
        "Ramp Input":        system.u_ramp(),
        "Sinusoidal Input":  system.u_sin(),
        "Exponential Input": system.u_exponential(),
        "Pulse Input":       system.u_pulse(),
    }

    # ------------------------------------------------------------------
    # Bromwich contour parameters
    # ------------------------------------------------------------------
    # • The rightmost pole of any H(s) = b/(s+a)·U(s) is at s = 0
    #   (from the step / ramp / exponential inputs whose Laplace transforms
    #   have poles at s=0).  We need c > 0.
    #   Choose c = 0.5:
    #     - safely right of all poles (0, -a, input poles)
    #     - exp(c·t_max) = e^{10} ≈ 2.2×10⁴  →  no floating-point overflow
    #     - forward Laplace truncation:  e^{-c·T} = e^{-10} ≈ 4.5×10⁻⁵  ≈ 0
    #
    # • W must exceed the highest significant frequency in the signal.
    #   Nyquist of the time grid: ω_N = π/dt ≈ 314 rad/s.
    #   Use W = 200 rad/s (covers all physics; saves memory/time).
    #
    # • N controls ω-resolution: Δω = 2W/N.
    #   Need Δω < 2π/t_max ≈ 0.31  →  N > 2×200/0.31 ≈ 1290.
    #   Use N = 4000 for good accuracy (Δω = 0.1 rad/s).
    # ------------------------------------------------------------------
    c = 0.5
    W = 200.0
    N = 4000
    omega  = np.linspace(-W, W, N, endpoint=False)
    s_list = c + 1j * omega

    colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']

    print(f"\n{'='*60}")
    print(f"  Configuration: a={a}, b={b}")
    print(f"  Bromwich: c={c}, W={W}, N={N}")
    print(f"{'='*60}")

    saved_plots = []

    for idx, (name, u) in enumerate(inputs.items()):
        print(f"\nProcessing: {name} ...")

        # ---- Laplace transform of input u(t) -------------------------
        # Vectorised: compute U(s) for all s_list at once
        #   exp_fwd[k, i] = exp(-s_k * t_i)
        exp_fwd  = np.exp(-np.outer(s_list, system.t))   # (N, M)
        U_s_vals = np.trapezoid(u * exp_fwd, system.t, axis=1)   # (N,)

        # ---- Transfer function H(s) = b/(s+a) * U(s) ----------------
        H_s_vals = system.H_s(s_list, U_s_vals)   # (N,)

        # ---- Inverse Laplace: recover h(t) ---------------------------
        h_laplace = system.inverse_laplace(s_list, H_s_vals)   # (M,)

        # ---- Metrics -------------------------------------------------
        print(f"\n  ► {name}")
        metrics = system.compute_metrics(h_laplace)
        for k, v in metrics.items():
            print(f"      {k.replace('_',' ').title():<22}: {v:.4f}" if isinstance(v, float) else
                  f"      {k.replace('_',' ').title():<22}: {v}")

        # ---- Euler reference -----------------------------------------
        h_euler = system.euler_simulate(u)

        # ---- Side-by-side plot ---------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        fig.suptitle(
            f"Smart Irrigation — {name}  [a={a}, b={b}]",
            fontsize=13, fontweight='bold'
        )

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
        fname = f"{save_prefix}{name.replace(' ', '_')}_a{a}_b{b}.png"
        fpath = os.path.join(OUTPUT_DIR, fname)
        plt.savefig(fpath, dpi=100, bbox_inches='tight')
        plt.close()
        saved_plots.append(fpath)
        print(f"      Plot saved → {fname}")

    return saved_plots


# ======================================================================
# Main: default config + parameter variation
# ======================================================================
all_plots = []

# --- Default configuration -------------------------------------------
all_plots += run_config(a=0.5, b=1.0)

# --- Parameter variation (Task 8) ------------------------------------
# Config 2: faster decay (a=1.0) → smaller steady-state, faster settling
all_plots += run_config(a=1.0, b=1.0)

# Config 3: higher gain (b=2.0) → larger steady-state output
all_plots += run_config(a=0.5, b=2.0)

# Config 4: slow decay + high gain → large, slow-settling response
all_plots += run_config(a=0.2, b=1.5)

print("\n\nAll done. Plots saved to:", OUTPUT_DIR)
print("Total plots:", len(all_plots))