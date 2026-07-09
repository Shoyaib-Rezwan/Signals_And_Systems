import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Abstract Base Class for Continuous-Time Signals
# =====================================================
class ContinuousSignal:
    """
    Abstract base class for all continuous-time signals.
    Every signal must be defined over a time axis t.
    """

    def __init__(self, t):
        self.t = t

    def values(self):
        """
        Returns the signal values evaluated over time axis t.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def plot(self, title="Signal"):
        """
        Plot the signal in the time domain.
        """
        vals = self.values()
        if np.iscomplexobj(vals):
            plt.figure(figsize=(10, 4))
            plt.plot(self.t, np.real(vals), label="Real part")
            plt.plot(self.t, np.imag(vals), label="Imag part", linestyle='--')
            plt.legend()
        else:
            plt.plot(self.t, vals)
        plt.xlabel("Time (t)")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.grid(True)
        plt.show()


# =====================================================
# Signal Generator Class
# =====================================================
class SignalGenerator(ContinuousSignal):
    """
    Generates various continuous-time signals.
    Each method returns a numpy array of signal samples.
    """

    def sine(self, amplitude, frequency):
        """Generate a sine wave."""
        return amplitude * np.sin(2 * np.pi * frequency * self.t)

    def cosine(self, amplitude, frequency):
        """Generate a cosine wave."""
        return amplitude * np.cos(2 * np.pi * frequency * self.t)

    def square(self, amplitude, frequency):
        """Generate a square wave using sign of sine."""
        sine_wave = np.sin(2 * np.pi * frequency * self.t)
        return amplitude * np.sign(sine_wave)

    def sawtooth(self, amplitude, frequency):
        """Generate a sawtooth wave."""
        return amplitude * 2 * (frequency * self.t - np.floor(0.5 + frequency * self.t))

    def triangle(self, amplitude, frequency):
        """Generate a triangle wave."""
        return 2 * (amplitude/np.pi) * np.arcsin(self.sine(1, frequency))

    def cubic(self, coefficient):
        """Generate a cubic polynomial signal."""
        return coefficient * self.t**3

    def parabolic(self, coefficient):
        """Generate a parabolic signal."""
        return coefficient * self.t**2

    def rectangular(self, width):
        """Generate a rectangular window centered at t=0."""
        return np.where(np.abs(self.t) <= width/2, 1.0, 0.0)

    def pulse(self, start, end):
        """Generate a finite pulse active between start and end."""
        return np.where((self.t >= start) & (self.t <= end), 1.0, 0.0)


# =====================================================
# Composite Signal Class
# =====================================================
class CompositeSignal(ContinuousSignal):
    """
    Combines multiple signals into a single composite signal.
    """

    def __init__(self, t):
        super().__init__(t)
        self.components = []

    def add_component(self, signal):
        """
        Add a signal component to the composite signal.
        """
        self.components.append(signal)

    def values(self):
        """
        Sum all signal components.
        """
        if not self.components:
            return np.zeros_like(self.t)
        res = np.zeros_like(self.t)
        for components in self.components:
            res += components
        return res


# =====================================================
# Modified Signal Class
# =====================================================
class ModifiedSignal(ContinuousSignal):
    """
    Applies time compression and phase shift to a base signal.
    y(t) = x(a*t) * e^(j2πf₀t)
    """
    def __init__(self, t,a,f0,base_signal:CompositeSignal):
        super().__init__(t)
        self.a=a
        self.f0=f0
        self.base_signal=base_signal
    def values(self):
        val=self.base_signal.values()
        val=np.interp(self.t*self.a,self.t,val,left=0,right=0)
        val=val*np.exp(1j*2*np.pi*self.f0*t)
        return val


# =====================================================
# Continuous Fourier Transform Analyzer
# =====================================================
class CFTAnalyzer:
    """
    Computes the Continuous Fourier Transform (CFT)
    using numerical integration (np.trapz).
    """

    def __init__(self, signal, t, frequencies):
        self.signal = signal
        self.t = t
        self.frequencies = frequencies

    def compute_cft(self):
        """
        Compute real and imaginary parts of the CFT.
        """
        real_part = np.zeros_like(self.frequencies)
        imaginary_part = np.zeros_like(self.frequencies)
        signal_values = self.signal.values()
        for i, freq in enumerate(self.frequencies):
            real_part[i] = np.trapezoid(signal_values * np.cos(2 * np.pi * freq * self.t), self.t)
            imaginary_part[i] = -np.trapezoid(signal_values * np.sin(2 * np.pi * freq * self.t), self.t)
        return (real_part, imaginary_part)

    def plot_spectrum(self):
        """
        Plot magnitude spectrum of the signal.
        """
        real, imag = self.compute_cft()
        magnitude = np.sqrt(real**2 + imag**2)
        
        plt.figure(figsize=(10, 4))
        plt.plot(self.frequencies, magnitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title("CFT Magnitude Spectrum")
        plt.grid(True)
        plt.show()


# =====================================================
# Complex CFT Analyzer (for complex signals)
# =====================================================
class ComplexCFTAnalyzer(CFTAnalyzer):
    """
    CFT for complex-valued signals.
    Extends CFTAnalyzer to handle y(t) = complex signal.
    """
    def compute_cft(self):
        """
        Compute CFT for complex signals.
        Uses e^(-j2πft) = cos(2πft) - j*sin(2πft)
        """
        real_part=[]
        imag_part=[]
        for f in self.frequencies:
            result=np.trapezoid(np.real(self.signal.values())*np.cos(2*np.pi*f*t)+np.imag(self.signal.values())*np.sin(2*np.pi*f*t),self.t)
            real_part.append(result)
            result=np.trapezoid(np.imag(self.signal.values())*np.cos(2*np.pi*f*t)-np.real(self.signal.values())*np.sin(2*np.pi*f*t),self.t)
            imag_part.append(result)
        return np.array(real_part),np.array(imag_part)


# =====================================================
# Inverse Continuous Fourier Transform
# =====================================================
class InverseCFT:
    """
    Reconstructs time-domain signal using ICFT.
    """

    def __init__(self, spectrum, frequencies, t):
        self.spectrum = spectrum
        self.frequencies = frequencies
        self.t = t

    def reconstruct(self):
        """
        Perform inverse CFT using numerical integration.
        """
        real_spectrum, imaginary_spectrum = self.spectrum
        reconstructed = np.zeros_like(self.t)
        for i, t_val in enumerate(self.t):
            integrand_real = real_spectrum * np.cos(2 * np.pi * self.frequencies * t_val) - imaginary_spectrum * np.sin(2 * np.pi * self.frequencies * t_val)
            reconstructed[i] = np.trapezoid(integrand_real, self.frequencies)
        return reconstructed


# =====================================================
# CFT Verifier Class
# =====================================================
class CFTVerifier:
    """
    Verifies: Y(f) = (1/|a|) * X((f - f0) / a)
    Plots magnitude and phase comparisons, computes MSE.
    """
    def __init__(self, spectrum_x, spectrum_y, frequencies, a, f0):
        self.spectrum_x = spectrum_x
        self.spectrum_y = spectrum_y
        self.frequencies = frequencies
        self.a = a
        self.f0 = f0
    
    def _magnitude_and_phase(self, spectrum):
        """Convert (real, imag) to (magnitude, phase)"""
        real, imag = spectrum
        magnitude = np.sqrt(real**2 + imag**2)
        phase = np.arctan2(imag, real)
        return magnitude, phase
    
    def _theoretical_prediction(self):
        """
        Compute (1/|a|) * X((f - f0) / a) by interpolating X(f)
        """
        mag_x, phase_x = self._magnitude_and_phase(self.spectrum_x)
        
        # Query frequencies: (f - f0) / a
        query_freqs = (self.frequencies - self.f0) / self.a
        
        # Interpolate X at query frequencies
        mag_predicted = (1.0 / abs(self.a)) * np.interp(
            query_freqs, self.frequencies, mag_x, left=0.0, right=0.0
        )
        phase_predicted = np.interp(
            query_freqs, self.frequencies, phase_x, left=0.0, right=0.0
        )
        
        return mag_predicted, phase_predicted
    
    def verify(self):
        """Plot comparisons and compute MSE"""
        mag_y, phase_y = self._magnitude_and_phase(self.spectrum_y)
        mag_pred, phase_pred = self._theoretical_prediction()
        
        # Plot magnitude comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.frequencies, mag_y, label="|Y(f)|", linewidth=2)
        plt.plot(self.frequencies, mag_pred, '--', 
                 label="(1/|a|)|X((f-f₀)/a)|", linewidth=2)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title("Magnitude Spectrum Verification")
        plt.legend()
        plt.grid(True)
        
        # Plot phase comparison
        plt.subplot(1, 2, 2)
        plt.plot(self.frequencies, phase_y, label="∠Y(f)", linewidth=2)
        plt.plot(self.frequencies, phase_pred, '--',
                 label="∠X((f-f₀)/a)", linewidth=2)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (radians)")
        plt.title("Phase Spectrum Verification")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Compute MSE
        N = len(self.frequencies)
        mse_magnitude = np.sum((mag_y - mag_pred) ** 2) / N
        mse_phase = np.sum((phase_y - phase_pred) ** 2) / N
        
        print("=" * 50)
        print("    CFT Property Verification Results")
        print("=" * 50)
        print(f"  Time compression factor a  = {self.a}")
        print(f"  Frequency shift         f0 = {self.f0} Hz")
        print(f"  Number of samples       N  = {N}")
        print("-" * 50)
        print(f"  MSE (Magnitude) = {mse_magnitude:.6e}")
        print(f"  MSE (Phase)     = {mse_phase:.6e}")
        print("=" * 50)
        
        return mse_magnitude, mse_phase


# =====================================================
# Main Execution
# =====================================================
t = np.linspace(-5, 5, 3000)
frequencies = np.linspace(-10, 10, 1000)

# Parameters
a = 10
f0 = 10

# Build x(t) = Square(t) + Triangle(t)
gen = SignalGenerator(t)

composite = CompositeSignal(t)
composite.add_component(gen.square(1, 1))
composite.add_component(gen.triangle(1, 1))

print("Plotting x(t) = Square(t) + Triangle(t)")
composite.plot("x(t) = Square(t) + Triangle(t)")

# Build y(t) = x(a*t) * e^(j2πf₀t)
modified = ModifiedSignal(t, a, f0, composite)

print("Plotting y(t) = x(a*t) * e^(j2πf₀t)")
modified.plot("y(t) = x(a*t) * e^(j2πf₀t)")

# CFT of x(t) - use original CFTAnalyzer (for real signals)
print("Computing CFT of x(t)...")
cft_x = CFTAnalyzer(composite, t, frequencies)
spectrum_x = cft_x.compute_cft()
cft_x.plot_spectrum()

# CFT of y(t) - use ComplexCFTAnalyzer (for complex signals)
print("Computing CFT of y(t)...")
cft_y = ComplexCFTAnalyzer(modified, t, frequencies)
spectrum_y = cft_y.compute_cft()
cft_y.plot_spectrum()