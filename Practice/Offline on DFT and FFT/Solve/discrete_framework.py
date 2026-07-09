import numpy as np


class DiscreteSignal:
    """
    Represents a discrete-time signal as a numpy array of complex samples.
    """

    def __init__(self, data):
        self.data = np.array(data, dtype=np.complex128)

    def __len__(self):
        return len(self.data)

    def pad(self, new_length):
        """
        Zero-pad or truncate signal to new_length.
        Returns a new DiscreteSignal.
        """
        current = len(self.data)
        if new_length >= current:
            # Zero-pad at the end
            padded = np.zeros(new_length, dtype=np.complex128)
            padded[:current] = self.data
        else:
            # Truncate
            padded = self.data[:new_length].copy()
        return DiscreteSignal(padded)

    def interpolate(self, new_length):
        """
        Resample signal to new_length using linear interpolation.
        Used when FFT requires a power-of-2 size (Task 1 / Drawing App).
        """
        N = len(self.data)
        if N == new_length:
            return DiscreteSignal(self.data.copy())

        # Build old and new index arrays
        old_indices = np.arange(N, dtype=np.float64)
        new_indices = np.linspace(0, N - 1, new_length)

        # Interpolate real and imaginary parts separately
        real_interp = np.interp(new_indices, old_indices, self.data.real)
        imag_interp = np.interp(new_indices, old_indices, self.data.imag)

        return DiscreteSignal(real_interp + 1j * imag_interp)


class DFTAnalyzer:
    """
    Computes the Discrete Fourier Transform using the naive O(N^2) summation.
    """

    def compute_dft(self, signal: DiscreteSignal) -> np.ndarray:
        """
        DFT Analysis Equation:
            X[k] = sum_{n=0}^{N-1} x[n] * e^{-j*2*pi*k*n / N}
        Returns numpy array of complex frequency coefficients X[k].
        """
        x = signal.data
        N = len(x)
        X = np.zeros(N, dtype=np.complex128)
        for k in range(N):
            for n in range(N):
                X[k] += x[n] * np.exp(-1j * 2 * np.pi * k * n / N)
        return X

    def compute_idft(self, spectrum: np.ndarray) -> np.ndarray:
        """
        IDFT Synthesis Equation:
            x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * e^{j*2*pi*k*n / N}
        Returns numpy array of time-domain samples x[n].
        """
        X = np.array(spectrum, dtype=np.complex128)
        N = len(X)
        x = np.zeros(N, dtype=np.complex128)
        for n in range(N):
            for k in range(N):
                x[n] += X[k] * np.exp(1j * 2 * np.pi * k * n / N)
            x[n] /= N
        return x


class FastFourierTransform(DFTAnalyzer):
    """
    Fast Fourier Transform using the Radix-2 Decimation-In-Time (DIT)
    Cooley-Tukey algorithm. Achieves O(N log N) complexity.
    N must be a power of 2 — callers are responsible for padding/interpolation.
    """

    def _is_power_of_two(self, n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    def _fft_recursive(self, x: np.ndarray) -> np.ndarray:
        """
        Recursive Radix-2 DIT FFT.
        """
        N = len(x)
        if N == 1:
            return x.copy()

        # Split into even and odd indices
        even = self._fft_recursive(x[0::2])
        odd  = self._fft_recursive(x[1::2])

        # Twiddle factors: W_N^k = e^{-j*2*pi*k/N}
        k = np.arange(N // 2)
        twiddle = np.exp(-1j * 2 * np.pi * k / N) * odd

        X = np.empty(N, dtype=np.complex128)
        X[:N // 2] = even + twiddle
        X[N // 2:] = even - twiddle
        return X

    def _ifft_recursive(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse FFT via conjugate trick:
            IFFT(X) = conj(FFT(conj(X))) / N
        """
        N = len(X)
        # conjugate input
        x_conj = np.conj(X)
        # forward FFT on conjugated input
        result = self._fft_recursive(x_conj)
        # conjugate again and normalise
        return np.conj(result) / N

    def compute_dft(self, signal: DiscreteSignal) -> np.ndarray:
        """
        Compute FFT of signal. Pads to next power of 2 if needed.
        """
        x = signal.data
        N = len(x)
        if not self._is_power_of_two(N):
            next_pow2 = 1 << (N - 1).bit_length()
            padded_signal = signal.pad(next_pow2)
            x = padded_signal.data
        return self._fft_recursive(x)

    def compute_idft(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Compute IFFT of spectrum array.
        """
        X = np.array(spectrum, dtype=np.complex128)
        N = len(X)
        if not self._is_power_of_two(N):
            next_pow2 = 1 << (N - 1).bit_length()
            padded = np.zeros(next_pow2, dtype=np.complex128)
            padded[:N] = X
            X = padded
        return self._ifft_recursive(X)