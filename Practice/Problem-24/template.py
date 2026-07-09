import math
import cmath
import numpy as np

class DiscreteSignal:
    """
    Represents a discrete-time signal.
    """
    def __init__(self, data):
        # Ensure data is a numpy array, potentially complex
        self.data = np.array(data, dtype=np.complex128)

    def __len__(self):
        return len(self.data)
        
    def pad(self, new_length):
        """
        Zero-pad or truncate signal to new_length.
        Returns a new DiscreteSignal object.
        """
        # Implement padding logic
        new_data=np.zeros(new_length, dtype=np.complex128)
        if(np.size(self.data)>=new_length):
            new_data=self.data[:new_length]
        else:
            new_data[:np.size(self.data)]=self.data
        # Placeholder return to prevent crash
        return DiscreteSignal(new_data)

    def interpolate(self, new_length):
        """
        Resample signal to new_length using linear interpolation.
        Required for Task 4 (Drawing App).
        """
        # Implement interpolation logic
        current_length=np.size(self.data)
        current_axis=np.arange(0,current_length)
        new_axis=np.linspace(0,current_length-1,new_length)
        real_data=np.interp(new_axis,current_axis,np.real(self.data),0,0)
        imag_data=np.interp(new_axis,current_axis,np.imag(self.data),0,0)
        return DiscreteSignal(real_data+1j*imag_data)


class DFTAnalyzer:
    """
    Performs Discrete Fourier Transform using O(N^2) method.
    """
    def compute_dft(self, signal: DiscreteSignal):
        """
        Compute DFT using naive summation.
        Returns: numpy array of complex frequency coefficients.
        """
        N = len(signal)
        n=np.arange(0,N)
        # Implement Naive DFT equation
        X=[]
        for k in range(N):
            result=np.sum(signal.data*np.exp(-1j*2*np.pi*k*n/N))
            X.append(result)
        # Placeholder: Return zeros so UI doesn't crash
        return np.array(X)

    def compute_idft(self, spectrum):
        """
        Compute Inverse DFT using naive summation.
        Returns: numpy array (time-domain samples).
        """
        # Implement Naive IDFT equation
        N=np.size(spectrum)
        k=np.arange(N)
        x=[]
        for n in range(N):
            result=(1/N)*np.sum(spectrum*np.exp(1j*2*np.pi*k*n/N))
            x.append(result)
        return np.array(x)

#-----------------------Radix-2 Decimation-in-Time (DIT) Cooley-Tukey Algorithm------------------------#

class FastFourierTransform(DFTAnalyzer):
    def fft_helper(self,x:np.ndarray):
        N=len(x)
        if N==1:
            return np.copy(x)
        x_even=self.fft_helper(x[0::2])
        x_odd=self.fft_helper(x[1::2])
        k=np.arange(N//2)
        scaled_odd=np.exp(-1j*2*np.pi*k/N)*x_odd
        X=np.zeros_like(x,dtype=np.complex128)
        X[:N//2]=x_even+scaled_odd
        X[N//2:]=x_even-scaled_odd
        return X
    
    def ifft_helper(self,X:np.ndarray):
        return np.conj(self.fft_helper(np.conj(X)))/len(X)

    def compute_dft(self, signal:DiscreteSignal):
        N = len(signal.data)
        if (N>0 and (N & (N-1))==0):
            return self.fft_helper(signal.data)
        else:
            x:DiscreteSignal
            x=signal.interpolate(1<<N.bit_length())
            return self.fft_helper(x.data)
    
    def compute_idft(self, spectrum:np.ndarray):
        N = len(spectrum)
        if (N>0 and (N & (N-1))==0):
            return self.ifft_helper(spectrum)
        else:
            next_pow2=1<<N.bit_length()
            X=np.zeros(next_pow2, dtype=np.complex128)
            X[:N]=spectrum
            return self.ifft_helper(X)

#-----------------------Mixed-Radix FFT------------------------#

# class FastFourierTransform(DFTAnalyzer):
    
#     def small_dft(self, x: np.ndarray):
#         N = len(x)
#         n = np.arange(N)
#         X = []
#         for k in range(N):
#             result = np.sum(x * np.exp(-1j * 2 * np.pi * k * n / N))
#             X.append(result)
#         return np.array(X)

#     def fft_helper(self, x: np.ndarray):
#         N = len(x)

#         if N <= 1:
#             return np.copy(x)
#         if N <= 4: 
#             return self.small_dft(x)
        
#         r = self.smallest_factor(N)

#         if r == N:
#             return self.small_dft(x)

#         m = N // r

#         sub = [self.fft_helper(x[i::r]) for i in range(r)]

#         X = np.zeros(N, dtype=np.complex128)
#         twiddles = np.exp(-1j * 2 * np.pi * np.outer(np.arange(N), np.arange(r)) / N)
#         for k in range(N):
#             for i in range(r):
#                 X[k] += twiddles[k, i] * sub[i][k % m]
#         return X

#     def smallest_factor(self, N: int):
#         if N % 2 == 0:
#             return 2
#         i = 3
#         while i * i <= N:
#             if N % i == 0:
#                 return i
#             i += 2
#         return N 

#     def ifft_helper(self, X: np.ndarray) :
#         return np.conj(self.fft_helper(np.conj(X))) / len(X)

#     def compute_dft(self, signal: DiscreteSignal) :
#         return self.fft_helper(signal.data)

#     def compute_idft(self, spectrum: np.ndarray):
#         return self.ifft_helper(spectrum)


def weighted_polynomial_multiply(P, Q, W):
    P=np.array(P)
    Q=np.array(Q)
    W=np.array(W)
    total_len=len(P)+len(Q)-1
    weighted_P=P*W
    weighted_P=DiscreteSignal(weighted_P).pad(total_len).data
    Q=DiscreteSignal(Q).pad(total_len).data
    dftAnalyzer=DFTAnalyzer()
    P_k=dftAnalyzer.compute_dft(DiscreteSignal(weighted_P))
    Q_k=dftAnalyzer.compute_dft(DiscreteSignal(Q))
    return np.round(np.real(dftAnalyzer.compute_idft(P_k*Q_k))).astype(np.int64)
    

if __name__ == "__main__":
    # P = [1, 3, 2, 6, 7]
    # Q = [4,1]
    # W = [3, 2, 1, 5, 6]

    P = [1, 3, 2]
    Q = [4,1]
    W = [3, 2, 1]
 

    R = weighted_polynomial_multiply(P, Q, W)

    print("Result:", R)