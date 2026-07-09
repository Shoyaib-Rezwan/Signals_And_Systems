import numpy as np
import matplotlib.pyplot as plt
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

def cross_correlation(x:np.ndarray, h:np.ndarray)->np.ndarray:
    dftAnalyzer=DFTAnalyzer()
    X=dftAnalyzer.compute_dft(DiscreteSignal(x))
    H=dftAnalyzer.compute_dft(DiscreteSignal(h))
    return dftAnalyzer.compute_idft(X*np.conj(H))

image = plt.imread("image.png")
shifted_image = plt.imread("shifted_image.png")

plt.figure(figsize=(12, 8))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Shifted Image
plt.subplot(2, 3, 2)
plt.imshow(shifted_image, cmap='gray')
plt.title(f"Shifted Image")
plt.axis('off')

reversed_shifted_image=np.copy(shifted_image)
row_size, col_size=np.shape(reversed_shifted_image)
# print(row_size, col_size)
best_row_idx=np.argmax(np.var(image,axis=1)+np.var(shifted_image, axis=1))
cc=cross_correlation(image[best_row_idx,:], shifted_image[best_row_idx,:])
shift_amt=np.argmax(cc)
if shift_amt>col_size//2:
    shift_amt-=col_size
print(shift_amt)
for k in range(row_size):
    reversed_shifted_image[k,:]=np.roll(reversed_shifted_image[k,:],shift_amt)

best_col_idx=np.argmax(np.var(image,axis=0)+np.var(shifted_image, axis=0))
cc=cross_correlation(image[:,best_col_idx], shifted_image[:,best_col_idx])
shift_amt=np.argmax(cc)
if shift_amt>row_size//2:
    shift_amt-=row_size
print(shift_amt)
for k in range(col_size):
    reversed_shifted_image[:,k]=np.roll(reversed_shifted_image[:,k],shift_amt)

plt.subplot(2, 3, 3)
plt.imshow(reversed_shifted_image, cmap='gray')
plt.title("Reversed Shifted Image")
plt.axis('off')

plt.tight_layout()
plt.show()