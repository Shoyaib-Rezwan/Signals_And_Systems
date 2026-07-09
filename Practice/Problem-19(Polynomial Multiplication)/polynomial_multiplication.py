import numpy as np
class DiscreteSignal:
    def __init__(self,data):
        self.data=np.array(data, dtype=np.complex128)

    def zero_pad(self,new_length):
        new_data=np.zeros(new_length)
        current_length=np.size(self.data)
        if new_length==current_length:
            new_data=np.copy(self.data)
        elif new_length>current_length:
            new_data[:current_length]=self.data
        else:
            new_data=self.data[:new_length]
        return DiscreteSignal(new_data)
    
    def interpolate(self,new_length):
        new_data=np.zeros(new_length)
        current_length=np.size(self.data)
        if current_length==new_length:
            new_data=np.copy(self.data)
        else:
            current_axis=np.arange(0,current_length)
            new_axis=np.arange(0,new_length)
            new_data=np.interp(new_axis,current_axis,self.data)
        return DiscreteSignal(new_data)

class DFTAnalyzer:
    def compute_dft(self,signal:DiscreteSignal)->np.ndarray:
        X=[]
        N=np.size(signal.data)
        n=np.arange(0,N)
        for k in range(0,N):
            result=np.sum(signal.data*np.exp(-1j*2*np.pi*n*k/N))
            X.append(result)
        return np.array(X)
    def compute_idft(self, X:np.ndarray)->np.ndarray:
        x=[]
        N=np.size(X)
        k=np.arange(0,N)
        for n in range(0,N):
            result=1/N*np.sum(X*np.exp(1j*2*np.pi*k*n/N))
            x.append(result)
        return np.array(x)
    
class FastFourierTransform(DFTAnalyzer):
    def fft_helper(self, x:np.ndarray):
        N=np.size(x)
        if N==1:
            return np.copy(x).astype(np.complex128)
        G=self.fft_helper(x[0::2])
        H=self.fft_helper(x[1::2])
        X=np.zeros(N,dtype=np.complex128)
        k=np.arange(0,N//2)
        scaled_H=np.exp(-1j*2*np.pi*k/N)*H
        X[0:N//2]=G+scaled_H
        X[N//2:]=G-scaled_H
        return X
    def compute_dft(self, signal):
        if len(signal.data) & (len(signal.data)-1)==0:
            return self.fft_helper(signal.data)
        nearest_two=1<<len(signal.data).bit_length()
        return self.fft_helper(signal.zero_pad(nearest_two).data)
        
def circular_convolution(x:np.ndarray, h:np.ndarray):
    N=np.size(x)
    y=[]
    for n in range(0,N):
        result=0
        for k in range (0,N):
            result+=x[k]*h[(n-k)%N]
        y.append(result)
    return y

def circular_convolution_by_dft(x:np.ndarray, h:np.ndarray):
    # dftAnalyzer=DFTAnalyzer()
    dftAnalyzer=FastFourierTransform()
    X=dftAnalyzer.compute_dft(DiscreteSignal(x))
    H=dftAnalyzer.compute_dft(DiscreteSignal(h))
    return dftAnalyzer.compute_idft(X*H)

def main():
#input coefficients of 2 polynomials in descending order of degree(i.e. 2x^2+5x+3 then [2,5,3])
    coeff1=list(map(int,input().split()))
    coeff2=list(map(int,input().split()))
    coeff1=coeff1[::-1]
    coeff2=coeff2[::-1]
    total_len=len(coeff1)+len(coeff2)-1
    sig1=DiscreteSignal(coeff1).zero_pad(total_len)
    sig2=DiscreteSignal(coeff2).zero_pad(total_len)
    # result=np.real(circular_convolution(sig1.data, sig2.data))[::-1].astype(np.int64)
    result=np.real(circular_convolution_by_dft(sig1.data, sig2.data))[::-1]
    print(result)

main()