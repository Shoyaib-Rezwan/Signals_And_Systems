import numpy as np
import matplotlib.pyplot as plt

class DiscontiuousSignal:
    def __init__(self,data):
        self.data=np.array(data, dtype=np.complex128)

    def len(self):
        return np.size(self.data)
    
    def zeropad(self, new_length):
        newdata=np.zeros(new_length, dtype=np.complex128)
        current_length=np.size(self.data)
        if new_length==current_length:
            newdata=np.copy(self.data)
        elif new_length>current_length:
            newdata[:current_length]=self.data
        else:
            newdata=self.data[:new_length]
        return DiscontiuousSignal(newdata)
    
    def interpolate(self, new_length):
        newdata=np.zeros(new_length, dtype=np.complex128)
        current_length=np.size(self.data)
        if current_length==new_length:
            newdata=np.copy(self.data)
        else:
            current_axis=np.arange(0,current_length)
            new_axis=np.arange(0,new_length)
            newdata=np.interp(new_axis,current_axis,np.real(np.data),0,0)+1j*np.interp(new_axis,current_axis,np.real(np.data),0,0)
        return DiscontiuousSignal(newdata)

class DFTAnalyzer:
    def compute_dft(self, signal:DiscontiuousSignal)->np.ndarray:
        N=signal.len()
        n=np.arange(0,N)
        X=[]
        for k in range(0,N):
            X.append(np.sum(signal.data*np.exp(-1j*2*np.pi*k*n/N)))
        return np.array(X, dtype=np.complex128)
    
    def compute_idft(self, spectrum: np.ndarray)->np.ndarray:
        N=np.size(spectrum)
        k=np.arange(0,N)
        x=[]
        for n in range(0,N):
            x.append(np.sum(spectrum*np.exp(1j*2*np.pi*k*n/N))/N)
        return np.array(x,dtype=np.complex128)
    
def stem_plot(x, title="Title", x_label="index", y_label="value"):
    x=np.array(x)
    N=len(x)
    n=np.arange(0,N)
    plt.stem(n,x)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.2)

def plot_spectrum(X, prefix=""):
    mag=np.abs(X)
    phase=np.angle(X)
    N=len(X)
    k=np.arange(0,N)

    plt.stem(k,mag)
    plt.title(f"{prefix}magnitude: |X[k]|")
    plt.xlabel("k")
    plt.ylabel("|X[k]|")
    plt.grid(True, alpha=0.2)
    plt.show()

    plt.stem(k,phase)
    plt.title(f"{prefix}phase: ∠X[k]")
    plt.xlabel("k")
    plt.ylabel("∠X[k] (rad)")
    plt.grid(True, alpha=0.2)

def plot_comparision(x:np.ndarray, y: np.ndarray, title="Comparision",legend1="x", legend2="y"):
    N=len(x)
    n=np.arange(0,N)
    plt.stem(n,x,label=legend1)
    plt.stem(n,y,label=legend2, linefmt="r--")
    plt.title(title)
    plt.xlabel("index")
    plt.ylabel("value")
    plt.show()

def max_abs_error(a:np.ndarray, b:np.ndarray)->np.ndarray:
    return max(np.abs(a-b))

def circular_convolution(x:np.ndarray, h:np.ndarray)->np.ndarray:
    y=[]
    N=len(x)
    for n in range(0,N):
        result=0
        for k in range(0,N):
            result+=x[k]*h[(n-k)%N]
        y.append(result)
    return np.array(y)

def circular_convolution_via_dft(x:np.ndarray, h:np.ndarray)->np.ndarray:
    dftAnalyzer=DFTAnalyzer()
    x_sig=DiscontiuousSignal(x)
    h_sig=DiscontiuousSignal(h)
    X=dftAnalyzer.compute_dft(x_sig)
    H=dftAnalyzer.compute_dft(h_sig)
    return dftAnalyzer.compute_idft(X*H)

def cross_correlation(x:np.ndarray, h:np.ndarray)->np.ndarray:
    y=[]
    N=len(x)
    for k in range(0,N):
        result=0
        for n in range(0,N):
            result+=x[n]*np.conj(h[(n-k)% N])
        y.append(result)
    return np.array(y)

def cross_correlation_via_dft(x:np.ndarray, h:np.ndarray)->np.ndarray:
    dftAnalyzer=DFTAnalyzer()
    x_sig=DiscontiuousSignal(x)
    h_sig=DiscontiuousSignal(h)
    X=dftAnalyzer.compute_dft(x_sig)
    H=dftAnalyzer.compute_dft(h_sig)
    return dftAnalyzer.compute_idft(X*np.conj(H))

def main():
    dftAnalyzer=DFTAnalyzer()

    #rectangual pulse
    N=64
    m=5
    ns=12

    data=np.zeros(N)
    data[0:N//8]=1
    x_rect=DiscontiuousSignal(data)
    stem_plot(x_rect.data,"Rectangular Pulse")
    plt.show()
    X=dftAnalyzer.compute_dft(x_rect)
    plot_spectrum(X,"Rectangular Pulse")
    plt.show()
    x=dftAnalyzer.compute_idft(X)
    print("Max Absolute Error: ", max_abs_error(x_rect.data,x))
    n=np.arange(0,N)
    plt.stem(n,np.real(x))
    plt.stem(n,x_rect.data, linefmt="r--")
    plt.title("Rectangular Pulse Reconstruction")
    plt.grid(True, alpha=0.2)
    plt.show()
    
    #cos signal
    x_cos=DiscontiuousSignal(np.cos(2*np.pi*m/N*np.arange(0,N)))
    stem_plot(x_cos.data, "Cosine Signal")
    plt.show()
    X=dftAnalyzer.compute_dft(x_cos)
    plot_spectrum(X,"Cosine Signal")
    plt.show()

    x=dftAnalyzer.compute_idft(X)
    print("Max Absolute Error: ", max_abs_error(x_cos.data,x))
    plt.stem(n,np.real(x))
    plt.stem(n,x_cos.data, linefmt="r--")
    plt.title("Cosine Signal Reconstruction")
    plt.grid(True, alpha=0.2)
    plt.show()

    #cicular convolution
    x=[1,2,3,4]
    h=[4,3,2,1]
    conv1=circular_convolution(x,h)
    conv2=circular_convolution_via_dft(x,h)
    plot_comparision(conv1,conv2,"Circular Convolution","Conv1","Conv2")

    #cross correlation
    x=x_cos.data
    y=np.roll(x,ns)
    cc1=cross_correlation(x,y)
    cc2=cross_correlation_via_dft(x,y)
    plot_comparision(cc1,cc2,"Cross Correlation","cc1", "cc2")

def main2():
    x=[1,2,3,4,0]
    dftAnalyzer=DFTAnalyzer()
    print(np.real(dftAnalyzer.compute_dft(DiscontiuousSignal(x))))
main2()
