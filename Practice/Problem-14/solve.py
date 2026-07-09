import numpy as np
import matplotlib.pyplot as plt
class Signal:
    def __init__(self,func):
        self.func=func

    def plot(self,t,title):
        x_t=self.func(t)
        plt.plot(t,x_t)
        plt.title(title)
        # plt.legend(title)
        plt.grid(True,alpha=0.2)
        plt.show()
    def calculateParseval(self,t):
        x_t=self.func(t)
        return np.trapezoid(x_t**2,t)
class CFTAnalyzer:
    def __init__(self,signal:Signal,t,freq):
        self.signal=signal
        self.t=t
        self.freq=freq
    
    def compute_cft(self):
        x_t=self.signal.func(self.t)
        X_f=[]
        for f in self.freq:
            result=np.trapezoid(x_t*np.exp(-1j*2*np.pi*f*self.t),self.t)
            X_f.append(result)
        X_f=np.array(X_f)
        return X_f

    def plot(self,title):
        X_f=self.compute_cft()
        plt.plot(self.freq, np.abs(X_f))
        plt.title(title)
        plt.grid(True,alpha=0.2)
        plt.show()

    def calculateParseval(self):
        X_f = self.compute_cft() 
        return np.trapezoid(np.abs(X_f)**2, self.freq)

def func(t):
    y=np.zeros_like(t)
    mask1=((t>=-3) & (t<=-1)) | ((t>=1) & (t<=3))
    y[mask1]=(3-np.abs(t[mask1]))**2
    mask2=((t>-1)&(t<1))
    y[mask2]=5-abs(t[mask2])
    return y
def main():
    t=np.linspace(-10,10,1000)
    x_t=Signal(func)
    x_t.plot(t,"Piecewise Function")
    freq=np.linspace(-10,10,1000)
    cftAnalyzer= CFTAnalyzer(x_t,t,freq)
    cftAnalyzer.plot("Fourier Transform")
    print("Parsevals value for time domain signal: ",x_t.calculateParseval(t))
    print("Parsevals value for frequency domain signal: ",cftAnalyzer.calculateParseval())
main()