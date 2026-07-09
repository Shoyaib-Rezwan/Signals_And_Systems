import numpy as np
import matplotlib.pyplot as plt

class Signal:
    def __init__(self, INF):
        self.INF=INF
        self.n=np.arange(-INF,INF+1,1)
        self.x=np.astype(np.zeros_like(self.n),float)

    def set_value_at_time(self, t, value):
        self.x[t+self.INF]=value

    def shift(self, k):
        # Shift the signal and return the resultant signal
        result=Signal(self.INF+np.abs(k))
        for n,x in zip(self.n,self.x):
            result.set_value_at_time(n+k,x)
        return result
            

    def add(self, other):
        # Add two signals and return the resultant signal
        result=Signal(np.max(np.array([self.INF,other.INF])))
        temp_x1=Signal(result.INF)
        temp_x2=Signal(result.INF)
        for n, x in zip(self.n,self.x):
            temp_x1.set_value_at_time(n,x)
        for n, x in zip(other.n,other.x):
            temp_x2.set_value_at_time(n,x)
        for n,x1,x2 in zip(result.n,temp_x1.x,temp_x2.x):
            result.set_value_at_time(n,x1+x2)
        return result

    def multiply(self, scalar):
        # Multiply a constant value with the signal
        result=Signal(self.INF)
        result.x=self.x*scalar
        return result

    def plot(self, title="Discrete Signal"):
        plt.stem(self.n,self.x)
        plt.title(title)
        plt.xlabel("n")
        plt.ylabel("x[n]")
        # plt.xlim(-self.INF-1, self.INF+1)
        # plt.xticks(self.n)
        plt.grid(True,alpha=0.2)
        # plt.savefig(f"{title}.png")
        plt.show()

class LTI_System:
    def __init__(self, impulse_response: Signal):
        self.impulse_response=impulse_response

    def linear_combination_of_impulses(self, input_signal: Signal):
        impulses=[]
        coefficients=input_signal.x

        for n in input_signal.n:
            imp=Signal(input_signal.INF)
            imp.set_value_at_time(n,1)
            impulses.append(imp)
        return impulses,coefficients

    def output(self, input_signal: Signal):
        impulses,x=self.linear_combination_of_impulses(input_signal)
        y=None
        for k,x_k in zip(input_signal.n, x):
            shifted_h=self.impulse_response.shift(k)
            scaled_h=shifted_h.multiply(x_k)
            y= y.add(scaled_h) if y else scaled_h

        return y

# Todo: Define Signal class
        
class SuperSignal:
    def __init__(self):
        self.components = []

    def add(self, signal: Signal, coefficient=1.0):
        self.components.append((coefficient, signal))
        
# Todo: Define LTI class

if __name__ == "__main__":
    INF = 10

    # Component signals
    x1 = Signal(INF)
    x1.set_value_at_time(0, 1)

    x2 = Signal(INF)
    x2.set_value_at_time(2, 1)

    # Todo: Create SuperSignal: x(n) = 2*x1(n) - x2(n)
    superSignal=SuperSignal()
    superSignal.add(x1,2)
    superSignal.add(x2,-1)

    # Impulse response
    h = Signal(INF)
    h.set_value_at_time(0, 1)
    h.set_value_at_time(1, 0.5)

    system = LTI_System(h)

    # Todo: Output using superposition
    output_signal=None
    for component in superSignal.components:
        output_signal:Signal
        component: tuple[float,Signal]
        input_signal=component[1].multiply(component[0])
        system_response=system.output(input_signal)
        if output_signal:
            output_signal=output_signal.add(system_response)
        else:
            output_signal=system_response
    output_signal.plot()
    
