import numpy as np
import matplotlib.pyplot as plt

class DiscreteSignal:
    def __init__(self,INF):
        self.INF=INF
        self.values=np.zeros(2*INF+1)
    def set_signal_range(self,newINF): #modifies the signal range from -newINF to +newINF
        if newINF<0:
            print('Invalid range')
            return self
        if newINF==self.INF:
            return self
        temp_signal=DiscreteSignal(newINF)
        if newINF<self.INF:
            temp_signal.values=self.values[self.INF-newINF:self.INF-newINF+2*newINF+1]
        elif newINF>self.INF:
            temp_signal.values[newINF-self.INF:newINF-self.INF+2*self.INF+1]=self.values
        return temp_signal
    def set_value_at_time(self,time,value):
        if(np.abs(time)>self.INF):
            self.set_signal_range(np.abs(time))
        self.values[self.INF+time]=value
    def shift_signal(self,shift):
        temp=DiscreteSignal(self.INF+np.abs(shift))
        if shift==0:
            temp.values=np.copy(self.values)
        if shift>0:
            temp.values[2*shift:2*shift+2*self.INF+1]=self.values
        if shift<0:
            temp.values[0:2*self.INF+1]=self.values
        return temp
    def add(self,other):
        newINF=np.max([self.INF,other.INF])
        signal1= self.set_signal_range(newINF)
        signal2= other.set_signal_range(newINF)
        result= DiscreteSignal(newINF)
        result.values=signal1.values+signal2.values
        return result
    def multiply(self,other):
        newINF=np.max([self.INF,other.INF])
        signal1= self.set_signal_range(newINF)
        signal2= other.set_signal_range(newINF)
        result= DiscreteSignal(newINF)
        result.values=signal1.values*signal2.values
        return result
    def multiply_const_scalar(self,scalar):
        result=DiscreteSignal(self.INF)
        result.values=self.values*scalar
        return result
    def plot(self, title=""):
        times=np.arange(-self.INF,self.INF+1)
        plt.stem(times,self.values)
        plt.xlabel("Time")
        plt.ylabel("x[n]")
        plt.title(title)
        plt.grid(True,alpha=0.2)
        # plt.show()

class LTI_Discrete_class:
    def __init__(self,impulse_response: DiscreteSignal):
        self.impulse_response=impulse_response
    def linear_combination_of_impulses(self,input_signal):
        impulses=[]
        coefficients= input_signal.values
        for i in range(0,2*input_signal.INF+1):
            impulse=DiscreteSignal(self.INF)
            impulse.values[i]=1
            impulses.append(impulse)
        return impulses,coefficients
    def output(self,input_signal):
        output_signal=None
        for k in range(-input_signal.INF, input_signal.INF+1):
            shifted_impulse=self.impulse_response.shift_signal(k)
            scaled_signal=shifted_impulse.multiply_const_scalar(input_signal.values[k+input_signal.INF])
            if output_signal:
                output_signal=output_signal.add(scaled_signal)
            else:
                output_signal=scaled_signal
        return output_signal

# Stock Market Prices as a Python List
# price_list = list(map(int, input("Stock Prices: ").split()))
# n = int(input("Window size: "))
# alpha = float(input("Alpha: "))

# You may use the following input for testing purpose
price_list = [10,11,12,9,10,13,15,16,17,18]
n = 3
alpha = 0.8
impulse_response=DiscreteSignal(len(price_list)-1)

for i in range(len(price_list)):
    impulse_response.set_value_at_time(i,price_list[i])
# impulse_response.plot()

input_signal=DiscreteSignal(n-1)
for k in range(n):
    input_signal.set_value_at_time(k,alpha*((1-alpha)**k))
# input_signal.plot()
# plt.show()
system=LTI_Discrete_class(input_signal)
output_signal=system.output(impulse_response)
output_signal.plot()
print(output_signal.values[output_signal.INF+n-1:output_signal.INF+len(price_list)])
plt.show()


# Determine the values after performing Exponential Smoothing
# The length of exsm should be = len(price_list) - n + 1
exsm = []

print("Exponential Smoothing: " + ", ".join(f"{num:.2f}" for num in exsm))
# Output should be: 11.68, 9.47, 9.82, 12.29, 14.40, 15.62, 16.64, 17.63