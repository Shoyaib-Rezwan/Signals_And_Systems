import matplotlib.pyplot as plt
import numpy as np

class Signal:
    def __init__(self, start_time=-100, end_time=100):
        # Initializes a signal with all values set to zero [cite: 10, 11, 12]
        self.indices = np.arange(start_time, end_time + 1)
        self.values = np.zeros(len(self.indices))

    def set_value_at_time(self, t, value):
        # Sets the signal value at time index t [cite: 14]
        if t in self.indices:
            idx = np.where(self.indices == t)[0][0]
            self.values[idx] = value

    def shift(self, k):
        # Returns a new Signal object corresponding to x(n-k) [cite: 15, 16]
        new_signal = Signal(self.indices[0] + k, self.indices[-1] + k)
        new_signal.values = self.values.copy()
        return new_signal

    def add(self, other):
        # Returns the sum of two signals y(n) = x1(n) + x2(n) [cite: 17, 18]
        start = min(self.indices[0], other.indices[0])
        end = max(self.indices[-1], other.indices[-1])
        new_signal = Signal(start, end)
        
        for i, t in enumerate(self.indices):
            new_signal.set_value_at_time(t, self.values[i])
        for i, t in enumerate(other.indices):
            existing = new_signal.values[np.where(new_signal.indices == t)[0][0]]
            new_signal.set_value_at_time(t, existing + other.values[i])
        return new_signal

    def multiply(self, scalar):
        # Returns a scaled signal y(n) = a * x(n) [cite: 19, 20]
        new_signal = Signal(self.indices[0], self.indices[-1])
        new_signal.values = self.values * scalar
        return new_signal

    def plot(self, title):
        # Produces a stem plot of the discrete-time signal [cite: 21]
        plt.stem(self.indices, self.values)
        plt.title(title)
        plt.xlabel('n')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

class LTI_System:
    def __init__(self, impulse_response):
        # Models a discrete-time LTI system using its impulse response h(n) [cite: 23, 24, 26]
        self.h = impulse_response

    def linear_combination_of_impulses(self, input_signal):
        # Decomposes input signal into impulses and coefficients [cite: 28, 29, 30]
        impulses = []
        coefficients = []
        for i, val in enumerate(input_signal.values):
            if val != 0:
                time_index = input_signal.indices[i]
                # Unit impulse delta(n - time_index)
                impulse = Signal(time_index, time_index)
                impulse.set_value_at_time(time_index, 1)
                impulses.append(impulse)
                coefficients.append(val)
        return impulses, coefficients 

    def output(self, input_signal):
        # Computes output using LTI property: y(n) = sum x(k)h(n-k) [cite: 32, 33, 34]
        impulses, coefficients = self.linear_combination_of_impulses(input_signal) 
        output_signal = None
        
        for k, x_k in zip([imp.indices[0] for imp in impulses], coefficients):
            # LTI property: shifted response is h(n-k) scaled by x(k)
            shifted_h = self.h.shift(k)
            scaled_h = shifted_h.multiply(x_k)
            
            if output_signal is None:
                output_signal = scaled_h
            else:
                output_signal = output_signal.add(scaled_h)
        return output_signal