import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

class FourierSeries:
    def __init__(self, func, L, terms=10):
        """
        Initialize the FourierSeries class with a target function, half period L, and number of terms.
        """
        # pass # Implement this method
        self.func=func
        self.L=L
        self.terms=terms

    def calculate_a0(self, N=1000):
        """
        Step 1: Compute the a0 coefficient using numerical integration.
        a0 = (1/2L) * integral(f(x), -L, L)
        """
        # # TODO: Implement numerical integration for a0
        # raise NotImplementedError("Implement calculate_a0")
        x=np.linspace(-self.L,self.L,N)
        y=self.func(x)
        dx=(2*self.L)/N
        a0=(1/(2*self.L))*np.trapezoid(y,x,dx)
        return a0

    def calculate_an(self, n, N=1000):
        """
        Step 2: Compute the an coefficient for the nth cosine term.
        an = (1/L) * integral(f(x) * cos(n*pi*x/L), -L, L)
        """
        # # TODO: Implement numerical integration for an
        # raise NotImplementedError("Implement calculate_an")
        x=np.linspace(-self.L,self.L,N)
        y=self.func(x)*np.cos(n*(np.pi/self.L)*x)
        dx=(2*self.L)/N
        an=(1/self.L)*np.trapezoid(y,x,dx)
        return an

    def calculate_bn(self, n, N=1000):
        """
        Step 3: Compute the bn coefficient for the nth sine term.
        bn = (1/L) * integral(f(x) * sin(n*pi*x/L), -L, L)
        """
        # # TODO: Implement numerical integration for bn
        # raise NotImplementedError("Implement calculate_bn")
        x=np.linspace(-self.L,self.L,N)
        y=self.func(x)*np.sin(n*(np.pi/self.L)*x)
        dx=(2*self.L)/N
        bn=(1/self.L)*np.trapezoid(y,x,dx)
        return bn

    def approximate(self, x):
        """
        Step 4: Use the calculated coefficients to build the Fourier series approximation.
        f(x) approx = a0/2 + sum(an*cos + bn*sin)
        """
        # # TODO: Implement the Fourier series summation
        # raise NotImplementedError("Implement approximate")
        a0=self.calculate_a0()
        approx=a0
        for n in range(1,self.terms+1):
            approx+=self.calculate_an(n)*np.cos(n*(np.pi/self.L)*x)
            approx+=self.calculate_bn(n)*np.sin(n*(np.pi/self.L)*x)
        return approx


    def plot(self, ax, wave_type="square"):
        """
        Step 5: Plot the original function and its Fourier series approximation.
        Now plots multiple periods.
        """
        x = np.linspace(-6,6,1000) if wave_type=="cubic" else np.linspace(-4*np.pi,4*np.pi,1000) #Implement this
        
        # Compute original function values
        original = target_function(x,wave_type) #Implement this
        
        # Compute Fourier series approximation
        approximation = self.approximate(x) #Implement this

        # Clear axis and Plotting
        ax.clear()
        ax.plot(x, original, label="Original Function", color="blue", alpha=0.5)
        ax.plot(x, approximation, label=f"Fourier Series (N={self.terms})", color="red", linestyle="--")
        
        # Dynamic Y-limits to ensure full view is seen for all wave types
        if wave_type == "sawtooth":
            # Sawtooth goes from -pi to +pi
            ax.set_ylim(-3.5, 3.5) 
        elif wave_type == "cubic":
            # Cubic x^3 on -1 to 1 ranges from -1 to 1.
            ax.set_ylim(-1.5, 1.5)
        elif wave_type == "pulse":
            ax.set_ylim(-0.5, 1.5)
        elif wave_type=="triangle":
            ax.set_ylim(-2.0,2.0)
        else:
            # Square, Triangle are roughly +/- 1
            ax.set_ylim(-1.5, 1.5)
            
        # Set X-limits to show multiple periods
        if wave_type == "cubic":
            ax.set_xlim(-6, 6)
        else:
            ax.set_xlim(-4 * np.pi, 4 * np.pi)
        
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_title(f"Fourier Series Approximation: {wave_type.replace('_', ' ').title()}")


def target_function(x, function_type="square"):
    """
    Defines target functions.
    """
    L=get_half_period(function_type)
    if function_type == "square":
        # Square wave: +1 when sin(x) > 0, -1 otherwise
        # pass
        y=np.where(np.sin(2*(np.pi/(2*L))*x)>0,1,-1)
        return y

    elif function_type == "sawtooth":
        # Mathematical Sawtooth: y = x for -pi < x < pi
        # pass
        y=(x+L)%(2*L)-L
        return y

    elif function_type == "triangle":
        # Mathematical Triangle Wave (Odd Function)
        # pass
        y=np.where((x+L/2)%(2*L)<L,(x+L/2)%(2*L),2*L-(x+L/2)%(2*L))-L/2
        return y

    elif function_type == "cubic":
        # Periodic Cubic: x^3 defined on -1 to 1, repeated.
        # pass
        y=((x+1)%(2*L)-L)**3 
        return y
    elif function_type == "pulse":
        # Pulse Train: A spike at 0 repeated every period (2*pi).
        # pass
        # y=np.where((x+L/2)%(2*L)<L,1,0)
        y=np.where((x+0.1)%(2*L)<0.2,1,0)
        return y

    else:
        raise ValueError("Invalid function_type.")


def get_half_period(wave_type):
    """
    Get the half-period L for different wave types.
    """
    # pass
    if wave_type=="cubic":
        return 1.0
    else:
        return np.pi


# Example of using these functions in the FourierSeries class with Sliders
if __name__ == "__main__":
    initial_terms = 1  # Start with 1 term
    initial_wave = "square"
    L = get_half_period(initial_wave)  # Half-period for initial function

    # Create the plot figure and axis
    fig_plot, ax_plot = plt.subplots(figsize=(10, 6))

    # Create the widgets figure
    fig_widgets = plt.figure(figsize=(8, 4))
    current_func = lambda x: target_function(x, initial_wave)
    fs = FourierSeries(current_func, L, initial_terms)
    # Pass the wave type to plot for correct axis scaling
    fs.plot(ax_plot, initial_wave)



    # 1. Radio Buttons (Positioned at the bottom left)
    ax_radio = fig_widgets.add_axes([0.05, 0.1, 0.15, 0.3], facecolor='#f0f0f0')
    radio = RadioButtons(ax_radio, ('square', 'sawtooth', 'triangle', 'cubic', 'pulse'))

    # 2. Slider Axes (Positioned above the radio buttons)
    ax_n = fig_widgets.add_axes([0.25, 0.6, 0.6, 0.1])

    # Slider for N (Number of terms)
    slider_n = Slider(
        ax=ax_n,
        label='Harmonics (N)',
        valmin=1,
        valmax=500,
        valinit=initial_terms,
        valstep=1
    )

    # --- Update Logic ---

    def update(val):
        """Callback when slider moves."""
        n = int(slider_n.val)
        wave_type = radio.value_selected
        
        # Update the FourierSeries object
        fs.terms = n
        fs.L = get_half_period(wave_type)
        fs.func = lambda x: target_function(x, wave_type)
        
        # Re-plot (Pass wave_type to handle axis scaling)
        fs.plot(ax_plot, wave_type)
        fig_plot.canvas.draw_idle()

    def change_wave(label):
        """Callback when radio button changes."""
        wave_type = label
        
        # Update the FourierSeries object
        fs.terms = int(slider_n.val) # Keep current N
        fs.L = get_half_period(wave_type)
        fs.func = lambda x: target_function(x, wave_type)
        
        # Re-plot (Pass wave_type to handle axis scaling)
        fs.plot(ax_plot, wave_type)
        fig_plot.canvas.draw_idle()

    # Connect widgets to functions
    slider_n.on_changed(update)
    radio.on_clicked(change_wave)

    plt.show()