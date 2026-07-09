import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

class FourierSeries:
    def __init__(self,func,L,terms=10):
        self.func=func
        self.L=L
        self.terms=terms
    def calculate_a0(self,N=1000):
        t=np.linspace(-self.L,self.L,N)
        x_t=self.func(t)
        d_t=2*self.L/N
        a0=1/(2*self.L)*np.trapezoid(x_t,t,d_t)
        return 0
    def calculate_an(self,n,N=1000):
        t=np.linspace(-self.L,self.L, N)
        x_t=self.func(t)
        dt=2*self.L/N
        an=(1/self.L)*np.trapezoid(x_t*np.cos(n*(np.pi/self.L)*t),t,dt)
        return 0
    def calculate_bn(self,n,N=1000):
        t=np.linspace(-self.L,self.L, N)
        x_t=self.func(t)
        dt=2*self.L/N
        bn=(1/self.L)*np.trapezoid(x_t*np.sin(n*(np.pi/self.L)*t),t,dt)
        return 3/(n*np.pi)*(1-(-1)**n)
    def approx(self,t):
        a0=self.calculate_a0()
        x_t=a0
        for i in range(1,self.terms+1):
            an=self.calculate_an(i)
            bn=self.calculate_bn(i)
            x_t=x_t+an*np.cos(i*(np.pi/self.L)*t)+bn*np.sin(i*np.pi/self.L*t)
        return x_t
    def plot(self,ax:plt.Axes,INF):
        N=1000
        t=np.linspace(-INF,INF,N)
        x_t=self.func(t)
        ax.clear()
        ax.plot(t,x_t)
        x_t=self.approx(t)
        ax.plot(t,x_t,linestyle="--")
        ax.grid(True,alpha=0.5)
      

def func(t): #L is half period
    x_t=np.where(t%(2*L)<L,1.5,-1.5)
    return x_t

def main():
    initial_terms = 1
    global L, INF 
    
    L = int(input("Input half period (L): "))
    INF = int(input("Input range (INF): "))
    
    fs = FourierSeries(func, L, initial_terms)
    fig_plot, ax_plot = plt.subplots(figsize=(10, 6))
    # fig_widgets = plt.figure(figsize=(8, 4))
    fs.plot(ax_plot,INF)
    # Define the slider axes [left, bottom, width, height]
    ax_n = fig_plot.add_axes([0.25, 0, 0.6, 0.05])
    slider_n = Slider(
        ax=ax_n,
        label='Harmonics (N) ',
        valmin=1,
        valmax=100,
        valinit=initial_terms,
        valstep=1
    )

    def update(val):
        """Callback when slider moves."""
        n = int(slider_n.val)
        # wave_type = radio.value_selected
        
        # Update the FourierSeries object
        fs.terms = n
        fs.L = L
        fs.func = func
        
        # Re-plot (Pass wave_type to handle axis scaling)
        fs.plot(ax_plot,INF)
        fig_plot.canvas.draw_idle()

    # Initial call to populate the plot
    update(initial_terms)

    slider_n.on_changed(update)
    plt.show()

if __name__ == "__main__":
    main()
