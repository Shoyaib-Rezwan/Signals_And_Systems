import numpy as np
import matplotlib.pyplot as plt

t=np.linspace(-1,1,1000)

# original signal
x_t=2*np.sin(14*np.pi*t)-np.sin(2*np.pi*t)*(4*np.sin(2*np.pi*t)*np.sin(14*np.pi*t)-1)

plt.plot(t,x_t)
plt.title("Original Signal")
plt.grid(True,alpha=0.2)
plt.plot()
plt.show()

#Fourier Transform
freq=np.arange(-10,11)
X_f=[]
for f in freq:
    result=np.trapezoid(x_t*np.exp(-1j*2*np.pi*f*t),t)
    X_f.append(result)
X_f=np.array(X_f)

plt.stem(freq,np.abs(X_f))
plt.title("Fourier Transform")
plt.plot()
plt.grid(True,alpha=0.2)
plt.show()

#recostruction
x_t1=np.sin(2*np.pi*t)
x_t2=np.sin(10*np.pi*t)
x_t3=np.sin(18*np.pi*t)

plt.plot(t,x_t,label="original")
plt.plot(t,x_t1+x_t2+x_t3,linestyle="--",label="Reconstructed")
plt.title("Original + Reconstructed Signal")
plt.grid(True,alpha=0.2)
plt.legend()
plt.plot()
plt.show()