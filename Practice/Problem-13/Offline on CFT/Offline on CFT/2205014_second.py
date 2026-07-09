import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

# =====================================================
# Continuous Image Class
# =====================================================
class ContinuousImage:
    """
    Represents an image as a continuous 2D signal.
    """

    def __init__(self, image_path):
        self.image = imread(image_path, mode='L')
        self.image = self.image / np.max(self.image)

        # Define continuous spatial axes
        self.x = np.linspace(-1, 1, self.image.shape[1])
        self.y = np.linspace(-1, 1, self.image.shape[0])

    def show(self, title="Image"):
        plt.imshow(self.image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()


# =====================================================
# 2D Continuous Fourier Transform Class
# =====================================================
class CFT2D:
    """
    Computes 2D Continuous Fourier Transform
    using separability and numerical integration.
    """

    def __init__(self, image_obj:ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y

    def compute_cft(self):
        """
        Compute real and imaginary parts of 2D CFT.
        """
        row_size, col_size = self.I.shape

        F = np.zeros((row_size, col_size), dtype=complex)

        expY = np.exp(-2j * np.pi * self.y[:, None] * self.y[None, :])

        expX = np.exp(-2j * np.pi * self.x[:, None] * self.x[None, :])

        Gx= np.trapezoid(self.I[:, :, None] * expY[:, None, :],self.y,axis=0) 

        for u in range(row_size):
            F[u, :] = np.trapezoid(Gx[:, u][:, None] * expX,self.x,axis=0)

        return F.real, F.imag
        
                


    def plot_magnitude(self):
        """
        Plot log-scaled magnitude spectrum.
        """
        real, imaginary = self.compute_cft()
        magnitude = np.log(1 + np.abs(real + 1j*imaginary))

        plt.imshow(magnitude, cmap='magma')
        plt.title("2D CFT Magnitude Spectrum")
        plt.axis('off')
        # plt.savefig('spectrum.png')
        plt.show()


# =====================================================
# Frequency Filtering
# =====================================================
class FrequencyFilter:
    def low_pass(self, real, imag, cutoff):
        rows, cols = real.shape
        cx, cy = rows//2, cols//2

        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i-cx)**2 + (j-cy)**2) > cutoff:
                    real[i,j] = 0
                    imag[i,j] = 0
        return real, imag

# =====================================================
# Inverse 2D Continuous Fourier Transform
# =====================================================
class InverseCFT2D:
    """
    Reconstructs image from 2D frequency spectrum.
    """

    def __init__(self, real, imag, x, y):
        self.real=real 
        self.imag = imag
        self.x = x
        self.y = y

    def reconstruct(self):
        """
        Perform inverse 2D CFT using numerical integration.
        """
        row_size, col_size = self.real.shape
        spectrum = self.real + 1j * self.imag
    
        reconstructed_img = np.zeros((row_size, col_size))

        expU = np.exp(1j * 2 * np.pi * self.y[:, None] * self.y[None, :])
        
        expV = np.exp(1j * 2 * np.pi * self.x[:, None] * self.x[None, :])
    
        Gx = np.trapezoid(spectrum[:, None, :] * expU[:, :, None],self.y,axis=0)
    
        for i in range(row_size):
            reconstructed_img[i, :] = np.trapezoid(Gx[i, :, None] * expV,self.x, axis=0).real

        return reconstructed_img


# =====================================================
# Main Execution (Task 2)
# =====================================================
img = ContinuousImage("noisy_image.png")
img.show("Original Image")

cft2d = CFT2D(img)
real, imag = cft2d.compute_cft()
cft2d.plot_magnitude()

filt = FrequencyFilter()
real_f, imag_f = filt.low_pass(real, imag, cutoff=40)

icft2d = InverseCFT2D(real_f, imag_f, img.x, img.y)
denoised = icft2d.reconstruct()

plt.imshow(denoised, cmap='gray')
plt.title("Reconstructed (Denoised) Image")
plt.axis('off')
plt.show()
