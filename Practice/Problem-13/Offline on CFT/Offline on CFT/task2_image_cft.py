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

    def __init__(self, image_obj: ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y

    def compute_cft(self):
        rows, cols = self.I.shape

        # output
        F = np.zeros((rows, cols), dtype=complex)

        # ------------------------------------------------
        # 1) Precompute exponentials
        # ------------------------------------------------
        exp_y = np.exp(
            -1j * 2 * np.pi * self.y[:, None] * self.y[None, :]
        )  # (rows, rows)

        exp_x = np.exp(
            -1j * 2 * np.pi * self.x[:, None] * self.x[None, :]
        )  # (cols, cols)

        # ------------------------------------------------
        # 2) Integrate over y (once!)
        #    G[u, x] = ∫ I(x,y) e^{-j2πuy} dy
        # ------------------------------------------------
        G = np.trapezoid(
            self.I[:, :, None] * exp_y[:, None, :],
            self.y,
            axis=0
        )  # shape: (cols, rows)

        # ------------------------------------------------
        # 3) Integrate over x
        #    F[u,v] = ∫ G[u,x] e^{-j2πvx} dx
        # ------------------------------------------------
        for u in range(rows):
            F[u, :] = np.trapezoid(
                G[:, u][:, None] * exp_x,
                self.x,
                axis=0
            )

        return F.real, F.imag


    def plot_magnitude(self):
        real, imag = self.compute_cft()
        magnitude = np.log(1 + np.sqrt(real**2 + imag**2))

        plt.imshow(magnitude, cmap='magma')
        plt.title("2D CFT Magnitude Spectrum")
        plt.axis('off')
        plt.savefig('spectrum.png')
        plt.show()


# =====================================================
# Frequency Filtering
# =====================================================
class FrequencyFilter:
    def low_pass(self, real, imag, cutoff):
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2

        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - cx) ** 2 + (j - cy) ** 2) > cutoff:
                    real[i, j] = 0
                    imag[i, j] = 0
        return real, imag


# =====================================================
# Inverse 2D Continuous Fourier Transform
# =====================================================
class InverseCFT2D:
    """
    Reconstructs image from 2D frequency spectrum.
    """

    def __init__(self, real, imag, x, y):
        self.real = real
        self.imag = imag
        self.x = x
        self.y = y

    def reconstruct(self):
        rows, cols = self.real.shape
        spectrum = self.real + 1j * self.imag
    
        image_rec = np.zeros((rows, cols))
    
        # ------------------------------------------------
        # 1) Precompute exponentials
        # ------------------------------------------------
        exp_u = np.exp(
            1j * 2 * np.pi * self.y[:, None] * self.y[None, :]
        )  # (rows, rows)
    
        exp_v = np.exp(
            1j * 2 * np.pi * self.x[:, None] * self.x[None, :]
        )  # (cols, cols)
    
        # ------------------------------------------------
        # 2) Integrate over u (frequency-y)
        #    G[y, v] = ∫ F(u,v) e^{j2πuy} du
        # ------------------------------------------------
        G = np.trapezoid(
            spectrum[:, None, :] * exp_u[:, :, None],
            self.y,
            axis=0
        )  # shape: (rows, cols)
    
        # ------------------------------------------------
        # 3) Integrate over v (frequency-x)
        #    I[y,x] = ∫ G(y,v) e^{j2πvx} dv
        # ------------------------------------------------
        for i in range(rows):
            image_rec[i, :] = np.trapezoid(
                G[i, :, None] * exp_v,
                self.x,
                axis=0
            ).real
    
        return image_rec




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
plt.savefig('reconstructructed.png')
plt.show()
