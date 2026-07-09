import tkinter as tk
import numpy as np
from discrete_framework import DiscreteSignal, DFTAnalyzer, FastFourierTransform


class DoodlingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fourier Epicycles Doodler")

        # --- UI Layout ---
        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack()

        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)

        tk.Button(control_frame, text="Clear Canvas", command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Draw Epicycles", command=self.run_transform).pack(side=tk.LEFT, padx=5)

        # Toggle Switch
        self.use_fft = tk.BooleanVar(value=False)
        tk.Label(control_frame, text=" |  Algorithm: ").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(control_frame, text="Naive DFT", variable=self.use_fft, value=False).pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="FFT",       variable=self.use_fft, value=True).pack(side=tk.LEFT)

        # State Variables
        self.points       = []
        self.drawing      = False
        self.fourier_coeffs = None   # X[k] sorted by magnitude
        self.N            = 0        # number of DFT coefficients
        self.is_animating = False
        self.after_id     = None
        self.traced_path  = []       # stores the tip trace so we can redraw it

        # Bindings
        self.canvas.bind("<Button-1>",      self.start_draw)
        self.canvas.bind("<B1-Motion>",     self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

    # ------------------------------------------------------------------
    # Drawing callbacks
    # ------------------------------------------------------------------
    def start_draw(self, event):
        self.is_animating = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.canvas.delete("all")
        self.points = []
        self.traced_path = []
        self.drawing = True

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.points.append((x, y))
            r = 2
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")

    def end_draw(self, event):
        self.drawing = False

    def clear(self):
        self.canvas.delete("all")
        self.points = []
        self.traced_path = []
        self.fourier_coeffs = None
        self.is_animating = False
        if self.after_id:
            self.root.after_cancel(self.after_id)

    # ------------------------------------------------------------------
    # Helper provided by starter code
    # ------------------------------------------------------------------
    def draw_epicycle(self, x, y, radius):
        self.canvas.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            outline="blue", tags="epicycle"
        )

    # ------------------------------------------------------------------
    # Task 1 – core logic
    # ------------------------------------------------------------------
    def run_transform(self):
        if not self.points:
            return

        # 1. Convert (x, y) → complex signal  z[n] = x[n] + j*y[n]
        raw = [x + 1j * y for x, y in self.points]
        signal = DiscreteSignal(raw)

        # 2. Choose algorithm
        if self.use_fft.get():
            analyzer = FastFourierTransform()
            # FFT requires power-of-2: interpolate to nearest power of 2
            N = len(signal)
            next_pow2 = 1 << (max(N - 1, 1)).bit_length()
            signal = signal.interpolate(next_pow2)
        else:
            analyzer = DFTAnalyzer()

        # 3. Compute DFT
        X = analyzer.compute_dft(signal)   # complex array of length N
        self.N = len(X)

        # 4. Build coefficient list sorted by magnitude (largest first)
        #    Each entry: (amplitude, phase, frequency_index k)
        coeffs = []
        for k in range(self.N):
            amp   = abs(X[k])
            phase = np.angle(X[k])
            coeffs.append((amp, phase, k))
        coeffs.sort(key=lambda c: c[0], reverse=True)
        self.fourier_coeffs = coeffs

        # Compute centre of the drawing to use as animation offset
        mean_x = np.mean([p[0] for p in self.points])
        mean_y = np.mean([p[1] for p in self.points])

        self.canvas.delete("all")
        self.traced_path = []
        self.animate_epicycles(complex(mean_x, mean_y))

    # ------------------------------------------------------------------
    def animate_epicycles(self, center_offset):
        self.is_animating   = True
        self.time_step      = 0
        self.num_frames     = self.N        # one full revolution = N frames
        self.center_offset  = center_offset
        self.update_frame()

    def update_frame(self):
        if not self.is_animating:
            return

        self.canvas.delete("epicycle")

        # 1. Current normalised time   t ∈ [0, 1)
        t = self.time_step / self.N

        # 2. Reconstruct z(t) by summing all rotating vectors
        #    z(t) = (1/N) * sum_k  X[k] * e^{j*2*pi*k*t}
        #    We start from the canvas mean so the drawing is centred.
        x = 0.0
        y = 0.0

        for amp, phase, k in self.fourier_coeffs:
            prev_x, prev_y = x, y

            # Contribution of this epicycle at time t
            angle = 2 * np.pi * k * t + phase
            dx = (amp / self.N) * np.cos(angle)
            dy = (amp / self.N) * np.sin(angle)

            x += dx
            y += dy

            # 3. Draw the epicycle circle centred at (prev_x, prev_y)
            radius = amp / self.N
            if radius > 0.5:                   # skip invisible tiny circles
                self.draw_epicycle(prev_x, prev_y, radius)

            # Draw the arm (vector) from circle centre to its tip
            self.canvas.create_line(prev_x, prev_y, x, y,
                                    fill="red", width=1, tags="epicycle")

        # 4. Tip of the last vector → the "pen" position
        #    Offset so drawing is centred on the canvas mean
        pen_x = x + self.center_offset.real - 0  # already in canvas coords
        pen_y = y + self.center_offset.imag - 0

        # Draw the dot at the pen tip
        self.canvas.create_oval(pen_x - 3, pen_y - 3, pen_x + 3, pen_y + 3,
                                fill="red", tags="epicycle")

        # Accumulate traced path
        self.traced_path.append((pen_x, pen_y))

        # Redraw the full traced path
        if len(self.traced_path) > 1:
            for i in range(1, len(self.traced_path)):
                x0, y0 = self.traced_path[i - 1]
                x1, y1 = self.traced_path[i]
                self.canvas.create_line(x0, y0, x1, y1,
                                        fill="black", width=2, tags="epicycle")

        # Advance time step; stop after one full cycle
        self.time_step += 1
        if self.time_step >= self.num_frames:
            self.is_animating = False
            return

        self.after_id = self.root.after(30, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = DoodlingApp(root)
    root.mainloop()