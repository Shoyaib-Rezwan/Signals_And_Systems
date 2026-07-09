import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from discrete_framework import DiscreteSignal, DFTAnalyzer, FastFourierTransform


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
CHUNK_SIZE = 1024   # samples per block (power of 2 → FFT-friendly)
NUM_BANDS  = 5      # number of equalizer bands


class AudioEqualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("DFT Audio Equalizer")

        self.samplerate     = 0
        self.original_audio = None
        self.processed_audio = None

        # --- UI Layout ---
        top_frame = tk.Frame(root)
        top_frame.pack(pady=10)

        tk.Button(top_frame, text="Load WAV File",   command=self.load_file).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Process & Play",  command=self.process_and_play).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Stop Audio",      command=sd.stop).pack(side=tk.LEFT, padx=10)

        # Toggle Switch
        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)
        self.use_fft = tk.BooleanVar(value=False)
        tk.Label(control_frame, text="Algorithm: ").pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="DFT (Slow)", variable=self.use_fft, value=False).pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="FFT (Fast)", variable=self.use_fft, value=True).pack(side=tk.LEFT)

        # Equalizer Sliders
        self.slider_frame = tk.Frame(root)
        self.slider_frame.pack(pady=20, padx=20)

        self.sliders = []
        labels = ["Low", "Low-Mid", "Mid", "High-Mid", "High"]
        for i in range(NUM_BANDS):
            frame = tk.Frame(self.slider_frame)
            frame.pack(side=tk.LEFT, padx=5)
            tk.Label(frame, text=labels[i], font=("Arial", 8)).pack()
            slider = tk.Scale(frame, from_=2.0, to=0.0,
                              resolution=0.1, length=150, orient=tk.VERTICAL)
            slider.set(1.0)
            slider.pack()
            self.sliders.append(slider)

        # Progress label
        self.status_var = tk.StringVar(value="No file loaded.")
        tk.Label(root, textvariable=self.status_var, font=("Arial", 9)).pack(pady=5)

    # ------------------------------------------------------------------
    # Load WAV
    # ------------------------------------------------------------------
    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not file_path:
            return
        try:
            self.samplerate, data = wav.read(file_path)

            # Normalise to float32 in [-1, 1]
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128.0) / 128.0
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            # Convert stereo → mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1).astype(np.float32)

            self.original_audio  = data
            self.processed_audio = None
            duration = len(data) / self.samplerate
            self.status_var.set(
                f"Loaded: {len(data)} samples | {self.samplerate} Hz | {duration:.1f}s"
            )
            print(f"Loaded: {len(data)} samples, {self.samplerate} Hz, {duration:.1f}s")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{e}")

    # ------------------------------------------------------------------
    # Process & Play
    # ------------------------------------------------------------------
    def process_and_play(self):
        if self.original_audio is None:
            messagebox.showwarning("Warning", "Please load a WAV file first.")
            return

        self.status_var.set("Processing…")
        self.root.update()

        gains = [s.get() for s in self.sliders]   # list of 5 gain values

        # Choose transform
        if self.use_fft.get():
            analyzer = FastFourierTransform()
        else:
            analyzer = DFTAnalyzer()

        audio     = self.original_audio
        total     = len(audio)
        output    = np.zeros(total, dtype=np.float32)

        # ---------------------------------------------------------------
        # Block processing  (non-overlapping for simplicity)
        # ---------------------------------------------------------------
        start = 0
        while start < total:
            end   = min(start + CHUNK_SIZE, total)
            chunk = audio[start:end]

            # Zero-pad last chunk to CHUNK_SIZE so DFT size is consistent
            N = CHUNK_SIZE
            if len(chunk) < N:
                padded_chunk = np.zeros(N, dtype=np.float32)
                padded_chunk[:len(chunk)] = chunk
                chunk = padded_chunk

            # 1. DFT of chunk
            sig     = DiscreteSignal(chunk)
            X       = analyzer.compute_dft(sig)   # length N complex array
            X_len   = len(X)

            # 2. Apply equalizer gains to each frequency band
            #    Split [0 .. X_len/2] into NUM_BANDS equal bands.
            #    Mirror (negative freqs) is adjusted symmetrically.
            half    = X_len // 2
            band_sz = max(half // NUM_BANDS, 1)

            for b in range(NUM_BANDS):
                gain      = gains[b]
                lo        = b * band_sz
                hi        = (b + 1) * band_sz if b < NUM_BANDS - 1 else half

                # Positive frequencies
                X[lo:hi] *= gain

                # Mirror negative frequencies (indices X_len-hi .. X_len-lo)
                if lo > 0:
                    X[X_len - hi: X_len - lo] *= gain

            # 3. IDFT back to time domain
            x_rec = analyzer.compute_idft(X)          # complex array
            x_rec = x_rec[:end - start].real.astype(np.float32)

            output[start:end] = x_rec
            start = end

        # Normalise to prevent clipping
        peak = np.max(np.abs(output))
        if peak > 1e-6:
            output /= peak

        self.processed_audio = output
        self.status_var.set("Done. Playing…")
        self.root.update()

        # Playback — do NOT specify device; let sounddevice pick the default output
        sd.stop()
        sd.play(self.processed_audio, self.samplerate)
        print("Playback started.")


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioEqualizer(root)
    root.mainloop()