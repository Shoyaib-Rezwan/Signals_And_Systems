import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from discrete_framework import DFTAnalyzer, DiscreteSignal, FastFourierTransform
from scipy.signal import resample_poly

CHUNK_SIZE=1024
BAND_NUM=5
DOWN_FACTOR=1

class AudioEqualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("DFT Audio Equalizer")
        
        self.samplerate = 0
        self.original_audio = None
        self.processed_audio = None
        
        # --- UI Layout ---
        top_frame = tk.Frame(root)
        top_frame.pack(pady=10)
        
        tk.Button(top_frame, text="Load WAV File", command=self.load_file).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Process & Play", command=self.process_and_play).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Stop Audio", command=sd.stop).pack(side=tk.LEFT, padx=10)
        
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
        for i in range(5):
            frame = tk.Frame(self.slider_frame)
            frame.pack(side=tk.LEFT, padx=5)
            tk.Label(frame, text=labels[i], font=("Arial", 8)).pack()
            slider = tk.Scale(frame, from_=2.0, to=0.0, resolution=0.1, length=150, orient=tk.VERTICAL)
            slider.set(1.0)
            slider.pack()
            self.sliders.append(slider)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            try:
                self.samplerate, data = wav.read(file_path)
                
                # Normalize to float [-1, 1]
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    data = (data.astype(np.float32) - 128.0) / 128.0
                
                # If already float, just ensure float32
                if data.dtype != np.float32:
                    data = data.astype(np.float32)

                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                
                #----------------------- Downsample ----------------------------#
                
                data=resample_poly(data,1,DOWN_FACTOR)
                self.samplerate=self.samplerate//DOWN_FACTOR
                #----------------------- Downsample ----------------------------#

                self.original_audio = data
                self.processed_audio = None
                duration = len(data) / self.samplerate
                print(f"Loaded: {len(data)} samples, {self.samplerate} Hz, {duration:.1f}s")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {e}")

    def process_and_play(self):
        if self.original_audio is None:
            messagebox.showwarning("Warning", "Please load a WAV file first.")
            return
        
        print("Starting processing...")
        
        # Get Slider Values
        gains = [s.get() for s in self.sliders]

        analyzer=DFTAnalyzer()
        if(self.use_fft.get()==True):
            analyzer=FastFourierTransform()

        # Implement the chunking, FFT, filtering, IFFT, and overlap-add here.
        output=np.zeros_like(self.original_audio)
        original_audio=self.original_audio
        audio_len= len(self.original_audio)
        start=0
        while start<audio_len:
            end=np.min([start+CHUNK_SIZE,audio_len])
            chunk=original_audio[start:end]

            sig=DiscreteSignal(chunk)
            sig=sig.pad(CHUNK_SIZE)
            X=analyzer.compute_dft(sig)

            half_len=CHUNK_SIZE//2
            band_size=np.max([half_len//BAND_NUM,1])
            for i in range(BAND_NUM):
                gn=gains[i]
                first_idx=i*band_size
                last_idx=np.min([(i+1)*band_size,half_len])
                X[first_idx:last_idx]*=gn

                if first_idx>0:
                    X[CHUNK_SIZE-last_idx:CHUNK_SIZE-first_idx]*=gn
            
            reconstruct=analyzer.compute_idft(X)
            reconstruct=np.real(reconstruct[:end-start]).astype(np.float32)
            output[start:end]=reconstruct
            start=end
        # For starter code, we just play the original audio so the button "works"
        peak=np.max(np.abs(output))
        if peak>1e-6:
            output/=peak
        
        # In the final version, this should play self.processed_audio

        # output_audio = self.original_audio 
        self.processed_audio = output
    
        sd.stop()
        # default_output = sd.default.device[0]
        sd.play(self.processed_audio, self.samplerate)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioEqualizer(root)
    root.mainloop()