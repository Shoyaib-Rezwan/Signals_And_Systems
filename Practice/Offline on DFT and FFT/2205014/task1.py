import tkinter as tk
import numpy as np
import math
from discrete_framework import DiscreteSignal, DFTAnalyzer, FastFourierTransform

#Control variables
FRAME_UPDATE_DURATION=1
CLEAR_CANVAS=False
TRACE_WIDTH=4
MINIMUM_RADIUS=2.0

class DoodlingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fourier Epicycles Doodler")
        
        # --- UI Layout ---
        self.canvas = tk.Canvas(root, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        root.state('zoomed')
        
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)
        
        # Buttons
        tk.Button(control_frame, text="Clear Canvas", command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Draw Epicycles", command=self.run_transform).pack(side=tk.LEFT, padx=5)
        
        # Toggle Switch (Radio Buttons)
        self.use_fft = tk.BooleanVar(value=False)
        tk.Label(control_frame, text=" |  Algorithm: ").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(control_frame, text="Naive DFT", variable=self.use_fft, value=False).pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="FFT", variable=self.use_fft, value=True).pack(side=tk.LEFT)

        # State Variables
        self.points = []
        self.drawing = False
        self.fourier_coeffs = None
        self.is_animating = False
        self.after_id = None
        self.tip_points=[]

        # Bindings
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

    def start_draw(self, event):
        self.is_animating = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.canvas.delete("all")
        self.points = []
        self.drawing = True

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            # print(x,y)
            self.points.append((x, y))
            r = 2
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="red", outline="red")

    def end_draw(self, event):
        self.drawing = False

    def clear(self):
        self.canvas.delete("all")
        self.points = []
        self.is_animating = False
        if self.after_id:
            self.root.after_cancel(self.after_id)

    def draw_epicycle(self, x, y, radius):
        """
        Helper method for students to draw a circle (epicycle).
        x, y: Center coordinates
        radius: Radius of the circle
        """
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, outline="blue", tags="epicycle")

    def run_transform(self):
        if not self.points: return
        
        # Implementation
        # 1. Convert (x,y) points to Complex Signal
        data=[]
        for (x,y) in self.points:
            data.append(x+1j*y)
        signal=DiscreteSignal(np.array(data))
        # 2. Select Algorithm
        analyzer=DFTAnalyzer()
        if self.use_fft.get():
            analyzer=FastFourierTransform()
        # 3. Compute Transform
        X=analyzer.compute_dft(signal)
        entries=[]
        for x in X:
            amp=np.abs(x)
            angle=np.angle(x)
            entries.append((amp, angle))
        self.fourier_coeffs=entries

        mean_point=np.mean([p[0] for p in self.points])+1j*np.mean([p[1] for p in self.points])
        # print("Transform logic needed.")
        if CLEAR_CANVAS:
            self.canvas.delete("all")
        self.tip_points=[]
        # self.animate_epicycles(mean_point)
        self.animate_epicycles(0)

    def animate_epicycles(self, center_offset):
        self.is_animating = True
        self.time_step = 0
        self.num_frames = len(self.fourier_coeffs)
        
        self.center_offset = center_offset
        self.update_frame()

    def update_frame(self):
        if not self.is_animating: return
        dot_width=TRACE_WIDTH
        minimum_radius=MINIMUM_RADIUS
        self.canvas.delete("epicycle") 
        
        # TODO: Implementation
        # 1. Calculate the current time 't' based on self.time_step
        N=self.num_frames
        # 2. Reconstruct the signal value 'z' at time 't' 
        x=np.real(self.center_offset)
        y=np.imag(self.center_offset)
        k=0
        for amp,phase in self.fourier_coeffs:
            angle=2*np.pi*k*self.time_step/N+phase
            r=amp/N

            dx=r*np.cos(angle)
            dy=r*np.sin(angle)
            k+=1

        # 3. Draw the epicycles:
            if r>=minimum_radius:
                self.draw_epicycle(x,y,r)
                self.canvas.create_line(x,y,x+dx,y+dy,arrow=tk.LAST,fill="green",width=2,tags="epicycle")
            x+=dx
            y+=dy
        # 4. Draw the tips
        pen_x=x
        pen_y=y

        
        self.canvas.create_oval(pen_x-dot_width,pen_y-dot_width,pen_x+dot_width,pen_y+dot_width,tags="epicycle")

        self.tip_points.append((pen_x,pen_y))

        if len(self.tip_points)>1:
            for i in range(1,len(self.tip_points)):
                x0,y0=self.tip_points[i-1]
                x1,y1=self.tip_points[i]
                self.canvas.create_line(x0,y0,x1,y1,width=TRACE_WIDTH,tags="epicycle")
        # Loop animation
        self.time_step+=1
        if self.time_step>=N:
            self.is_animating=False
            return
        self.after_id = self.root.after(FRAME_UPDATE_DURATION, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = DoodlingApp(root)
    root.mainloop()