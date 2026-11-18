
import csv
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from hydrogen_math import compute_all, psi, radial

VIEWS = ("Wavefunction", "Radial probability", "Energy", "Approximation error")

class Lab:
    def __init__(self, root):
        self.root = root
        root.title("Project 5 - Hydrogen Atom")
        root.geometry("1150x760")
        root.minsize(950, 600)
        self.results = {}
        self.basis = tk.StringVar(value="STO-1G")
        self.view = tk.StringVar(value=VIEWS[0])
        self.distance = tk.DoubleVar(value=1.0)
        self.rmax = tk.DoubleVar(value=7.0)
        self.exact = tk.BooleanVar(value=True)
        self.info = tk.StringVar(value="Calculating basis sets...")
        self.redraw_pending = False

        controls = ttk.Frame(root, padding=12)
        controls.pack(fill="x")
        ttk.Label(controls, text="Basis:").grid(row=0, column=0, sticky="w")
        self.bases = ttk.OptionMenu(controls, self.basis, "STO-1G")
        self.bases.configure(width=17, state="disabled")
        self.bases.grid(row=0, column=1, padx=8, sticky="w")
        ttk.Label(controls, text="Plot:").grid(row=0, column=2, padx=5)
        self.plot_box = ttk.OptionMenu(controls, self.view, VIEWS[0], *VIEWS, command=self.change_plot)
        self.plot_box.configure(width=23)
        self.plot_box.grid(row=0, column=3, padx=8, sticky="w")
        ttk.Checkbutton(controls, text="Show exact", variable=self.exact, command=self.schedule_draw).grid(row=0, column=4, padx=12)
        ttk.Button(controls, text="Export CSV", command=self.export).grid(row=0, column=5, padx=12)

        ttk.Label(controls, text="Distance r (Bohr)").grid(row=1, column=0, columnspan=2, sticky="w", pady=(15, 0))
        ttk.Scale(controls, from_=0, to=8, variable=self.distance, command=lambda _: self.schedule_draw()).grid(row=1, column=2, columnspan=2, sticky="ew", padx=8)
        ttk.Label(controls, text="Plot range").grid(row=1, column=4)
        ttk.Scale(controls, from_=3, to=12, variable=self.rmax, command=lambda _: self.schedule_draw()).grid(row=1, column=5, sticky="ew", padx=8)
        controls.columnconfigure(3, weight=1, minsize=260)
        controls.columnconfigure(5, weight=1, minsize=170)

        ttk.Label(root, textvariable=self.info, padding=(15, 8), wraplength=1100).pack(fill="x")
        self.canvas = tk.Canvas(root, bg="white", highlightthickness=1, highlightbackground="#dddddd")
        self.canvas.pack(fill="both", expand=True, padx=15, pady=12)
        self.canvas.bind("<Configure>", lambda e: self.schedule_draw())
        ttk.Label(root, text="Atomic units | Exact ground-state energy: -0.500000 Ha | Gaussian coefficients optimized variationally", padding=12).pack(fill="x")
        threading.Thread(target=self.load, daemon=True).start()

    def change_basis(self, name):
        self.basis.set(name)
        print("Selected basis:", name)
        self.schedule_draw()

    def change_plot(self, name):
        self.view.set(name)
        print("Selected plot:", name)
        self.schedule_draw()

    def schedule_draw(self):
        if getattr(self, "redraw_job", None) is not None:
            self.root.after_cancel(self.redraw_job)
        self.redraw_job = self.root.after(60, self.run_draw)

    def run_draw(self):
        self.redraw_job = None
        self.draw()

    def load(self):
        try:
            results, errors = compute_all(80)
        except Exception as e:
            results, errors = {}, {"Solver": str(e)}
        def finish():
            self.results = results
            names = list(results.keys())
            menu = self.bases["menu"]
            menu.delete(0, "end")
            for name in names:
                menu.add_command(label=name, command=lambda n=name: self.change_basis(n))
            self.bases.configure(state="normal" if names else "disabled")
            if names:
                self.basis.set(names[0])
                self.schedule_draw()
            else:
                self.info.set("No basis sets could be solved.")
            if errors:
                messagebox.showwarning("Calculation warnings", "\n".join(f"{k}: {v}" for k, v in errors.items()))
        self.root.after(0, finish)

    def draw(self):
        self.canvas.delete("all")
        name = self.basis.get()
        if name not in self.results: return
        z = self.results[name]
        r0, view = self.distance.get(), self.view.get()
        self.info.set(f'{name} | r = {r0:.2f} Bohr | T = {z["T"]:.8f} Ha | V = {z["V"]:.8f} Ha | E = {z["E"]:.9f} Ha | Error = {z["E"]+0.5:.3e} Ha | -V/T = {z["Virial"]:.6f}')
        w, h = max(self.canvas.winfo_width(), 500), max(self.canvas.winfo_height(), 350)
        left, right, top, bottom = 90, w-45, 65, h-75

        if view == "Energy":
            labels = ["Kinetic T", "Potential V", "Total E", "Exact E"]
            vals = [z["T"], z["V"], z["E"], -0.5]
            colors = ["#3b82c4", "#ed9234", "#34a276", "#777777"]
            lo, hi = min(vals+[0])-0.1, max(vals+[0])+0.1
            def yy(y): return bottom-(y-lo)/(hi-lo)*(bottom-top)
            self.canvas.create_text(w/2, 25, text="Hydrogen Atom Energy Comparison", fill="black", font=("Arial", 15, "bold"))
            self.canvas.create_line(left, yy(0), right, yy(0), fill="#555555")
            for i, (label, val) in enumerate(zip(labels, vals)):
                x, bw = left+(i+0.5)*(right-left)/4, (right-left)/9
                self.canvas.create_rectangle(x-bw/2, yy(0), x+bw/2, yy(val), fill=colors[i], outline="")
                self.canvas.create_text(x, bottom+25, text=label, fill="black", font=("Arial", 11))
                self.canvas.create_text(x, yy(val)+(-15 if val>=0 else 15), text=f"{val:.6f}", fill="black")
            self.canvas.create_text(18, top, text="Energy (Ha)", anchor="w", fill="black")
            return

        rmax = self.rmax.get()
        rr = np.linspace(0, rmax, 250)
        if view == "Wavefunction":
            series = [(f"{name} Approximation", psi(name, self.results, rr), "#1769aa")]
            if self.exact.get(): series.append(("Exact 1s", psi("Exact", self.results, rr), "#222222"))
            ylabel, title = "Wavefunction psi(r)", "Hydrogen 1s Wavefunction"
        elif view == "Radial probability":
            series = [(f"{name} Approximation", radial(name, self.results, rr), "#1769aa")]
            if self.exact.get(): series.append(("Exact 1s", radial("Exact", self.results, rr), "#222222"))
            ylabel, title = "4 pi r^2 |psi(r)|^2", "Radial Probability Distribution"
        elif view == "Approximation error":
            series = [("Approximation - Exact", psi(name, self.results, rr)-psi("Exact", self.results, rr), "#8855a8")]
            ylabel, title = "Wavefunction difference", "Approximation Error"
        else:
            return

        vals = np.concatenate([arr for _, arr, _ in series])
        lo, hi = float(min(vals.min(), 0)), float(max(vals.max(), 0))
        pad = max((hi-lo)*0.12, 0.02)
        lo, hi = lo-pad, hi+pad
        def xx(x): return left+x/rmax*(right-left)
        def yy(y): return bottom-(y-lo)/(hi-lo)*(bottom-top)
        self.canvas.create_text(w/2, 25, text=title, fill="black", font=("Arial", 15, "bold"))

        for i in range(6):
            x, y = left+(right-left)*i/5, top+(bottom-top)*i/5
            self.canvas.create_line(x, top, x, bottom, fill="#eeeeee")
            self.canvas.create_line(left, y, right, y, fill="#eeeeee")
            self.canvas.create_text(x, bottom+18, text=f"{rmax*i/5:.1f}", fill="black")
            self.canvas.create_text(left-10, y, text=f"{hi-(hi-lo)*i/5:.3f}", anchor="e", fill="black")
        self.canvas.create_rectangle(left, top, right, bottom, outline="#666666")
        if lo < 0 < hi: self.canvas.create_line(left, yy(0), right, yy(0), fill="#aaaaaa", dash=(3, 3))

        for label, arr, color in series:
            points = []
            for x, y in zip(rr, arr): points.extend((xx(float(x)), yy(float(y))))
            self.canvas.create_line(*points, fill=color, width=2)

        if r0 <= rmax:
            x = xx(r0)
            self.canvas.create_line(x, top, x, bottom, fill="#d35400", dash=(5, 4))
            self.canvas.create_text(min(x+8, right-85), top+12, text=f"r = {r0:.2f}", anchor="w", fill="#d35400")

        self.canvas.create_text((left+right)/2, h-20, text="Distance r (Bohr)", fill="black", font=("Arial", 12))
        self.canvas.create_text(15, top-18, text=ylabel, anchor="w", fill="black")
        for i, (label, _, color) in enumerate(series):
            x, y = right-215, top+20+i*25
            self.canvas.create_line(x, y, x+28, y, fill=color, width=3)
            self.canvas.create_text(x+36, y, text=label, anchor="w", fill="black", font=("Arial", 11))

    def export(self):
        if not self.results: return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="Project5_Results.csv", filetypes=[("CSV files", "*.csv")])
        if not path: return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Basis", "Kinetic T (Ha)", "Potential V (Ha)", "Total E (Ha)", "Energy error (Ha)", "-V/T"])
            writer.writerow(["Exact", 0.5, -1.0, -0.5, 0, 2])
            for name, z in self.results.items():
                writer.writerow([name, z["T"], z["V"], z["E"], z["E"]+0.5, z["Virial"]])
        messagebox.showinfo("Saved", f"Results saved to:\n{path}")

if __name__ == "__main__":
    root = tk.Tk()
    Lab(root)
    root.mainloop()
