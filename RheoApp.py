import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Visuele instellingen
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class RheoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TPU Rheology Master Curve Tool")
        self.geometry("1400x850")

        self.df = None
        self.shifts = {}
        self.sliders = {}

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Zijpaneel
        self.sidebar = ctk.CTkFrame(self, width=300)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.sidebar, text="Bediening", font=("Arial", 20, "bold")).pack(pady=20)
        
        self.load_btn = ctk.CTkButton(self.sidebar, text="1. Laad Reometer CSV", command=self.load_data)
        self.load_btn.pack(pady=10, padx=20)

        self.auto_btn = ctk.CTkButton(self.sidebar, text="2. Automatisch Uitlijnen", 
                                      command=self.auto_align, fg_color="#2ecc71", hover_color="#27ae60")
        self.auto_btn.pack(pady=10, padx=20)

        self.slider_container = ctk.CTkScrollableFrame(self.sidebar, label_text="Handmatige Shift log(aT)")
        self.slider_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Grafiek venster
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 10), height_ratios=[3, 1])
        self.fig.tight_layout(pad=5.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
        if not file_path:
            return

        try:
            # Stap 1: Zoek de start van de data (waar "Point No." staat)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            start_row = 0
            for i, line in enumerate(lines):
                if "Point No." in line:
                    start_row = i
                    break
            
            # Stap 2: Inlezen met de gevonden regel als header
            # We gebruiken sep='\t' voor jouw bestand
            df = pd.read_csv(file_path, sep='\t', skiprows=start_row, decimal='.')
            
            # Stap 3: Opschonen
            # Verwijder de rij met eenheden (bijv. [°C]) door te kijken of 'Point No.' een getal is
            df['Point No.'] = pd.to_numeric(df['Point No.'], errors='coerce')
            df = df.dropna(subset=['Point No.'])

            # Kolommapping gebaseerd op jouw specifieke bestand
            mapping = {
                'Temperature': 'T',
                'Angular Frequency': 'omega',
                'Storage Modulus': 'Gp',
                'Loss Modulus': 'Gpp'
            }
            df = df.rename(columns=mapping)
            
            # Converteer naar floats voor berekeningen
            for col in ['T', 'omega', 'Gp', 'Gpp']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            self.df = df.dropna(subset=['omega', 'Gp'])
            
            # Groeperen op temperatuur (afronden op 0 decimalen voor 100, 110, etc.)
            self.df['T_group'] = self.df['T'].round(0)
            self.temps = sorted(self.df['T_group'].unique())
            
            self.create_sliders()
            self.update_plot()
            messagebox.showinfo("Succes", f"Data geladen! {len(self.temps)} temperaturen gevonden.")

        except Exception as e:
            messagebox.showerror("Fout", f"Kon bestand niet inlezen: {e}")

    def create_sliders(self):
        for widget in self.slider_container.winfo_children():
            widget.destroy()
        
        self.shifts = {t: 0.0 for t in self.temps}
        self.sliders = {}
        
        for t in self.temps:
            ctk.CTkLabel(self.slider_container, text=f"T = {t}°C").pack()
            s = ctk.CTkSlider(self.slider_container, from_=-10, to=10, 
                               command=lambda v, temp=t: self.on_slider_move(temp, v))
            s.set(0)
            s.pack(pady=(0, 10))
            self.sliders[t] = s

    def on_slider_move(self, temp, value):
        self.shifts[temp] = float(value)
        self.update_plot()

    def auto_align(self):
        if self.df is None: return
        
        # Gebruik de middelste temperatuur als referentie
        ref_temp = self.temps[len(self.temps)//2]
        self.shifts[ref_temp] = 0.0
        self.sliders[ref_temp].set(0.0)

        for t in self.temps:
            if t == ref_temp: continue
            
            def objective(log_at):
                ref_data = self.df[self.df['T_group'] == ref_temp]
                target_data = self.df[self.df['T_group'] == t]
                
                log_w_ref = np.log10(ref_data['omega'])
                log_g_ref = np.log10(ref_data['Gp'])
                
                log_w_target = np.log10(target_data['omega']) + log_at
                log_g_target = np.log10(target_data['Gp'])
                
                f_interp = interp1d(log_w_ref, log_g_ref, bounds_error=False, fill_value=None)
                val_at_target = f_interp(log_w_target)
                
                mask = ~np.isnan(val_at_target)
                if np.sum(mask) < 2: return 9999
                return np.sum((val_at_target[mask] - log_g_target[mask])**2)

            res = minimize(objective, x0=0.0, method='Nelder-Mead')
            self.shifts[t] = float(res.x[0])
            self.sliders[t].set(float(res.x[0]))
            
        self.update_plot()

    def update_plot(self):
        if self.df is None: return
        
        self.ax1.clear()
        self.ax2.clear()
        
        for t in self.temps:
            data = self.df[self.df['T_group'] == t]
            a_t = 10**self.shifts[t]
            
            # Master Curve Plot
            self.ax1.loglog(data['omega'] * a_t, data['Gp'], 'o-', label=f"{t}°C G'", markersize=4)
            self.ax1.loglog(data['omega'] * a_t, data['Gpp'], 'x--', label=f"{t}°C G''", markersize=3, alpha=0.6)

        self.ax1.set_xlabel("Verschoven Frequentie ω·aT (rad/s)")
        self.ax1.set_ylabel("Modulus G', G'' (Pa)")
        self.ax1.set_title("Master Curve (TTS)")
        self.ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 7})
        self.ax1.grid(True, which="both", alpha=0.3)

        # Shift Factor Plot
        t_vals = list(self.shifts.keys())
        at_vals = list(self.shifts.values())
        self.ax2.plot(t_vals, at_vals, 's-', color='orange')
        self.ax2.set_xlabel("Temperatuur (°C)")
        self.ax2.set_ylabel("log(aT)")
        self.ax2.set_title("Verschuivingsfactoren")
        self.ax2.grid(True, alpha=0.3)

        self.canvas.draw()

if __name__ == "__main__":
    app = RheoApp()
    app.mainloop()