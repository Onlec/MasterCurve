import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Pagina instellingen
st.set_page_config(page_title="TPU Rheology Tool", layout="wide")

st.title("🧪 TPU Rheology Master Curve Tool")

# --- FUNCTIE VOOR DATA INLEZEN ---
def load_rheo_data(file):
    content = file.getvalue().decode('utf-8').splitlines()
    start_row = 0
    for i, line in enumerate(content):
        if "Point No." in line:
            start_row = i
            break
    
    file.seek(0)
    df = pd.read_csv(file, sep='\t', skiprows=start_row, decimal='.')
    
    # Clean data: verwijder eenheden-rij en lege regels
    df['Point No.'] = pd.to_numeric(df['Point No.'], errors='coerce')
    df = df.dropna(subset=['Point No.'])
    
    mapping = {
        'Temperature': 'T',
        'Angular Frequency': 'omega',
        'Storage Modulus': 'Gp',
        'Loss Modulus': 'Gpp'
    }
    df = df.rename(columns=mapping)
    
    for col in ['T', 'omega', 'Gp', 'Gpp']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.dropna(subset=['omega', 'Gp'])

# --- SIDEBAR: BESTAND UPLOADEN ---
st.sidebar.header("1. Data Import")
uploaded_file = st.sidebar.file_uploader("Upload je Reometer CSV", type=['csv', 'txt'])

if uploaded_file:
    df = load_rheo_data(uploaded_file)
    df['T_group'] = df['T'].round(0)
    temps = sorted(df['T_group'].unique())
    
    st.sidebar.success(f"{len(temps)} temperaturen geladen")

    # --- SIDEBAR: CONTROLS ---
    st.sidebar.header("2. TTS Instellingen")
    ref_temp = st.sidebar.selectbox("Referentie Temperatuur (°C)", temps, index=len(temps)//2)
    
    # Initialiseer shifts in session state
    if 'shifts' not in st.session_state:
        st.session_state.shifts = {t: 0.0 for t in temps}

    if st.sidebar.button("🚀 Automatisch Uitlijnen"):
        # Optimalisatie logica
        for t in temps:
            if t == ref_temp:
                st.session_state.shifts[t] = 0.0
                continue
            
            def objective(log_at):
                ref_data = df[df['T_group'] == ref_temp]
                target_data = df[df['T_group'] == t]
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
            st.session_state.shifts[t] = float(res.x[0])

    # Handmatige sliders
    st.sidebar.subheader("Handmatige aanpassing")
    for t in temps:
        st.session_state.shifts[t] = st.sidebar.slider(
            f"log(aT) @ {t}°C", -10.0, 10.0, st.session_state.shifts[t]
        )

    # --- HOOFDSCHERM: GRAFIEKEN ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Master Curve")
        fig1, ax1 = plt.subplots(figsize=(10, 7))
        for t in temps:
            data = df[df['T_group'] == t]
            a_t = 10**st.session_state.shifts[t]
            ax1.loglog(data['omega'] * a_t, data['Gp'], 'o-', label=f"{t}°C G'")
            ax1.loglog(data['omega'] * a_t, data['Gpp'], 'x--', alpha=0.5)
        
        ax1.set_xlabel("Verschoven Frequentie ω·aT (rad/s)")
        ax1.set_ylabel("Modulus G', G'' (Pa)")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend(loc='best', fontsize='small', ncol=2)
        st.pyplot(fig1)

    with col2:
        st.subheader("Shift Factors")
        fig2, ax2 = plt.subplots(figsize=(5, 8))
        t_vals = list(st.session_state.shifts.keys())
        at_vals = list(st.session_state.shifts.values())
        ax2.plot(t_vals, at_vals, 's-', color='orange')
        ax2.set_xlabel("Temperatuur (°C)")
        ax2.set_ylabel("log(aT)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
        # Export data
        st.subheader("Export")
        export_df = pd.DataFrame({"Temperatuur": t_vals, "log_aT": at_vals})
        st.download_button("Download Shift Factors", export_df.to_csv(index=False), "shifts.csv")

else:
    st.info("Upload een bestand in de zijbalk om te beginnen.")