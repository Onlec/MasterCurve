import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Pagina instellingen
st.set_page_config(page_title="TPU Rheology Tool", layout="wide")

st.title("🧪 TPU Rheology Master Curve Tool")

# --- 1. DE ROBUUSTE INLEESFUNCTIE ---
def load_rheo_data(file):
    try:
        raw_text = file.getvalue().decode('latin-1')
    except:
        raw_text = file.getvalue().decode('utf-8')
    
    lines = raw_text.splitlines()
    
    start_row = -1
    for i, line in enumerate(lines):
        # Specifieke check om de 'Interval data' samenvatting over te slaan
        if "Point No." in line and "Temperature" in line and "Interval data:" not in line:
            start_row = i
            break
            
    if start_row == -1:
        # Fallback voor als de bovenstaande check te streng is
        for i, line in enumerate(lines):
            if "Point No." in line and "Temperature" in line:
                start_row = i
                break

    if start_row == -1:
        return pd.DataFrame()

    file.seek(0)
    df = pd.read_csv(file, sep='\t', skiprows=start_row, encoding='latin-1', on_bad_lines='warn')

    df.columns = df.columns.str.strip()
    
    new_cols = {}
    for col in df.columns:
        c = col.lower()
        if 'point' in c: new_cols[col] = 'Point'
        elif 'temperature' in c: new_cols[col] = 'T'
        elif 'angular frequency' in c: new_cols[col] = 'omega'
        elif 'storage modulus' in c: new_cols[col] = 'Gp'
        elif 'loss modulus' in c: new_cols[col] = 'Gpp'
    
    df = df.rename(columns=new_cols)

    if 'Point' in df.columns:
        df['Point'] = pd.to_numeric(df['Point'], errors='coerce')
        df = df.dropna(subset=['Point'])
    
    for col in ['T', 'omega', 'Gp', 'Gpp']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df.dropna(subset=['T', 'omega', 'Gp'])
# --- 2. SIDEBAR: DATA IMPORT ---
st.sidebar.header("1. Data Import")
uploaded_file = st.sidebar.file_uploader("Upload je Reometer CSV", type=['csv', 'txt'])

if uploaded_file:
    # Reset de file pointer voor de zekerheid
    uploaded_file.seek(0)
    df = load_rheo_data(uploaded_file)
    
    if not df.empty and 'T' in df.columns:
        # Groepeer temperaturen (bijv. 100.01 wordt 100)
        df['T_group'] = df['T'].round(0)
        temps = sorted(df['T_group'].unique())
        st.sidebar.success(f"{len(temps)} temperaturen geladen")

        # --- 3. SIDEBAR: TTS SETTINGS ---
        st.sidebar.header("2. TTS Instellingen")
        ref_temp = st.sidebar.selectbox("Referentie Temperatuur (°C)", temps, index=len(temps)//2)
        
        # Initialiseer shift factoren in session state
        if 'shifts' not in st.session_state or len(st.session_state.shifts) != len(temps):
            st.session_state.shifts = {t: 0.0 for t in temps}

        if st.sidebar.button("🚀 Automatisch Uitlijnen"):
            for t in temps:
                if t == ref_temp:
                    st.session_state.shifts[t] = 0.0
                    continue
                
                def objective(log_at):
                    ref_data = df[df['T_group'] == ref_temp]
                    target_data = df[df['T_group'] == t]
                    
                    # Logaritmische schaal voor rheologie
                    log_w_ref = np.log10(ref_data['omega'])
                    log_g_ref = np.log10(ref_data['Gp'])
                    log_w_target = np.log10(target_data['omega']) + log_at
                    log_g_target = np.log10(target_data['Gp'])
                    
                    # Interpolatie om de overlap te vergelijken
                    f_interp = interp1d(log_w_ref, log_g_ref, bounds_error=False, fill_value=None)
                    val_at_target = f_interp(log_w_target)
                    mask = ~np.isnan(val_at_target)
                    
                    if np.sum(mask) < 2: return 9999
                    return np.sum((val_at_target[mask] - log_g_target[mask])**2)

                res = minimize(objective, x0=0.0, method='Nelder-Mead')
                st.session_state.shifts[t] = float(res.x[0])

        # Handmatige sliders voor fijnafstelling
        for t in temps:
            st.session_state.shifts[t] = st.sidebar.slider(
                f"log(aT) @ {t}°C", -10.0, 10.0, st.session_state.shifts[t]
            )

        # --- 4. HOOFDSCHERM: VISUALISATIE ---
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Master Curve")
            fig1, ax1 = plt.subplots(figsize=(10, 7))
            
            # Gebruik een kleurenkaart voor de temperaturen
            colors = plt.cm.viridis(np.linspace(0, 1, len(temps)))
            
            for t, color in zip(temps, colors):
                data = df[df['T_group'] == t]
                a_t = 10**st.session_state.shifts[t]
                
                # Plot G' (Storage Modulus)
                ax1.loglog(data['omega'] * a_t, data['Gp'], 'o-', color=color, label=f"{t}°C G'")
                
                # Plot G'' (Loss Modulus) indien aanwezig
                if 'Gpp' in data.columns:
                    ax1.loglog(data['omega'] * a_t, data['Gpp'], 'x--', color=color, alpha=0.3)
            
            ax1.set_xlabel("Verschoven Frequentie ω·aT (rad/s)")
            ax1.set_ylabel("Modulus G', G'' (Pa)")
            ax1.grid(True, which="both", alpha=0.3)
            ax1.legend(loc='lower right', fontsize='small', ncol=2)
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
            
            st.subheader("Export")
            export_df = pd.DataFrame({"Temperatuur": t_vals, "log_aT": at_vals})
            st.download_button("Download Shift Factors", export_df.to_csv(index=False), "shifts.csv")
    else:
        st.warning("Kon geen geldige data vinden in het bestand. Controleer of het een tab-gescheiden CSV is met 'Point No.' en 'Temperature' kolommen.")
else:
    st.info("Upload een reometer bestand (.csv of .txt) via de zijbalk om de Master Curve te genereren.")