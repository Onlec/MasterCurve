import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import io

# --- CONFIGURATIE EN CSS ---
st.set_page_config(page_title="TPU Rheology Tool", layout="wide")

# Forceer een bredere zijbalk voor betere controle over de sliders
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 350px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧪 TPU Rheology Master Curve Tool")

def load_rheo_data(file):
    """Robuuste parser voor Anton Paar reometer CSV exports."""
    try:
        file.seek(0)
        raw_bytes = file.read()
        if raw_bytes[:2] == b'\xff\xfe':
            decoded_text = raw_bytes.decode('utf-16-le')
        elif raw_bytes[:3] == b'\xef\xbb\xbf':
            decoded_text = raw_bytes.decode('utf-8-sig')
        else:
            try:
                decoded_text = raw_bytes.decode('latin-1')
            except:
                decoded_text = raw_bytes.decode('utf-8')
    except Exception as e:
        st.error(f"Encoding error: {e}")
        return pd.DataFrame()
    
    lines = decoded_text.splitlines()
    all_data = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'Interval data:' in line and 'Point No.' in line and 'Storage Modulus' in line:
            header_parts = line.split('\t')
            clean_headers = [p.strip() for p in header_parts if p.strip() and p.strip() != 'Interval data:']
            i += 3
            while i < len(lines):
                data_line = lines[i]
                if 'Result:' in data_line or 'Interval data:' in data_line: break
                if not data_line.strip():
                    i += 1
                    continue
                parts = data_line.split('\t')
                non_empty_parts = [p.strip() for p in parts if p.strip()]
                if len(non_empty_parts) >= 4:
                    row_dict = {clean_headers[idx]: non_empty_parts[idx] for idx in range(len(clean_headers)) if idx < len(non_empty_parts)}
                    if 'Temperature' in row_dict and 'Storage Modulus' in row_dict:
                        all_data.append(row_dict)
                i += 1
        else:
            i += 1
    
    if not all_data: return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df = df.rename(columns={'Temperature': 'T', 'Angular Frequency': 'omega', 'Storage Modulus': 'Gp', 'Loss Modulus': 'Gpp'})
    
    def safe_float(val):
        try: return float(str(val).replace(',', '.'))
        except: return np.nan
    
    for col in ['T', 'omega', 'Gp', 'Gpp']:
        if col in df.columns: df[col] = df[col].apply(safe_float)
    
    return df.dropna(subset=['T', 'omega', 'Gp']).query("Gp > 0 and omega > 0")

# --- SIDEBAR ---
st.sidebar.header("1. Data Import")
uploaded_file = st.sidebar.file_uploader("Upload je Reometer CSV", type=['csv', 'txt'])

if uploaded_file:
    df = load_rheo_data(uploaded_file)
    
    if not df.empty and 'T' in df.columns:
        df['T_group'] = df['T'].round(0)
        temps = sorted(df['T_group'].unique())
        st.sidebar.success(f"✅ {len(temps)} temperaturen geladen")
        
        st.sidebar.header("2. TTS Instellingen")
        selected_temps = st.sidebar.multiselect("Selecteer temperaturen", options=temps, default=temps)
        ref_temp = st.sidebar.selectbox("Referentie Temperatuur (°C)", selected_temps if selected_temps else temps, index=len(selected_temps)//2 if selected_temps else 0)
        
        if 'shifts' not in st.session_state or set(st.session_state.shifts.keys()) != set(temps):
            st.session_state.shifts = {t: 0.0 for t in temps}

        if 'reset_id' not in st.session_state:
            st.session_state.reset_id = 0

        c_auto, c_reset = st.sidebar.columns(2)
        
        # RESET KNOP
        if c_reset.button("🔄 Reset"):
            for t in temps: 
                st.session_state.shifts[t] = 0.0
            st.session_state.reset_id += 1
            st.rerun()

        # AUTO-ALIGN KNOP
        if c_auto.button("🚀 Auto-Align"):
            for t in selected_temps:
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
                    
                    f_interp = interp1d(log_w_ref, log_g_ref, bounds_error=False, fill_value=np.nan)
                    val_at_target = f_interp(log_w_target)
                    mask = ~np.isnan(val_at_target)
                    
                    if np.sum(mask) < 2: return 9999
                    return np.sum((val_at_target[mask] - log_g_target.values[mask])**2)

                res = minimize(objective, x0=st.session_state.shifts[t], method='Nelder-Mead')
                st.session_state.shifts[t] = round(float(res.x[0]), 2)
            
            # CRUCIAAL: Verhoog de ID zodat de sliders de nieuwe waarden laden
            st.session_state.reset_id += 1 
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.subheader("Handmatige Shift (log aT)")
        
        for t in selected_temps:
            # Door de key te veranderen bij elke auto-align/reset, 
            # springt de slider fysiek naar de berekende waarde.
            st.session_state.shifts[t] = st.sidebar.slider(
                f"{int(t)}°C", 
                -15.0, 15.0, 
                float(st.session_state.shifts[t]), 
                step=0.1,
                format="%.1f",
                key=f"slide_{t}_{st.session_state.reset_id}"
            )

        # --- VISUALISATIE ---
        st.write(f"### Master Curve bij {ref_temp}°C")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            colors = plt.cm.viridis(np.linspace(0, 1, len(selected_temps)))
            for t, color in zip(selected_temps, colors):
                data = df[df['T_group'] == t].copy()
                a_t = 10**st.session_state.shifts[t]
                ax1.loglog(data['omega'] * a_t, data['Gp'], 'o-', color=color, label=f"{int(t)}°C G'", markersize=4)
                if 'Gpp' in data.columns:
                    ax1.loglog(data['omega'] * a_t, data['Gpp'], 'x--', color=color, alpha=0.3, markersize=3)
            ax1.set_xlabel("ω·aT (rad/s)")
            ax1.set_ylabel("G', G'' (Pa)")
            ax1.legend(loc='lower right', fontsize=8, ncol=2)
            ax1.grid(True, which="both", alpha=0.2)
            st.pyplot(fig1)
        
        with col2:
            st.subheader("Shift Factors")
            fig2, ax2 = plt.subplots(figsize=(5, 6))
            t_list = sorted(st.session_state.shifts.keys())
            s_list = [st.session_state.shifts[t] for t in t_list]
            ax2.plot(t_list, s_list, 's-', color='orange')
            ax2.axvline(ref_temp, color='red', linestyle='--')
            ax2.set_ylabel("log(aT)")
            ax2.set_xlabel("T (°C)")
            st.pyplot(fig2)
            
            shifts_df = pd.DataFrame(list(st.session_state.shifts.items()), columns=['T_C', 'log_aT'])
            st.download_button("📥 Download Shifts", data=shifts_df.to_csv(index=False), file_name="shifts.csv")
    else:
        st.error("Bestand niet herkend.")
else:
    st.info("Upload een bestand.")