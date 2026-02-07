import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

st.set_page_config(page_title="TPU Rheology Tool", layout="wide")
st.title("🧪 TPU Rheology Master Curve Tool")

def load_rheo_data(file):
    try:
        raw_text = file.getvalue().decode('latin-1')
    except:
        raw_text = file.getvalue().decode('utf-8')
    
    lines = raw_text.splitlines()
    
    # 1. Zoek de echte header (we negeren de samenvatting op regel 7)
    start_row = -1
    for i, line in enumerate(lines):
        clean_line = line.strip()
        # Een echte dataregel heeft veel tabs en begint niet met "Interval data"
        if "Point No." in clean_line and "Storage Modulus" in clean_line:
            if "Interval data:" not in line:
                start_row = i
                break

    if start_row == -1:
        return pd.DataFrame()

    # 2. Lees het bestand in
    file.seek(0)
    df = pd.read_csv(
        file, 
        sep='\t', 
        skiprows=start_row, 
        encoding='latin-1', 
        on_bad_lines='warn',
        skip_blank_lines=True
    )

    # 3. Kolomnamen extreem goed schoonmaken
    # Verwijder spaties, tabs en onzichtbare tekens
    df.columns = [str(c).strip() for c in df.columns]
    
    # 4. Mapping (gebaseerd op jouw exacte kolomnamen)
    mapping = {
        'Temperature': 'T',
        'Angular Frequency': 'omega',
        'Storage Modulus': 'Gp',
        'Loss Modulus': 'Gpp'
    }
    df = df.rename(columns=mapping)

    # 5. Filter de rommel eruit
    # We zoeken de eerste kolom die 'Point' bevat (vaak 'Point No.')
    point_col = [c for c in df.columns if 'Point' in c]
    if point_col:
        # Gooi rijen weg die geen getal zijn (zoals de eenhedenrij [°C])
        df[point_col[0]] = pd.to_numeric(df[point_col[0]], errors='coerce')
        df = df.dropna(subset=[point_col[0]])

    # 6. Forceer numeriek en verwijder NaN's in cruciale kolommen
    for col in ['T', 'omega', 'Gp']:
        if col in df.columns:
            # Vervang eventuele komma's door punten voor de zekerheid
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Debug: laat zien welke kolommen we hebben gevonden in de console/Streamlit
    # st.write("Gevonden kolommen:", list(df.columns)) 
    
    return df.dropna(subset=['T', 'omega', 'Gp'])
# --- SIDEBAR ---
st.sidebar.header("1. Data Import")
uploaded_file = st.sidebar.file_uploader("Upload je Reometer CSV", type=['csv', 'txt'])

if uploaded_file:
    df = load_rheo_data(uploaded_file)
    
    if not df.empty and 'T' in df.columns:
        df['T_group'] = df['T'].round(0)
        temps = sorted(df['T_group'].unique())
        st.sidebar.success(f"{len(temps)} temperaturen geladen")

        st.sidebar.header("2. TTS Instellingen")
        ref_temp = st.sidebar.selectbox("Referentie Temperatuur (°C)", temps, index=len(temps)//2)
        
        if 'shifts' not in st.session_state or set(st.session_state.shifts.keys()) != set(temps):
            st.session_state.shifts = {t: 0.0 for t in temps}

        if st.sidebar.button("🚀 Automatisch Uitlijnen"):
            for t in temps:
                if t == ref_temp:
                    st.session_state.shifts[t] = 0.0
                    continue
                
                def objective(log_at):
                    ref_data = df[df['T_group'] == ref_temp]
                    target_data = df[df['T_group'] == t]
                    log_w_ref, log_g_ref = np.log10(ref_data['omega']), np.log10(ref_data['Gp'])
                    log_w_target, log_g_target = np.log10(target_data['omega']) + log_at, np.log10(target_data['Gp'])
                    f_interp = interp1d(log_w_ref, log_g_ref, bounds_error=False, fill_value=None)
                    val_at_target = f_interp(log_w_target)
                    mask = ~np.isnan(val_at_target)
                    return np.sum((val_at_target[mask] - log_g_target[mask])**2) if np.sum(mask) >= 2 else 9999

                res = minimize(objective, x0=0.0, method='Nelder-Mead')
                st.session_state.shifts[t] = float(res.x[0])

        for t in temps:
            st.session_state.shifts[t] = st.sidebar.slider(f"log(aT) @ {t}°C", -10.0, 10.0, st.session_state.shifts[t])

        # --- GRAFIEKEN ---
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Master Curve")
            fig1, ax1 = plt.subplots(figsize=(10, 7))
            colors = plt.cm.plasma(np.linspace(0, 0.9, len(temps)))
            for t, color in zip(temps, colors):
                data = df[df['T_group'] == t]
                a_t = 10**st.session_state.shifts[t]
                ax1.loglog(data['omega'] * a_t, data['Gp'], 'o-', color=color, label=f"{t}°C G'")
                if 'Gpp' in data.columns:
                    ax1.loglog(data['omega'] * a_t, data['Gpp'], 'x--', color=color, alpha=0.3)
            ax1.set_xlabel("Verschoven Frequentie ω·aT (rad/s)")
            ax1.set_ylabel("Modulus G', G'' (Pa)")
            ax1.grid(True, which="both", alpha=0.3)
            ax1.legend(loc='lower right', fontsize='x-small', ncol=2)
            st.pyplot(fig1)

        with col2:
            st.subheader("Shift Factors")
            fig2, ax2 = plt.subplots(figsize=(5, 8))
            ax2.plot(list(st.session_state.shifts.keys()), list(st.session_state.shifts.values()), 's-', color='orange')
            ax2.set_xlabel("Temperatuur (°C)")
            ax2.set_ylabel("log(aT)")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            st.download_button("Download Shifts", pd.DataFrame(list(st.session_state.shifts.items()), columns=['T', 'log_aT']).to_csv(index=False), "shifts.csv")
    else:
        st.error("Data kon niet worden verwerkt. Controleer of de kolommen 'Temperature', 'Angular Frequency' en 'Storage Modulus' aanwezig zijn.")
else:
    st.info("Upload een bestand om te beginnen.")