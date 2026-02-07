import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import io

st.set_page_config(page_title="TPU Rheology Tool", layout="wide")
st.title("🧪 TPU Rheology Master Curve Tool")

def load_rheo_data(file):
    """
    Robuuste parser voor Anton Paar reometer CSV exports.
    Ondersteunt UTF-16 LE encoding en handelt meerdere temperatuur-intervallen af.
    """
    # Stap 1: Probeer verschillende encodings
    try:
        file.seek(0)
        raw_bytes = file.read()
        
        # Detecteer encoding op basis van BOM
        if raw_bytes[:2] == b'\xff\xfe':
            decoded_text = raw_bytes.decode('utf-16-le')
        elif raw_bytes[:3] == b'\xef\xbb\xbf':
            decoded_text = raw_bytes.decode('utf-8-sig')
        else:
            # Probeer latin-1 of utf-8
            try:
                decoded_text = raw_bytes.decode('latin-1')
            except:
                decoded_text = raw_bytes.decode('utf-8')
    except Exception as e:
        st.error(f"Encoding error: {e}")
        return pd.DataFrame()
    
    # Stap 2: Split in regels en zoek alle data-blokken
    lines = decoded_text.splitlines()
    
    all_data = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Zoek naar de regel met kolomnamen
        if 'Interval data:' in line and 'Point No.' in line and 'Storage Modulus' in line:
            # Parse header - split op tabs
            header_parts = line.split('\t')
            
            # Haal schone kolomnamen op (skip eerste "Interval data:")
            clean_headers = []
            for part in header_parts:
                part = part.strip()
                if part and part != 'Interval data:':
                    clean_headers.append(part)
            
            # Skip lege regel (i+1) en eenheden regel (i+2)
            i += 3
            
            # Nu data inlezen tot volgende sectie
            while i < len(lines):
                data_line = lines[i]
                
                # Stop als nieuwe sectie begint
                if 'Result:' in data_line or 'Interval data:' in data_line:
                    break
                
                # Skip lege regels
                if not data_line.strip():
                    i += 1
                    continue
                
                # Parse data regel - split op tabs
                parts = data_line.split('\t')
                
                # Filter lege cellen aan het begin/einde
                non_empty_parts = [p.strip() for p in parts if p.strip()]
                
                # Als we genoeg data hebben, maak een rij
                if len(non_empty_parts) >= 4:  # Minimaal Point No, T, omega, Gp
                    row_dict = {}
                    # Map eerste N waarden naar headers (skip eerste lege cel)
                    for idx, col_name in enumerate(clean_headers):
                        if idx < len(non_empty_parts):
                            row_dict[col_name] = non_empty_parts[idx]
                    
                    # Alleen toevoegen als we de belangrijkste kolommen hebben
                    if 'Temperature' in row_dict and 'Storage Modulus' in row_dict:
                        all_data.append(row_dict)
                
                i += 1
        else:
            i += 1
    
    if not all_data:
        return pd.DataFrame()
    
    # Stap 3: Maak DataFrame
    df = pd.DataFrame(all_data)
    
    # Stap 4: Hernoem kolommen naar standaard namen
    column_mapping = {
        'Temperature': 'T',
        'Angular Frequency': 'omega',
        'Storage Modulus': 'Gp',
        'Loss Modulus': 'Gpp'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Stap 5: Converteer naar numerieke waarden
    def safe_float(val):
        """Converteer string naar float, handel wetenschappelijke notatie en komma's af"""
        if pd.isna(val) or val == '':
            return np.nan
        try:
            # Vervang komma door punt voor EU formaat
            val_str = str(val).replace(',', '.')
            # Parse wetenschappelijke notatie (bijv. 2.1816E+05)
            return float(val_str)
        except:
            return np.nan
    
    # Converteer belangrijke kolommen
    for col in ['T', 'omega', 'Gp', 'Gpp']:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)
    
    # Stap 6: Filter ongeldige data
    df = df.dropna(subset=['T', 'omega', 'Gp'])
    
    # Verwijder rijen met negatieve of nul waarden (voor log plots)
    df = df[(df['Gp'] > 0) & (df['omega'] > 0)]
    
    return df

# --- SIDEBAR ---
st.sidebar.header("1. Data Import")
uploaded_file = st.sidebar.file_uploader("Upload je Reometer CSV", type=['csv', 'txt'])

if uploaded_file:
    df = load_rheo_data(uploaded_file)
    
    if not df.empty and 'T' in df.columns:
        # Groepeer per temperatuur (rond af op hele graden)
        df['T_group'] = df['T'].round(0)
        temps = sorted(df['T_group'].unique())
        st.sidebar.success(f"✅ {len(temps)} temperaturen geladen: {temps}")
        
        st.sidebar.header("2. TTS Instellingen")
        ref_temp = st.sidebar.selectbox("Referentie Temperatuur (°C)", temps, index=len(temps)//2)
        
        # Initialiseer shift factors in session state
        if 'shifts' not in st.session_state or set(st.session_state.shifts.keys()) != set(temps):
            st.session_state.shifts = {t: 0.0 for t in temps}
        
        # Automatische uitlijning
        if st.sidebar.button("🚀 Automatisch Uitlijnen"):
            for t in temps:
                if t == ref_temp:
                    st.session_state.shifts[t] = 0.0
                    continue
                
                def objective(log_at):
                    """Minimaliseer verschil tussen verschoven curve en referentie"""
                    ref_data = df[df['T_group'] == ref_temp]
                    target_data = df[df['T_group'] == t]
                    
                    # Log schaal voor beide
                    log_w_ref = np.log10(ref_data['omega'])
                    log_g_ref = np.log10(ref_data['Gp'])
                    log_w_target = np.log10(target_data['omega']) + log_at
                    log_g_target = np.log10(target_data['Gp'])
                    
                    # Interpoleer referentie curve
                    f_interp = interp1d(log_w_ref, log_g_ref, bounds_error=False, fill_value=np.nan)
                    val_at_target = f_interp(log_w_target)
                    
                    # Bereken sum of squares voor overlap regio
                    mask = ~np.isnan(val_at_target)
                    if np.sum(mask) < 2:
                        return 9999  # Te weinig overlap
                    
                    return np.sum((val_at_target[mask] - log_g_target.values[mask])**2)
                
                # Optimaliseer shift factor
                res = minimize(objective, x0=0.0, method='Nelder-Mead')
                st.session_state.shifts[t] = float(res.x[0])
            
            st.sidebar.success("Automatische uitlijning compleet!")
        
        # Handmatige sliders voor fine-tuning
        st.sidebar.subheader("Handmatige Aanpassingen")
        for t in temps:
            st.session_state.shifts[t] = st.sidebar.slider(
                f"log(aT) @ {t}°C", 
                -10.0, 10.0, 
                st.session_state.shifts[t],
                step=0.1
            )
        
        # --- VISUALISATIE ---
        st.write("### Ingeladen Data Preview")
        st.dataframe(df[['T', 'omega', 'Gp', 'Gpp']].head(10))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Master Curve")
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            
            # Kleurenschema voor temperaturen
            colors = plt.cm.plasma(np.linspace(0, 0.9, len(temps)))
            
            for t, color in zip(temps, colors):
                data = df[df['T_group'] == t].copy()
                a_t = 10**st.session_state.shifts[t]
                
                # Plot G' (Storage Modulus)
                ax1.loglog(
                    data['omega'] * a_t, 
                    data['Gp'], 
                    'o-', 
                    color=color, 
                    label=f"{int(t)}°C G'",
                    markersize=6,
                    linewidth=1.5
                )
                
                # Plot G'' (Loss Modulus) indien aanwezig
                if 'Gpp' in data.columns and not data['Gpp'].isna().all():
                    ax1.loglog(
                        data['omega'] * a_t, 
                        data['Gpp'], 
                        'x--', 
                        color=color, 
                        alpha=0.4,
                        markersize=5,
                        linewidth=1
                    )
            
            ax1.set_xlabel("Verschoven Frequentie ω·aT (rad/s)", fontsize=12)
            ax1.set_ylabel("Modulus G', G'' (Pa)", fontsize=12)
            ax1.grid(True, which="both", alpha=0.3)
            ax1.legend(loc='lower right', fontsize=8, ncol=2)
            ax1.set_title(f"TTS Master Curve @ {ref_temp}°C Referentie", fontsize=14)
            st.pyplot(fig1)
        
        with col2:
            st.subheader("Shift Factors")
            fig2, ax2 = plt.subplots(figsize=(6, 8))
            
            temps_list = list(st.session_state.shifts.keys())
            shifts_list = list(st.session_state.shifts.values())
            
            ax2.plot(temps_list, shifts_list, 's-', color='orange', markersize=8, linewidth=2)
            ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax2.axvline(ref_temp, color='red', linestyle='--', alpha=0.5, label=f'Ref: {ref_temp}°C')
            
            ax2.set_xlabel("Temperatuur (°C)", fontsize=11)
            ax2.set_ylabel("log(aT)", fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_title("WLF/Arrhenius Shift", fontsize=12)
            st.pyplot(fig2)
            
            # Download knop voor shift factors
            shifts_df = pd.DataFrame(
                list(st.session_state.shifts.items()), 
                columns=['Temperature_C', 'log_aT']
            )
            shifts_df['aT'] = 10**shifts_df['log_aT']
            
            st.download_button(
                label="📥 Download Shifts CSV",
                data=shifts_df.to_csv(index=False),
                file_name="shift_factors.csv",
                mime="text/csv"
            )
    else:
        st.error("❌ Data kon niet worden verwerkt. Controleer of het bestand de juiste kolommen bevat.")
        st.info("Verwachte kolommen: Temperature, Angular Frequency, Storage Modulus, Loss Modulus")
else:
    st.info("👆 Upload een Anton Paar reometer CSV bestand om te beginnen.")
    
    # Voorbeeld instructies
    with st.expander("ℹ️ Hoe werkt het?"):
        st.markdown("""
        ### Time-Temperature Superposition (TTS)
        
        Deze tool helpt je om:
        1. **Data importeren** van Anton Paar MCR reometer exports
        2. **Master curves bouwen** door metingen bij verschillende temperaturen te verschuiven
        3. **Shift factors bepalen** (handmatig of automatisch met optimalisatie)
        
        **Gebruik:**
        - Upload een frequency sweep CSV met meerdere temperaturen
        - Kies een referentie temperatuur
        - Klik op "Automatisch Uitlijnen" of pas handmatig aan met sliders
        - Download de resulterende shift factors voor verdere analyse
        
        **Ondersteunde formaten:**
        - Anton Paar RheoCompass exports (UTF-16, Latin-1, UTF-8)
        - Meerdere temperatuur-intervallen in één bestand
        """)