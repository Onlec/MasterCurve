import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d, UnivariateSpline

# --- CONFIGURATIE ---
st.set_page_config(page_title="TPU Rheology Expert Tool", layout="wide")

def load_rheo_data(file):
    try:
        file.seek(0)
        raw_bytes = file.read()
        if raw_bytes[:2] == b'\xff\xfe': decoded_text = raw_bytes.decode('utf-16-le')
        elif raw_bytes[:3] == b'\xef\xbb\xbf': decoded_text = raw_bytes.decode('utf-8-sig')
        else:
            try: decoded_text = raw_bytes.decode('latin-1')
            except: decoded_text = raw_bytes.decode('utf-8')
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
        else: i += 1
    
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
st.sidebar.title("🧪 Rheo-Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Anton Paar CSV/TXT", type=['csv', 'txt'])

if uploaded_file:
    df = load_rheo_data(uploaded_file)
    if not df.empty:
        df['T_group'] = df['T'].round(0)
        temps = sorted(df['T_group'].unique())
        
        st.sidebar.header("1. Selectie & Kleur")
        selected_temps = st.sidebar.multiselect("Temperaturen", temps, default=temps)
        if not selected_temps: st.stop()
        
        ref_temp = st.sidebar.selectbox("Referentie T (°C)", selected_temps, index=len(selected_temps)//2)
        cmap_opt = st.sidebar.selectbox("Kleurenschema", ["coolwarm", "viridis", "magma", "jet"])
        
        if 'shifts' not in st.session_state: st.session_state.shifts = {t: 0.0 for t in temps}
        if 'reset_id' not in st.session_state: st.session_state.reset_id = 0

        c_auto, c_reset = st.sidebar.columns(2)
        if c_reset.button("🔄 Reset"):
            for t in temps: st.session_state.shifts[t] = 0.0
            st.session_state.reset_id += 1
            st.rerun()

        if c_auto.button("🚀 Auto-Align"):
            for t in selected_temps:
                if t == ref_temp: continue
                def objective(log_at):
                    ref_d, tgt_d = df[df['T_group'] == ref_temp], df[df['T_group'] == t]
                    f = interp1d(np.log10(ref_d['omega']), np.log10(ref_d['Gp']), bounds_error=False)
                    v = f(np.log10(tgt_d['omega']) + log_at)
                    m = ~np.isnan(v)
                    return np.sum((v[m] - np.log10(tgt_d['Gp'].values[m]))**2) if np.sum(m) >= 2 else 9999
                res = minimize(objective, x0=st.session_state.shifts[t], method='Nelder-Mead')
                st.session_state.shifts[t] = round(float(res.x[0]), 2)
            st.session_state.reset_id += 1
            st.rerun()

        st.sidebar.header("2. Handmatige Shift")
        for t in selected_temps:
            st.session_state.shifts[t] = st.sidebar.slider(f"{int(t)}°C", -15.0, 15.0, float(st.session_state.shifts[t]), 0.1, key=f"{t}_{st.session_state.reset_id}")

        # --- DATA PREP ---
        color_map = plt.get_cmap(cmap_opt)
        colors = color_map(np.linspace(0, 0.9, len(selected_temps)))
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Master Curve", "🧪 Structuur (vGP)", "🧬 Thermisch (Ea)", "🔬 TTS Validatie", "💾 Smooth Export"])

        with tab1:
            st.subheader(f"Master Curve bij {ref_temp}°C")
            col_m1, col_m2 = st.columns([2, 1])
            with col_m1:
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                for t, color in zip(selected_temps, colors):
                    d = df[df['T_group'] == t].copy()
                    at = 10**st.session_state.shifts[t]
                    ax1.loglog(d['omega']*at, d['Gp'], 'o-', color=color, label=f"{int(t)}°C G'", markersize=4)
                    ax1.loglog(d['omega']*at, d['Gpp'], 'x--', color=color, alpha=0.3, markersize=3)
                ax1.set_xlabel("ω·aT (rad/s)"); ax1.set_ylabel("Modulus (Pa)"); ax1.legend(ncol=2, fontsize=8); ax1.grid(True, alpha=0.1)
                st.pyplot(fig1)
            with col_m2:
                st.write("**Shift Factor Trend**")
                t_list = sorted([t for t in selected_temps])
                s_list = [st.session_state.shifts[t] for t in t_list]
                fig2, ax2 = plt.subplots(); ax2.plot(t_list, s_list, 's-', color='red'); ax2.set_xlabel("T (°C)"); ax2.set_ylabel("log(aT)"); st.pyplot(fig2)

        with tab2:
            st.subheader("Van Gurp-Palmen (vGP) Analyse")
            st.info("💡 Thermorheologische eenvoud: Liggen alle curves op één lijn? Zo ja, dan is de structuur temperatuur-onafhankelijk.")
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            for t, color in zip(selected_temps, colors):
                d = df[df['T_group'] == t]
                g_star = np.sqrt(d['Gp']**2 + d['Gpp']**2)
                delta = np.degrees(np.arctan2(d['Gpp'], d['Gp']))
                ax3.plot(g_star, delta, 'o-', color=color, label=f"{int(t)}°C")
            ax3.set_xscale('log'); ax3.set_xlabel("|G*| (Pa)"); ax3.set_ylabel("δ (°)"); ax3.grid(True, alpha=0.2); st.pyplot(fig3)

        with tab3:
            st.subheader("🧬 Activeringsenergie & ODT")
            all_omegas = sorted(df['omega'].unique())
            target_w = st.select_slider("Selecteer ω voor Ea", options=all_omegas, value=all_omegas[len(all_omegas)//2])
            t_k = np.array([t + 273.15 for t in selected_temps])
            inv_t, log_at = 1/t_k, np.array([st.session_state.shifts[t] for t in selected_temps])
            slope, intercept = np.polyfit(inv_t, log_at, 1)
            ea = abs(slope * 8.314 * np.log(10) / 1000)
            st.metric("Ea (Shift)", f"{ea:.1f} kJ/mol")
            fig_ea, ax_ea = plt.subplots(); ax_ea.scatter(inv_t, log_at, color='red'); ax_ea.plot(inv_t, slope*inv_t + intercept, 'k--'); st.pyplot(fig_ea)

        with tab4:
            st.subheader("🔬 Geavanceerde TTS Validatie")
            st.markdown("### 1. Han Plot ($G'$ vs $G''$)")
            st.markdown("""
            **Wat je ziet:** Deze plot elimineert de variabele 'temperatuur' en 'frequentie'. 
            * **Interpretatie voor TPU:** Als de data bij alle temperaturen op één enkele curve valt, is je TPU 'thermorheologisch simpel'. 
            * **Afwijkingen:** Zie je dat de lijnen bij hogere temperaturen 'wegbuigen'? Dat duidt vaak op het **verlies van microfase-scheiding** of het smelten van hard-segment domeinen. TTS is daar eigenlijk niet meer geldig.
            """)
            fig_h, ax_h = plt.subplots(figsize=(8, 5))
            for t, color in zip(selected_temps, colors):
                d = df[df['T_group'] == t]
                ax_h.loglog(d['Gpp'], d['Gp'], 'o', color=color, label=f"{int(t)}°C", alpha=0.7)
            ax_h.set_xlabel("G'' (Pa)"); ax_h.set_ylabel("G' (Pa)"); ax_h.legend(); ax_h.grid(True, which="both", alpha=0.2)
            st.pyplot(fig_h)

            st.divider()

            st.markdown("### 2. Cole-Cole Plot ($\eta''$ vs $\eta'$)")
            st.markdown("""
            **Interpretatie van de Boog:**
            * **Symmetrische Halve Cirkel:** Wijst op een zeer nauwe molecuulgewichtsverdeling (monodispers), zoals een Maxwell-model.
            * **Afgeplatte of Scheve Boog:** Hoe 'platter' of breder de boog, hoe **breder de molecuulgewichtsverdeling (MWD)** van je TPU.
            * **Uitschieters bij lage η':** Wijst op de aanwezigheid van een elastisch netwerk of ongesmolten deeltjes (hard-segments) die de stroming hinderen.
            """)
            fig_c, ax_c = plt.subplots(figsize=(8, 5))
            for t, color in zip(selected_temps, colors):
                d = df[df['T_group'] == t]
                eta_p = d['Gpp'] / d['omega']
                eta_pp = d['Gp'] / d['omega']
                ax_c.plot(eta_p, eta_pp, 'o-', color=color, label=f"{int(t)}°C", markersize=4)
            ax_c.set_xlabel("η' (Pa·s)"); ax_c.set_ylabel("η'' (Pa·s)"); ax_c.legend(); ax_c.grid(True, alpha=0.2)
            st.pyplot(fig_c)

            st.divider()
            
            st.write("**3. Cross-over Punten ($G' = G''$)**")
            co_list = []
            for t in selected_temps:
                d = df[df['T_group'] == t].sort_values('omega')
                if len(d) > 2:
                    try:
                        f_diff = interp1d(np.log10(d['omega']), np.log10(d['Gp']) - np.log10(d['Gpp']), bounds_error=False)
                        # Zoek nulpunt (waar log(Gp/Gpp) = 0)
                        w_range = np.logspace(np.log10(d['omega'].min()), np.log10(d['omega'].max()), 500)
                        diffs = f_diff(np.log10(w_range))
                        idx_zero = np.nanargmin(np.abs(diffs))
                        w_co = w_range[idx_zero]
                        g_co = 10**float(interp1d(np.log10(d['omega']), np.log10(d['Gp']))(np.log10(w_co)))
                        co_list.append({"T (°C)": int(t), "ω_co (rad/s)": round(w_co, 2), "G_co (Pa)": round(g_co, 0)})
                    except: pass
            if co_list: st.table(pd.DataFrame(co_list))

        with tab5:
            st.subheader("💾 Smooth Export")
            m_list = []
            for t in selected_temps:
                d = df[df['T_group'] == t].copy()
                at = 10**st.session_state.shifts[t]
                d['w_s'] = d['omega'] * at
                d['eta_s'] = np.sqrt(d['Gp']**2 + d['Gpp']**2) / d['w_s']
                m_list.append(d)
            m_df = pd.concat(m_list).sort_values('w_s')
            s_val = st.slider("Smoothing Sterkte", 0.0, 2.0, 0.4)
            log_w, log_eta = np.log10(m_df['w_s']), np.log10(m_df['eta_s'])
            spl = UnivariateSpline(log_w, log_eta, s=s_val)
            w_new = np.logspace(log_w.min(), log_w.max(), 50)
            eta_new = 10**spl(np.log10(w_new))
            fig_s, ax_s = plt.subplots(); ax_s.loglog(m_df['w_s'], m_df['eta_s'], 'k.', alpha=0.1); ax_s.loglog(w_new, eta_new, 'r-'); st.pyplot(fig_s)
            st.download_button("Download CSV", pd.DataFrame({'w': w_new, 'eta': eta_new}).to_csv(index=False).encode('utf-8'))