import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from io import BytesIO

# --- CONFIGURATIE & STYLING ---
st.set_page_config(page_title="RheoApp", layout="wide")
st.title("RheoApp")
# Custom CSS voor betere leesbaarheid van expert-notes
st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 2rem; }
    .expert-note { background-color: #f0f2f6; padding: 15px; border-left: 5px solid #ff4b4b; border-radius: 5px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCTIES ---
def load_rheo_data(file):

    
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
                if 'Result:' in data_line or 'Interval data:' in data_line: 
                    break
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
    
    if not all_data: 
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df = df.rename(columns={'Temperature': 'T', 'Angular Frequency': 'omega', 'Storage Modulus': 'Gp', 'Loss Modulus': 'Gpp'})
    
    def safe_float(val):
        try: 
            return float(str(val).replace(',', '.'))
        except: 
            return np.nan
    
    for col in ['T', 'omega', 'Gp', 'Gpp']:
        if col in df.columns: 
            df[col] = df[col].apply(safe_float)
    
    return df.dropna(subset=['T', 'omega', 'Gp']).query("Gp > 0 and omega > 0")

def extract_sample_name(file):
        try:
            file.seek(0)
            raw_bytes = file.read()
            if raw_bytes[:2] == b'\xff\xfe': 
                text = raw_bytes.decode('utf-16-le')
            elif raw_bytes[:3] == b'\xef\xbb\xbf': 
                text = raw_bytes.decode('utf-8-sig')
            else:
                try: 
                    text = raw_bytes.decode('latin-1')
                except: 
                    text = raw_bytes.decode('utf-8')
            lines = text.splitlines()
            
            if len(lines) >= 3:
                row_3 = lines[2].split('\t') # Splitsen op tab
                if len(row_3) >= 2:
                    sample_name = row_3[1].strip() # Kolom 2
                    return sample_name if sample_name else "Onbekend_Sample"
            
            return "Onbekend_Sample"
        except Exception as e:
            return f"Error_bij_lezen_{e}"

def to_excel(summary_df, shift_df, crossover_df):
    output = BytesIO()
    # We converteren alles naar standaard Python types om de ValueError te voorkomen
    summary_df = summary_df.copy()
    summary_df['Waarde'] = summary_df['Waarde'].apply(
        lambda x: float(x) if isinstance(x, (np.float64, np.float32, np.ndarray)) else str(x)
    )
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        shift_df.to_excel(writer, sheet_name='ShiftFactors', index=False)
        crossover_df.to_excel(writer, sheet_name='Crossovers', index=False)
        
        # Kleine extra touch: kolombreedte aanpassen
        for sheetname in writer.sheets:
            writer.sheets[sheetname].set_column('A:C', 20)
            
    return output.getvalue()

# --- SIDEBAR ---
st.sidebar.title("Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload frequency sweep CSV/TXT", type=['csv', 'txt'])

if uploaded_file:
    df = load_rheo_data(uploaded_file)
    sample_name=extract_sample_name(uploaded_file)
    
    if not df.empty:
        df['T_group'] = df['T'].round(0)
        temps = sorted(df['T_group'].unique())
        
        selected_temps = st.sidebar.multiselect("Selecteer Temperaturen", temps, default=temps)
        ref_temp = st.sidebar.selectbox("Referentie T (°C)", selected_temps, index=len(selected_temps)//2)
        cmap_opt = st.sidebar.selectbox("Kleurenschema", ["coolwarm", "viridis", "magma", "jet"])

        st.sidebar.divider()
        st.sidebar.markdown("**WLF Optimalisatie**")
        tg_hint = st.sidebar.number_input("Verwachte Tg (°C) voor WLF-hint", value=-40.0)

        if 'shifts' not in st.session_state: 
            st.session_state.shifts = {t: 0.0 for t in temps}
        if 'reset_id' not in st.session_state: 
            st.session_state.reset_id = 0

        c_auto, c_reset = st.sidebar.columns(2)
        
        if c_reset.button("🔄 Reset"):
            for t in temps: 
                st.session_state.shifts[t] = 0.0
            st.session_state.reset_id += 1
            st.rerun()

        if c_auto.button("🚀 Auto-Align"):
            for t in selected_temps:
                if t == ref_temp: 
                    continue
                def objective(log_at):
                    ref_d = df[df['T_group'] == ref_temp]
                    tgt_d = df[df['T_group'] == t]
                    f = interp1d(np.log10(ref_d['omega']), np.log10(ref_d['Gp']), bounds_error=False)
                    v = f(np.log10(tgt_d['omega']) + log_at)
                    m = ~np.isnan(v)
                    return np.sum((v[m] - np.log10(tgt_d['Gp'].values[m]))**2) if np.sum(m) >= 2 else 9999
                res = minimize(objective, x0=st.session_state.shifts[t], method='Nelder-Mead')
                st.session_state.shifts[t] = round(float(res.x[0]), 2)
            st.session_state.reset_id += 1
            st.rerun()

        for t in selected_temps:
            st.session_state.shifts[t] = st.sidebar.slider(
                f"{int(t)}°C", 
                -10.0, 10.0, 
                float(st.session_state.shifts[t]), 
                0.1, 
                key=f"{t}_{st.session_state.reset_id}"
            )

        # --- BEREKENINGEN (NU OP DE JUISTE PLEK - NA VARIABELE DEFINITIE) ---
        t_k_global = np.array([t + 273.15 for t in selected_temps])
        log_at_global = np.array([st.session_state.shifts[t] for t in selected_temps])
        tr_k_global = ref_temp + 273.15

        # Arrhenius
        inv_t_global = 1/t_k_global
        slope_g, intercept_g = np.polyfit(inv_t_global, log_at_global, 1)
        ea_final = float(abs(slope_g * 8.314 * np.log(10) / 1000))
        r2_final = float(1 - (np.sum((log_at_global - (slope_g * inv_t_global + intercept_g))**2) / 
                              np.sum((log_at_global - np.mean(log_at_global))**2)))

        
        # WLF met Tg-hint van Professor
        def wlf_model(p, t, tr): return -p[0]*(t-tr) / (p[1] + (t-tr))
        def wlf_err(p): return np.sum((log_at_global - wlf_model(p, t_k_global, tr_k_global))**2)
        
        # Startwaarden aanpassen op basis van Tg
        c2_init = max(50.0, ref_temp - tg_hint) 
        res_wlf = minimize(wlf_err, x0=[17.4, c2_init])
        wlf_c1, wlf_c2 = res_wlf.x

        # --- DATA PREP ---
        color_map = plt.get_cmap(cmap_opt)
        colors = color_map(np.linspace(0, 0.9, len(selected_temps)))
        
        # WLF met Tg-hint van Professor
        def wlf_model(p, t, tr): return -p[0]*(t-tr) / (p[1] + (t-tr))
        def wlf_err(p): return np.sum((log_at_global - wlf_model(p, t_k_global, tr_k_global))**2)

        # Startwaarden aanpassen op basis van Tg
        c2_init = max(50.0, ref_temp - tg_hint) 
        res_wlf = minimize(wlf_err, x0=[17.4, c2_init])
        wlf_c1, wlf_c2 = res_wlf.x

        def cross_model(omega, eta_0, tau, n):
            """Cross model voor viscositeit: eta = eta_0 / (1 + (tau*omega)^n)"""
            return eta_0 / (1 + (tau * omega)**n)
        
        def calculate_rheo_metrics(master_df):
            """Bereken η0 en GN0 vanuit de mastercurve data"""
            # 1. Zero Shear Viscosity (η0) extrapolatie via Cross Model
            w = master_df['w_s'].values
            eta_complex = master_df['eta_s'].values
            
            try:
                # Initiële schatting: eta_0 is vaak de hoogste gemeten viscositeit
                popt, _ = curve_fit(cross_model, w, eta_complex, 
                                    p0=[eta_complex.max(), 0.1, 0.8], 
                                    bounds=([0, 0, 0], [np.inf, np.inf, 1]))
                eta_0, tau_cross, n_cross = popt
            except:
                eta_0, tau_cross, n_cross = np.nan, np.nan, np.nan

            # 2. Plateau Modulus (GN0) - Zoek waar G'' een minimum heeft of G' vlakt af
            gp = master_df['Gp'].values
            gpp = master_df['Gpp'].values
            
            # Theoretische methode: GN0 is vaak G' waarbij tan delta minimaal is in het rubberplateau
            tan_d = gpp / gp
            min_tan_idx = np.argmin(tan_d)
            gn0 = gp[min_tan_idx] if w[min_tan_idx] > 1.0 else np.nan # Alleen zinvol bij hogere frequentie
            
            return eta_0, gn0, (tau_cross, n_cross)
        
        
        st.subheader(f"{sample_name}")
        # --- TABS ---
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "📈 Master Curve", "🧪 Structuur", "📉 tan δ Analyse", 
                "🧬 Thermisch (Ea/WLF)", "🔬 Validatie", "💾 Export", 
                "⚛️ Moleculaire Analyse", "📊 Dashboard"
            ])

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
                ax1.set_xlabel("ω·aT (rad/s)")
                ax1.set_ylabel("Modulus (Pa)")
                ax1.legend(ncol=2, fontsize=8)
                ax1.grid(True, alpha=0.1)
                st.pyplot(fig1)
            
            with col_m2:
                st.write("**Shift Factor Trend**")
                t_plot = sorted(selected_temps)
                s_plot = [st.session_state.shifts[t] for t in t_plot]
                fig2, ax2 = plt.subplots()
                ax2.plot(t_plot, s_plot, 's-', color='red')
                ax2.set_xlabel("T (°C)")
                ax2.set_ylabel("log(aT)")
                st.pyplot(fig2)
                st.info("💡 Een lineaire trend wijst op Arrhenius gedrag; een sterke kromming op WLF.")

        with tab2:
            st.subheader("Van Gurp-Palmen (vGP) Analyse")
            st.markdown("""
            > **Expert Interpretatie:** Deze plot is de 'vingerafdruk' van de structuur. 
            > * **Overlappende lijnen:** Thermorheologisch eenvoudig (homogene smelt).
            > * **Spreiding van lijnen:** Thermorheologisch complex. Bij TPU duidt dit vaak op het smelten van hard-segment domeinen of fase-veranderingen.
            """)
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            for t, color in zip(selected_temps, colors):
                d = df[df['T_group'] == t]
                g_star = np.sqrt(d['Gp']**2 + d['Gpp']**2)
                delta = np.degrees(np.arctan2(d['Gpp'], d['Gp']))
                ax3.plot(g_star, delta, 'o-', color=color, label=f"{int(t)}°C")
            ax3.set_xscale('log')
            ax3.set_xlabel("|G*| (Pa)")
            ax3.set_ylabel("δ (°)")
            ax3.grid(True, alpha=0.2)
            ax3.legend()
            st.pyplot(fig3)
        with tab3:
            st.subheader("Loss Tangent (tan δ) - Relaxation Spectrum")
            fig_tan, ax_tan = plt.subplots(figsize=(10, 5))
            
            for t, color in zip(selected_temps, colors):
                d = df[df['T_group'] == t]
                tan_d = d['Gpp'] / d['Gp']
                ax_tan.semilogx(d['omega'], tan_d, 'o-', color=color, label=f"{int(t)}°C")
            
            ax_tan.axhline(1, color='red', linestyle='--', alpha=0.5, label='G\' = G\'\'')
            ax_tan.set_xlabel("ω (rad/s)")
            ax_tan.set_ylabel("tan δ")
            ax_tan.legend(ncol=2, fontsize=8)
            ax_tan.grid(True, alpha=0.2)
            st.pyplot(fig_tan)
            st.info("💡 Peaks in tan δ geven karakteristieke relaxatietijden aan. Bij TPU zie je vaak een verschuiving die duidt op de beweeglijkheid van de zachte segmenten.")
        with tab4:
            st.subheader("🧬 Arrhenius vs WLF Vergelijking")
            
            # Gebruik de reeds berekende waarden
            col_t1, col_t2 = st.columns([2, 1])
            
            with col_t1:
                fig_t, ax_t = plt.subplots()
                ax_t.scatter(selected_temps, log_at_global, color='black', label='Data', s=50)
                
                # Arrhenius fit lijn
                ax_t.plot(
                    selected_temps, 
                    slope_g*(1/(np.array(selected_temps)+273.15)) + intercept_g, 
                    'r--', 
                    label='Arrhenius Fit', 
                    linewidth=2
                )
                
                # WLF fit lijn
                ax_t.plot(
                    selected_temps, 
                    wlf_model([wlf_c1, wlf_c2], t_k_global, tr_k_global), 
                    'b-', 
                    label='WLF Fit', 
                    linewidth=2
                )
                
                ax_t.set_xlabel("T (°C)")
                ax_t.set_ylabel("log(aT)")
                ax_t.legend()
                ax_t.grid(True, alpha=0.3)
                st.pyplot(fig_t)
            
            with col_t2:
                st.metric("Ea (Arrhenius)", f"{ea_final:.1f} kJ/mol")
                st.write(f"**WLF C1:** {wlf_c1:.2f}")
                st.write(f"**WLF C2:** {wlf_c2:.2f}")
                if ea_final > 150:
                    st.info("💡 Hoge Ea: Dit materiaal reageert zeer gevoelig op temperatuurveranderingen in de extruder/oven.")
                st.warning("""
                **Welke te volgen?**
                * Gebruik **Arrhenius** ($E_a$) als de TPU ver boven $T_g$ is (meestal in de smelt).
                * Gebruik **WLF** als je dicht bij de glasovergang meet ($T_g < T < T_g + 100^\\circ\\text{C}$).
                """)

        with tab5:
            st.subheader("🔬 Geavanceerde TTS Validatie")
            cv1, cv2 = st.columns(2)
            
            with cv1:
                st.write("**1. Han Plot ($G'$ vs $G''$)**")
                fig_h, ax_h = plt.subplots()
                for t, color in zip(selected_temps, colors):
                    d = df[df['T_group'] == t]
                    ax_h.loglog(d['Gpp'], d['Gp'], 'o', color=color, alpha=0.6)
                ax_h.set_xlabel("G'' (Pa)")
                ax_h.set_ylabel("G' (Pa)")
                ax_h.grid(True, alpha=0.3)
                st.pyplot(fig_h)
                st.caption("Gevaar: Als de lijnen spreiden, verandert de morfologie en is TTS ongeldig.")
                st.markdown('<div class="warning-note"><b>TPU Check:</b> Zie je een opwaartse shift bij hogere temperaturen? Dat duidt op <b>thermal crosslinking</b> (na-reactie van NCO groepen).</div>', unsafe_allow_html=True)

            with cv2:
                st.write("**2. Cole-Cole Plot ($\\eta''$ vs $\\eta'$)**")
                fig_c, ax_c = plt.subplots()
                for t, color in zip(selected_temps, colors):
                    d = df[df['T_group'] == t]
                    ax_c.plot(d['Gpp']/d['omega'], d['Gp']/d['omega'], 'o-', color=color)
                ax_c.set_xlabel("η' (Pa·s)")
                ax_c.set_ylabel("η'' (Pa·s)")
                ax_c.grid(True, alpha=0.3)
                st.pyplot(fig_c)
                st.caption("Interpretatie: Een afgeplatte boog duidt op een brede molecuulgewichtsverdeling (MWD).")
            st.divider()
            st.subheader("⚖️ TTS Kwaliteitscontrole")
            
            # Eenvoudige check op R²
            if r2_final > 0.98:
                st.success(f"✅ Hoge betrouwbaarheid: R² = {r2_final:.4f}")
            elif r2_final > 0.90:
                st.warning(f"⚠️ Matige fit: R² = {r2_final:.4f}. Controleer de Van Gurp-Palmen plot.")
            else:
                st.error(f"❌ Lage betrouwbaarheid: R² = {r2_final:.4f}. TTS is waarschijnlijk niet geldig voor dit bereik.")

        with tab6:
            st.subheader("💾 Smooth Export")
            
            # Spline logic
            m_list = []
            for t in selected_temps:
                d = df[df['T_group'] == t].copy()
                at = 10**st.session_state.shifts[t]
                d['w_s'] = d['omega'] * at
                d['eta_s'] = np.sqrt(d['Gp']**2 + d['Gpp']**2) / d['w_s']
                m_list.append(d)
            
            
            m_df = pd.concat(m_list).sort_values('w_s')
            s_val = st.slider("Smoothing Sterkte", 0.0, 2.0, 0.4)
            
            eta0, gn0, cross_params = calculate_rheo_metrics(m_df)

            log_w = np.log10(m_df['w_s'])
            log_eta = np.log10(m_df['eta_s'])
            spl = UnivariateSpline(log_w, log_eta, s=s_val)
            
            w_new = np.logspace(log_w.min(), log_w.max(), 50)
            eta_new = 10**spl(np.log10(w_new))
            
            fig_s, ax_s = plt.subplots()
            ax_s.loglog(m_df['w_s'], m_df['eta_s'], 'k.', alpha=0.1, label='Raw data')
            ax_s.loglog(w_new, eta_new, 'r-', linewidth=2, label='Smoothed')
            ax_s.set_xlabel("ω·aT (rad/s)")
            ax_s.set_ylabel("η* (Pa·s)")
            ax_s.legend()
            ax_s.grid(True, alpha=0.3)
            st.pyplot(fig_s)
            
            csv = pd.DataFrame({
                'omega_shifted': w_new, 
                'eta_complex': eta_new
            }).to_csv(index=False).encode('utf-8')
            
            st.download_button(
                "Download Smooth CSV", 
                csv, 
                "mastercurve.csv",
                mime="text/csv"
            )
        with tab7:
            st.header("⚛️ Moleculaire Analyse")
        
            m1, m2, m3 = st.columns(3)
            m1.metric("Zero Shear Viscosity (η₀)", f"{eta0:.2e} Pa·s" if not np.isnan(eta0) else "N/A")
            m2.metric("Plateau Modulus (Gₙ⁰)", f"{gn0:.2e} Pa" if not np.isnan(gn0) else "N/A")
            
            # Professor's Insight over Mw
            if not np.isnan(eta0):
                # Voor TPU is de constante afhankelijk van de chemie, maar we tonen de trend
                st.info(f"💡 **Moleculair Gewicht Trend:** η₀ is evenredig met $M_w^{{3.4}}$. Een stijging van 15% in η₀ duidt op een stijging van ca. 4% in $M_w$.")

            st.divider()
            
            # Visuele extrapolatie plot
            st.subheader("Extrapolatie naar η₀ (Cross Model)")
            fig_ext, ax_ext = plt.subplots()
            w_fit = np.logspace(np.log10(m_df['w_s'].min())-2, np.log10(m_df['w_s'].max()), 100)
            
            ax_ext.loglog(master_df['w_s'], master_df['eta_s'], 'ko', alpha=0.3, label='Meetdata')
            if not np.isnan(eta0):
                ax_ext.loglog(w_fit, cross_model(w_fit, *cross_params), 'r--', label='Cross Model Fit')
                ax_ext.axhline(eta0, color='red', linestyle=':', label=f'η₀ = {eta0:.1e}')
                
            ax_ext.set_xlabel("ω·aT (rad/s)")
            ax_ext.set_ylabel("η* (Pa·s)")
            ax_ext.legend()
            st.pyplot(fig_ext)
            
            st.markdown(f"""
            <div class="expert-note">
            <b>Waarom dit cruciaal is voor TPU:</b><br>
            De <b>η₀ (Zero Shear Viscosity)</b> is de beste indicator voor de processtabiliteit. 
            Bij TPU-coatings bepaalt dit of de film egaal blijft liggen (vloei) of gaat druipen (sagging) voordat het stolt. 
            Als η₀ veel lager is dan je standaard batch, heb je waarschijnlijk last van vocht (hydrolyse) tijdens de extrusie of een te lage NCO:OH ratio.
            </div>
            """, unsafe_allow_html=True)
            
        with tab8:
            st.header("📊 Expert Dashboard")
            
            # Kolommen voor metrics
            col_a, col_b, col_c = st.columns(3)
            
            col_a.metric("Flow Activation (Ea)", f"{ea_final:.1f} kJ/mol")
            
            # Crossover berekening
            co_list = []
            for t in selected_temps:
                d = df[df['T_group'] == t].sort_values('omega')
                if len(d) > 3:
                    try:
                        f_diff = interp1d(
                            np.log10(d['omega']), 
                            np.log10(d['Gp']) - np.log10(d['Gpp']), 
                            bounds_error=False
                        )
                        w_range = np.logspace(
                            np.log10(d['omega'].min()), 
                            np.log10(d['omega'].max()), 
                            500
                        )
                        diffs = f_diff(np.log10(w_range))
                        idx_zero = np.nanargmin(np.abs(diffs))
                        if np.abs(diffs[idx_zero]) < 0.1:
                            w_co = w_range[idx_zero]
                            co_list.append({
                                "T_C": int(t), 
                                "w_co": round(w_co, 2), 
                                "Lambda_s": round(1/w_co, 4)
                            })
                    except: 
                        pass
            
            co_df = pd.DataFrame(co_list) if co_list else pd.DataFrame(columns=["T_C", "w_co", "Lambda_s"])

            if not co_df.empty:
                col_b.metric("Max Relaxatietijd", f"{co_df['Lambda_s'].max():.2e} s")
            else:
                col_b.metric("Max Relaxatietijd", "N/A")
                
            col_c.metric("TTS Fit (R²)", f"{r2_final:.4f}")
            
            st.divider()
            
            # Schone tabel voor export en weergave
            summ_df = pd.DataFrame({
                'Parameter': [
                    'Activatie-energie Ea (kJ/mol)', 
                    'WLF C1', 
                    'WLF C2', 
                    'Referentie T (°C)', 
                    'Fit Kwaliteit (R²)'
                ],
                'Waarde': [
                    round(ea_final, 2), 
                    round(float(wlf_c1), 2), 
                    round(float(wlf_c2), 2), 
                    float(ref_temp), 
                    round(r2_final, 4)
                ]
            })
            
            # Nieuwe berekening voor slopes (Terminal zone)
            # We pakken de master curve data (m_df uit de export tab logica)
            log_w = np.log10(m_df['w_s'].values)
            log_gp = np.log10(m_df['Gp'].values)
            
            # Terminal slope (lage frequentie)
            slope_term = np.polyfit(log_w[:5], log_gp[:5], 1)[0]
            
            col_d, col_e = st.columns(2)
            col_d.metric("Terminal G' Slope", f"{slope_term:.2f}", help="Ideaal voor lineaire polymeren is 2.0")
            
            # Interpretatie
            if slope_term < 1.7:
                st.error(f"⚠️ **Lage helling gedetecteerd ({slope_term:.2f})**")
                st.markdown(f"""
                **Analyse voor Productie:**
                * **Mogelijke oorzaak:** Onvolledig gesmolten harde segmenten (fysiek netwerk).
                * **Gevolg:** De coating vloeit niet goed uit.
                * **Advies:** Verhoog de verwerkingstemperatuur in zone 1 van de machine of controleer op crosslinking.
                """)
            else:
                st.success("✅ Materiaal vertoont goed vloeigedrag (lineair regime).")

            st.subheader("Overzichtstabel")
            st.table(summ_df)
            
            if not co_df.empty:
                st.subheader("Crossover Punten & Relaxatie")
                st.dataframe(co_df, use_container_width=True)

            # Excel Download
            shift_export_df = pd.DataFrame({
                'Temperatuur_C': selected_temps,
                'log_aT': [st.session_state.shifts[t] for t in selected_temps]
            })
            
            excel_data = to_excel(summ_df, shift_export_df, co_df)
            st.download_button(
                label="📥 Download Geformatteerd Excel Rapport",
                data=excel_data,
                file_name=f"RheoApp_{sample_name}_{int(ref_temp)}C.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.error("❌ Geen data gevonden in het bestand. Controleer het bestandsformaat.")
else:
    st.info("👆 Upload een frequency sweep CSV/TXT bestand om te beginnen.")
    
    with st.expander("ℹ️ Gebruiksinstructies"):
        st.markdown("""
        ### TPU Rheology Expert Tool
        
        **Features:**
        - 📈 **Master Curve**: Time-Temperature Superposition met automatische en handmatige shift factors
        - 🧪 **Van Gurp-Palmen**: Structurele analyse en thermorheologische complexiteit
        - 🧬 **Arrhenius & WLF**: Activatie-energie en glasovergang karakterisatie
        - 🔬 **Validatie**: Han plot en Cole-Cole plot voor TTS geldigheid
        - 💾 **Export**: Smooth master curves en Excel rapportage
        - 📊 **Dashboard**: Overzicht van alle kritieke parameters
        
        **Gebruik:**
        1. Upload een frequency sweep CSV
        2. Selecteer temperaturen en referentie temperatuur
        3. Klik op "🚀 Auto-Align" of pas handmatig aan
        4. Verken de verschillende tabs voor analyse
        5. Download je resultaten als CSV of Excel
        """)