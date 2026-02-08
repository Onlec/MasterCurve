import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from io import BytesIO

# --- CONFIGURATIE & STYLING ---
st.set_page_config(page_title="RheoApp", layout="wide")
st.title("RheoApp")
st.caption("-Rheologie is 50% meten en 50% gezond verstand.")
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

def find_crossover(omega, Gp, Gpp):
    """Vindt het snijpunt waar G' = G'' via log-lineaire interpolatie."""
    if len(omega) < 2: return None, None
    
    # We zoeken naar tekenwisseling van (log10(Gp) - log10(Gpp))
    diff = np.log10(Gp) - np.log10(Gpp)
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] <= 0: # Tekenwisseling gevonden
            # Interpolatie voor omega
            f_omega = interp1d([diff[i], diff[i+1]], [np.log10(omega[i]), np.log10(omega[i+1])])
            omega_co = 10**f_omega(0)
            # Interpolatie voor modulus
            f_modulus = interp1d([np.log10(omega[i]), np.log10(omega[i+1])], [np.log10(Gp[i]), np.log10(Gp[i+1])])
            modulus_co = 10**f_modulus(np.log10(omega_co))
            return omega_co, modulus_co
    return None, None

def cross_model(omega, eta_0, tau, n):
    return eta_0 / (1 + (tau * omega)**n)

def calculate_rheo_metrics(m_df):
    if m_df.empty:
        return np.nan, np.nan, [0, 0, 0], False
    
    w = m_df['w_s'].values
    eta_complex = m_df['eta_s'].values
    p0 = [eta_complex.max(), 0.1, 0.8]
    
    try:
        popt, _ = curve_fit(lambda o, e, t, n: e / (1 + (t * o)**n), w, eta_complex, p0=p0, maxfev=5000)
        eta0 = popt[0]
        # Verbeterde plateau modulus volgens Professor:
        gn0 = m_df.loc[m_df['Gpp'] == m_df['Gpp'].min(), 'Gp'].max()
        return eta0, gn0, popt, True
    except:
        return np.nan, np.nan, p0, False


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

        # Nieuwe Matplotlib colormap syntax
        cmap = mpl.colormaps[cmap_opt]
        colors = [cmap(i) for i in np.linspace(0, 1, len(selected_temps))]

        

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
            st.session_state.shifts[ref_temp] = 0.0

            for t in selected_temps:
                if t == ref_temp: 
                    continue
                def objective(log_at):
                    ref_d = df[df['T_group'] == ref_temp]
                    tgt_d = df[df['T_group'] == t]
                    f = interp1d(np.log10(ref_d['omega']), np.log10(ref_d['Gp']), bounds_error=False)
                    v = f(np.log10(tgt_d['omega']) + log_at)
                    m = ~np.isnan(v)
                    if np.sum(m) >= 2:
                        return np.sum((v[m] - np.log10(tgt_d['Gp'].values[m]))**2)
                    else:
                        return 9999 # Strafwaarde als er geen overlap is
                res = minimize(objective, x0=0.0, method='Nelder-Mead')
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
        # Genereer kleurenlijst op basis van geselecteerde optie
        import matplotlib.cm as cm
        cmap = cm.get_cmap(cmap_opt)
        colors = [cmap(i) for i in np.linspace(0, 1, len(selected_temps))]

        # Initialiseer lege dataframes voor export/dashboard (voorkomt crashes)
        summ_df = pd.DataFrame(columns=['Parameter', 'Waarde', 'Eenheid'])
        co_df = pd.DataFrame(columns=['T (°C)', 'Crossover ω (rad/s)', 'G=G\'\' (Pa)'])

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
        res_wlf = minimize(wlf_err, x0=[17.4, c2_init], bounds=[(1, 50), (10, 200)])
        wlf_c1, wlf_c2 = res_wlf.x

        # --- BEREKEN CROSSOVERS PER TEMPERATUUR ---
        co_list = []
        for t in selected_temps:
            d_t = df[df['T_group'] == t].sort_values('omega')
            w_co, g_co = find_crossover(d_t['omega'].values, d_t['Gp'].values, d_t['Gpp'].values)
            if w_co:
                co_list.append({'T (°C)': t, 'Crossover ω (rad/s)': round(w_co, 2), 'G=G\'\' (Pa)': round(g_co, 0)})
        co_df = pd.DataFrame(co_list)

        # --- 1. DATA AGGREGATIE (Cruciaal voor alle tabs) ---
        m_list = []
        for t in selected_temps:
            d = df[df['T_group'] == t].copy()
            at = 10**st.session_state.shifts[t]
            d['w_s'] = d['omega'] * at
            d['eta_s'] = np.sqrt(d['Gp']**2 + d['Gpp']**2) / d['w_s']
            
            # --- TOEGEVOEGD: Bereken delta (fasehoek) voor de Terminal Slope check ---
            d['delta'] = np.degrees(np.arctan2(d['Gpp'], d['Gp']))
            m_list.append(d)
        
        # Maak één centrale mastercurve dataframe
        m_df = pd.concat(m_list).sort_values('w_s')
        

        # --- 2. BEREKEN METRICS (Eén keer uitvoeren) ---
        eta0, gn0, fit_params, fit_success = calculate_rheo_metrics(m_df)
        
        # 1. Terminal Slope Verbetering (Professor's Delta > 75 graden criterium)
        terminal_zone = m_df[m_df['delta'] > 75]
        if len(terminal_zone) > 3:
            slope_term = np.polyfit(np.log10(terminal_zone['w_s']), np.log10(terminal_zone['Gp']), 1)[0]
        else:
            slope_term = np.nan

        # Bereken Terminal Slope (G') robuust
        # We pakken de laagste 20% van de frequentie-range voor de vloeizone
        term_idx = int(len(m_df) * 0.2)
        if term_idx > 3:
            log_w_term = np.log10(m_df['w_s'].values[:term_idx])
            log_gp_term = np.log10(m_df['Gp'].values[:term_idx])
            slope_term = np.polyfit(log_w_term, log_gp_term, 1)[0]
        else:
            slope_term = np.nan

        # Vul de summary tabel voor het dashboard
        summ_df = pd.DataFrame([
            {'Parameter': 'Activatie Energie (Ea)', 'Waarde': f"{ea_final:.2f}", 'Eenheid': 'kJ/mol'},
            {'Parameter': 'Zero Shear Viscosity', 'Waarde': f"{eta0:.2e}", 'Eenheid': 'Pa·s'},
            {'Parameter': 'WLF C1', 'Waarde': f"{wlf_c1:.2f}", 'Eenheid': '-'},
            {'Parameter': 'Terminal Slope G\'', 'Waarde': f"{slope_term:.2f}", 'Eenheid': '-'}
        ])

        # --- 3. TABS STARTEN ---
        st.subheader(f"Sample: {sample_name}")
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📈 Master Curve", "🧪 Structuur", "📉 tan δ Analyse", 
                "🧬 Thermisch (Ea/WLF)", "🔬 Validatie", 
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
                
                eta0, gn0, fit_params, fit_success = calculate_rheo_metrics(m_df)

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
            ax3.set_ylim(0, 95)
            ax3.grid(True, which="both", alpha=0.2)
            ax3.legend("Meettemperatuur")
            st.pyplot(fig3)

            st.markdown("### 🔍 Morfologische Diagnose")
            
            # Een simpele check: liggen de delta's bij de hoogste moduli dicht bij elkaar?
            st.info("""
            **Hoe lees je dit als een expert?**
            * **Lijnen vallen samen (Superpositie):** Je sample is een homogene smelt. WLF en Arrhenius zijn hier zeer betrouwbaar.
            * **Lijnen wijken af (Spreiding):** Dit is typisch voor TPU. De harde segmenten lossen op of kristalliseren uit. 
            * **De 'Bult' in de curve:** Als de curve omlaag duikt bij lage moduli, heb je te maken met een elastisch netwerk (onvolledige smelt).
            """)
            
            if len(selected_temps) > 1:
                st.warning("👉 **Observatie:** Als je hier duidelijke 'trappen' of verschuivingen tussen de kleuren ziet, verklaart dat je negatieve WLF C1 waarde. Het materiaal is thermorheologisch complex.")

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
            st.subheader("🧬 Thermische Karakterisatie: Arrhenius, WLF & VFT")
            
            # 1. Definieer de modellen
            def vft_model(T, A, B, T0):
                return A + B / (T - T0)

            # 2. Voorbereiding data
            T_vals_K = np.array(selected_temps) + 273.15
            y_vals = log_at_global
            
            # 3. Bereken VFT fit met Professor's Bounds
            vft_success = False
            try:
                p0_vft = [-10, 500, (tg_hint + 273.15) - 50]
                # Bounds: T0 moet onder de laagste meettemperatuur liggen om explosie te voorkomen
                lower_b = [-np.inf, 10, 50] 
                upper_b = [np.inf, 5000, min(T_vals_K) - 0.5]
                
                popt_vft, _ = curve_fit(vft_model, T_vals_K, y_vals, p0=p0_vft, bounds=(lower_b, upper_b), maxfev=10000)
                vft_success = True
            except:
                vft_success = False

            # --- Layout ---
            col_t1, col_t2 = st.columns([2, 1])
            
            with col_t1:
                fig_t, ax_t = plt.subplots(figsize=(10, 6))
                ax_t.scatter(selected_temps, y_vals, color='black', label='Shift Factors (Data)', s=80, zorder=5)
                
                t_smooth = np.linspace(min(selected_temps)-10, max(selected_temps)+10, 150)
                t_smooth_k = t_smooth + 273.15
                
                # Modellen plotten
                y_arr = slope_g*(1/t_smooth_k) + intercept_g
                y_wlf = wlf_model([wlf_c1, wlf_c2], t_smooth_k, tr_k_global)
                
                ax_t.plot(t_smooth, y_arr, 'r--', label='Arrhenius (Smelt-model)', alpha=0.6)
                ax_t.plot(t_smooth, y_wlf, 'b-', label='WLF (Rubber-model)', linewidth=2)
                
                if vft_success:
                    ax_t.plot(t_smooth, vft_model(t_smooth_k, *popt_vft), 'g:', label='VFT Hybride Fit', linewidth=3)

                # --- SOFTENING POINT INDICATOR ---
                # We zoeken het punt waar de modellen het meest divergeren of elkaar snijden
                diff = np.abs(y_arr - y_wlf)
                softening_idx = np.argmin(diff)
                t_softening = t_smooth[softening_idx]
                
                ax_t.axvline(t_softening, color='orange', linestyle='-.', alpha=0.5, label='Softening Transition')
                
                ax_t.set_xlabel("Temperatuur (°C)")
                ax_t.set_ylabel("log(aT)")
                ax_t.legend()
                ax_t.grid(True, alpha=0.2)
                st.pyplot(fig_t)

            with col_t2:
                st.metric("Ea (Arrhenius)", f"{ea_final:.1f} kJ/mol")
                
                # De Softening Point metric
                st.metric("Estimated Softening Point", f"{t_softening:.1f} °C")
                
                # --- DYNAMISCHE VALIDATIE ---
                st.write("---")
                st.write("**Referentie T Validatie:**")
                
                # Check 1: Ligt de referentietemperatuur in het veilige gebied?
                if ref_temp < t_softening:
                    st.error(f"⚠️ **Kritieke Waarschuwing:** Je referentietemperatuur ({ref_temp}°C) ligt **onder** het softening point ({t_softening:.1f}°C).")
                    st.markdown("""
                        <p style='color: #ff4b4b; font-size: 0.9em;'>
                        In dit gebied zijn de harde segmenten nog niet volledig gesmolten. 
                        De <b>Master Curve</b> die je nu ziet is een wiskundige benadering, 
                        maar fysisch niet 100% correct (thermorheologisch complex).
                        </p>
                    """, unsafe_allow_html=True)
                    st.info("💡 **Advies:** Kies een hogere referentietemperatuur (bijv. de hoogste meting) voor een betrouwbaardere shift.")
                else:
                    st.success(f"✅ **Referentie T is stabiel:** Je bouwt de Master Curve vanuit de homogene smeltfase ({ref_temp}°C > {t_softening:.1f}°C).")

                # Check 2: Betrouwbaarheid van de fit
                if r2_final > 0.98:
                    st.success(f"📈 Uitstekende Arrhenius fit (R²={r2_final:.3f})")
                elif r2_final < 0.90:
                    st.warning(f"📉 Zwakke fit (R²={r2_final:.3f}). De shift-factors volgen geen standaard thermisch model.")

                with st.expander("🔬 Model Parameters"):
                    st.write(f"**VFT T₀ (Vogel):** {popt_vft[2]-273.15:.1f} °C" if vft_success else "VFT: N/A")
                    st.write(f"**WLF C1 / C2:** {wlf_c1:.1f} / {wlf_c2:.1f}")

                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #ffa500;">
                <b>Professor's Eindoordeel:</b><br>
                De overgang bij <b>{t_softening:.1f}°C</b> bepaalt de wetten van je TPU. 
                Bij lagere temperaturen "vechten" de harde segmenten tegen de vloei, wat de WLF-curve doet afwijken. 
                Bij hogere temperaturen wint de entropie en regeert Arrhenius.
                </div>
                """, unsafe_allow_html=True)
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
            ax_ext.loglog(m_df['w_s'], m_df['eta_s'], 'ko', alpha=0.3, label='Meetdata')
            if fit_success and not np.isnan(eta0):
                w_fit = np.logspace(np.log10(m_df['w_s'].min())-2, np.log10(m_df['w_s'].max()), 100)
                # Bereken de fit-lijn
                eta_fit = cross_model(w_fit, fit_params[0], fit_params[1], fit_params[2])

                ax_ext.loglog(w_fit, eta_fit, 'r--', linewidth=2, label='Cross Model Fit')
                ax_ext.axhline(eta0, color='red', linestyle=':', label=f'η₀ = {eta0:.1e} Pa·s')
                st.write(f"**Gevonden η₀:** {eta0:.2e} Pa·s | **Karakteristieke tijd (τ):** {fit_params[1]:.3f} s")
            else:
                st.warning("⚠️ η₀ extrapolatie mislukt. De data is mogelijk te beperkt voor een stabiele fit.")

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
            
        with tab7:
            st.header("📊 Expert Dashboard")
            
            # --- KPI Quick-Look ---
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Flow Activation (Ea)", f"{ea_final:.1f} kJ/mol")
            col_b.metric("Zero Shear (η₀)", f"{eta0:.2e} Pa·s" if not np.isnan(eta0) else "N/A")
            col_c.metric("TTS Fit (R²)", f"{r2_final:.4f}")
            col_d.metric("Terminal Slope", f"{slope_term:.2f}" if not np.isnan(slope_term) else "N/A")

            st.divider()

            # --- BLOK 1: TOTAAL OVERZICHT PARAMETERS ---
            st.subheader("📋 1. Globale Rheologische Parameters")
            
            # We bouwen een uitgebreide tabel met alle inzichten
            dashboard_data = [
                {"Categorie": "Thermisch", "Parameter": "Activatie Energie (Ea)", "Waarde": f"{ea_final:.2f}", "Eenheid": "kJ/mol", "Info": "Gevoeligheid voor T-veranderingen"},
                {"Categorie": "Thermisch", "Parameter": "WLF C1 (Logat)", "Waarde": f"{wlf_c1:.2f}", "Eenheid": "-", "Info": "Vrije volume factor"},
                {"Categorie": "Thermisch", "Parameter": "WLF C2", "Waarde": f"{wlf_c2:.2f}", "Eenheid": "K", "Info": "Afstand tot Tg"},
                {"Categorie": "Viscositeit", "Parameter": "Zero Shear Viscosity (η₀)", "Waarde": f"{eta0:.2e}", "Eenheid": "Pa·s", "Info": "Maat voor Mw en processtabiliteit"},
                {"Categorie": "Viscositeit", "Parameter": "Relaxatietijd (τ)", "Waarde": f"{fit_params[1]:.3f}" if fit_success else "N/A", "Eenheid": "s", "Info": "Gemiddelde keten-ontwarringstijd"},
                {"Categorie": "Structuur", "Parameter": "Terminal Slope G'", "Waarde": f"{slope_term:.2f}", "Eenheid": "-", "Info": "Vloeigedrag (Ideaal = 2.0)"},
                {"Categorie": "Structuur", "Parameter": "Plateau Modulus (Gₙ⁰)", "Waarde": f"{gn0:.2e}", "Eenheid": "Pa", "Info": "Maat voor netwerkdichtheid/entanglements"},
                {"Categorie": "Validatie", "Parameter": "TTS Mastercurve Fit (R²)", "Waarde": f"{r2_final:.4f}", "Eenheid": "-", "Info": "Betrouwbaarheid van de verschuiving"}
            ]
            
            summary_table_df = pd.DataFrame(dashboard_data)
            st.table(summary_table_df)

            st.subheader("🔍 Model Betrouwbaarheid")
            
            check_col1, check_col2 = st.columns(2)
            
            with check_col1:
                st.write("**WLF Validatie:**")
                if wlf_c1 < 0 or wlf_c2 < 0:
                    st.error(f"❌ **WLF Fysiek Onmogelijk:** De gevonden waarden (C1: {wlf_c1:.2f}) zijn natuurkundig incorrect voor een smelt.")
                    st.info("💡 *Oorzaak:* Waarschijnlijk is het materiaal thermorheologisch complex (fasescheiding) of zijn de shift-factors niet vloeiend genoeg.")
                elif wlf_c1 < 5 or wlf_c1 > 30:
                    st.warning(f"⚠️ **WLF Onwaarschijnlijk:** De waarden wijken sterk af van de universele constanten. Gebruik met voorzichtigheid.")
                else:
                    st.success("✅ **WLF Stabiel:** De parameters vallen binnen het normale bereik voor polymeren.")

            with check_col2:
                st.write("**Arrhenius Validatie:**")
                if r2_final > 0.98:
                    st.success(f"✅ **Arrhenius dominant:** De lineaire fit is uitstekend (R²: {r2_final:.4f}).")
                else:
                    st.warning(f"⚠️ **Arrhenius afwijking:** R² is {r2_final:.4f}. Mogelijk fase-overgangen in dit T-bereik.")

            # --- DE DIAGNOSE (Inzichten uit alle tabs) ---
            st.subheader("🧠 2. Professor's Diagnose")
            
            diag_col1, diag_col2 = st.columns(2)
            
            with diag_col1:
                st.markdown("**Verwerkingsadvies:**")
                # Inzicht uit Terminal Slope & Eta0
                if not np.isnan(slope_term) and slope_term < 1.7:
                    st.error(f"⚠️ **Vloeiprobleem:** De lage slope ({slope_term:.2f}) duidt op onvolledige smelt of crosslinking. Pas op voor 'sharkskin' of inhomogeniteiten.")
                else:
                    st.success("✅ **Goede smeltkwaliteit:** Het materiaal vloeit Newtoniaans in de terminale zone.")
                
                # Inzicht uit Ea
                if ea_final > 150:
                    st.warning(f"🌡️ **Hoge T-gevoeligheid:** Kleine variaties in de extruder-T veroorzaken grote viscositeitsschommelingen.")

            with diag_col2:
                st.markdown("**Structurele Integriteit:**")
                # Inzicht uit TTS fit
                if r2_final < 0.95:
                    st.error("❌ **Thermorheologisch Complex:** TTS fit is matig. Dit wijst op faseveranderingen tijdens de meting (bijv. kristallisatie van harde segmenten).")
                else:
                    st.success("✅ **Homogene Smelt:** Het materiaal volgt TTS perfect; stabiele fase-morfologie.")
                
                # Inzicht uit η₀
                st.info(f"🧬 **Mw Indicatie:** De η₀ van {eta0:.1e} Pa·s is je referentiepunt. Lagere waarden bij volgende batches wijzen vaak op hydrolyse.")

            st.divider()

            # --- BLOK 2: CROSSOVER & EXPORTS ---
            st.subheader("⚖️ 3. Crossover Punten")
            if not co_df.empty:
                st.dataframe(co_df, use_container_width=True)
            
            st.divider()
            st.subheader("💾 Data Export (CSV)")
            
            col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)

            # 1. Summary CSV
            col_ex1.download_button("📊 Summary CSV", summary_table_df.to_csv(index=False).encode('utf-8'), f"{sample_name}_Summary.csv", "text/csv")

            # 2. Shifts CSV
            shift_export_df = pd.DataFrame({'T_C': selected_temps, 'log_aT': [st.session_state.shifts[t] for t in selected_temps]})
            col_ex2.download_button("🕒 Shifts CSV", shift_export_df.to_csv(index=False).encode('utf-8'), f"{sample_name}_Shifts.csv", "text/csv")

            # 3. Crossovers CSV
            if not co_df.empty:
                col_ex3.download_button("⚖️ Crossovers CSV", co_df.to_csv(index=False).encode('utf-8'), f"{sample_name}_Crossovers.csv", "text/csv")

            # --- 4. Master Curve CSV (Foutbestendige versie) ---
            # We definiëren welke kolommen we ZOUDEN WILLEN hebben
            gewenste_kolommen = {
                'w_s': 'omega_shifted_rad_s',
                'Gp': 'Gp_Pa',
                'Gpp': 'Gpp_Pa',
                'eta_s': 'Complex_Visc_Pas',
                'delta': 'PhaseAngle_deg',
                'T_group': 'Original_T_C'
            }

            # Check welke van deze kolommen daadwerkelijk in m_df zitten
            beschikbare_kolommen = [k for k in gewenste_kolommen.keys() if k in m_df.columns]
            
            # Maak de export dataframe enkel met wat we hebben
            master_export_df = m_df[beschikbare_kolommen].copy()
            
            # Hernoem de kolommen naar de nette namen
            master_export_df = master_export_df.rename(columns=gewenste_kolommen)
            
            # Voeg tan_delta handmatig toe als Gp en Gpp er zijn (vaak berekend)
            if 'Gp' in m_df.columns and 'Gpp' in m_df.columns:
                master_export_df['tan_delta'] = m_df['Gpp'] / m_df['Gp']
            
            csv_master = master_export_df.to_csv(index=False).encode('utf-8')
            col_ex4.download_button(
                label="📈 Master Curve CSV",
                data=csv_master,
                file_name=f"{sample_name}_MasterCurve.csv",
                mime="text/csv"
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