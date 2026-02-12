import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from io import BytesIO
import json
import seaborn as sns


# --- LANGUAGE & TRANSLATIONS SETUP ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'NL'

@st.cache_data
def load_translations():
    try:
        with open('translations.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ Translation file 'translations.json' not found!")
        return {"NL": {"main_app": {}}, "EN": {"main_app": {}}}

all_translations = load_translations()
texts = all_translations.get(st.session_state.lang, all_translations["NL"]).get("main_app", {})


# --- CONFIGURATIE & STYLING ---
st.title(texts.get("title", "RheoApp - TPU Rheology Expert Tool"))
st.caption(texts.get("caption", ""))

st.logo("assets/logo.png", size="large", icon_image="assets/logo.png")


with st.expander(texts.get("disclaimer_title", "⚠️ Disclaimer")):
    st.warning(texts.get("disclaimer_text", ""))

st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 2rem; }
    .expert-note { background-color: #f0f2f6; padding: 15px; border-left: 5px solid #ff4b4b; border-radius: 5px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- GLOBALE GRAFIEK STIJL ---
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    'figure.facecolor': 'white',
    'figure.dpi': 110,
    'axes.facecolor': '#f8f8f8',
    'axes.edgecolor': '#cccccc',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '#e5e5e5',
    'grid.linewidth': 0.8,
    'grid.alpha': 1.0,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.color': '#555555',
    'ytick.color': '#555555',
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#dddddd',
    'legend.fancybox': True,
    'font.family': 'sans-serif',
    'font.size': 10,
})


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
            row_3 = lines[2].split('\t')
            if len(row_3) >= 2:
                sample_name = row_3[1].strip()
                return sample_name if sample_name else "Onbekend_Sample"
        return "Onbekend_Sample"
    except Exception as e:
        return f"Error_bij_lezen_{e}"


def to_excel(summary_df, shift_df, crossover_df):
    output = BytesIO()
    summary_df = summary_df.copy()
    summary_df['Waarde'] = summary_df['Waarde'].apply(
        lambda x: float(x) if isinstance(x, (np.float64, np.float32, np.ndarray)) else str(x)
    )
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        shift_df.to_excel(writer, sheet_name='ShiftFactors', index=False)
        crossover_df.to_excel(writer, sheet_name='Crossovers', index=False)
        for sheetname in writer.sheets:
            writer.sheets[sheetname].set_column('A:C', 20)
    return output.getvalue()


def find_crossover(omega, Gp, Gpp):
    if len(omega) < 2:
        return None, None
    diff = np.log10(Gp) - np.log10(Gpp)
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] <= 0:
            f_omega = interp1d([diff[i], diff[i+1]], [np.log10(omega[i]), np.log10(omega[i+1])])
            omega_co = 10**f_omega(0)
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
        plateau_zone = m_df[m_df['Gp'] > 2 * m_df['Gpp']]
        if len(plateau_zone) > 3:
            gn0 = plateau_zone['Gp'].median()
        else:
            gn0 = m_df['Gp'].max()
        return eta0, gn0, popt, True
    except:
        return np.nan, np.nan, p0, False


def find_all_crossovers(omega, Gp, Gpp):
    crossovers = []
    log_gp = np.log10(Gp)
    log_gpp = np.log10(Gpp)
    diff = log_gp - log_gpp
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            frac = abs(diff[i]) / (abs(diff[i]) + abs(diff[i+1]))
            omega_co = 10**(np.log10(omega[i]) + frac * (np.log10(omega[i+1]) - np.log10(omega[i])))
            modulus_co = 10**(log_gp[i] + frac * (log_gp[i+1] - log_gp[i]))
            crossovers.append({"omega": omega_co, "modulus": modulus_co})
    return crossovers


# --- SIDEBAR: CONTROLS ---
st.sidebar.title(texts.get("sidebar_title", "🎛️ Control Panel"))
st.sidebar.caption(texts.get("sidebar_caption", ""))

uploaded_file = st.sidebar.file_uploader(
    texts.get("upload_label", "📁 Upload Frequency Sweep"),
    type=['csv', 'txt']
)

if uploaded_file:
    df = load_rheo_data(uploaded_file)
    sample_name = extract_sample_name(uploaded_file)

    if not df.empty:
        df['T_group'] = df['T'].round(0)
        temps = sorted(df['T_group'].unique())

        selected_temps = st.sidebar.multiselect(
            texts.get("select_temps", "🌡️ Select Temperatures"),
            temps,
            default=temps,
            help=texts.get("select_temps_help", "")
        )

        if len(selected_temps) < 3:
            st.sidebar.warning(texts.get("select_temps_warning", "⚠️ Select at least 3 temperatures!"))

        ref_temp = st.sidebar.selectbox(
            texts.get("ref_temp_label", "📌 Reference Temperature (°C)"),
            selected_temps,
            index=len(selected_temps) - 1,
            help=texts.get("ref_temp_help", "")
        )

        cmap_opt = st.sidebar.selectbox(
            texts.get("colorscheme", "🎨 Color Scheme"),
            ["coolwarm", "viridis", "magma", "jet"]
        )

        cmap = mpl.colormaps[cmap_opt]
        colors = [cmap(i) for i in np.linspace(0, 1, len(selected_temps))]

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**{texts.get('wlf_section', '⚙️ WLF Parameters')}**")

        tg_hint = st.sidebar.number_input(
            texts.get("expected_tg", "Expected Tg (°C)"),
            value=-40.0,
            help=texts.get("expected_tg_help", "")
        )
        st.sidebar.caption(texts.get("tg_tip", ""))

        if 'shifts' not in st.session_state:
            st.session_state.shifts = {t: 0.0 for t in temps}
        if 'reset_id' not in st.session_state:
            st.session_state.reset_id = 0

        c_auto, c_reset = st.sidebar.columns(2)

        if c_reset.button(texts.get("reset_button", "🔄 Reset"), help=texts.get("reset_help", "")):
            for t in temps:
                st.session_state.shifts[t] = 0.0
            st.session_state.reset_id += 1
            st.rerun()

        if c_auto.button(texts.get("auto_align", "🚀 Auto-Align"), help=texts.get("auto_align_help", "")):
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
                        return 9999
                res = minimize(objective, x0=0.0, method='Nelder-Mead')
                st.session_state.shifts[t] = round(float(res.x[0]), 2)
            st.session_state.reset_id += 1
            st.rerun()

        st.sidebar.markdown(f"**{texts.get('manual_shifts', '🎚️ Manual Shift Factors')}**")
        for t in selected_temps:
            st.session_state.shifts[t] = st.sidebar.number_input(
                f"{int(t)}°C",
                min_value=-10.0,
                max_value=10.0,
                value=float(st.session_state.shifts[t]),
                step=0.1,
                format="%.2f",
                key=f"{t}_{st.session_state.reset_id}"
            )

        st.sidebar.divider()
        st.sidebar.markdown(f"**{texts.get('help_section_title', '📚 Need Help?')}**")
        st.sidebar.info(texts.get("help_section_text", ""))

        # ============================================================
        # CENTRALE DATA AGGREGATIE
        # ============================================================
        m_list = []
        for t in selected_temps:
            d = df[df['T_group'] == t].copy()
            at = 10**st.session_state.shifts[t]
            d['w_s'] = d['omega'] * at
            d['eta_s'] = np.sqrt(d['Gp']**2 + d['Gpp']**2) / d['w_s']
            d['delta'] = np.degrees(np.arctan2(d['Gpp'], d['Gp']))
            m_list.append(d)

        m_df = pd.concat(m_list).sort_values('w_s')

        # ============================================================
        # BEREKENINGEN
        # ============================================================
        t_k_global = np.array([t + 273.15 for t in selected_temps])
        log_at_global = np.array([st.session_state.shifts[t] for t in selected_temps])
        tr_k_global = ref_temp + 273.15

        inv_t_global = 1 / t_k_global
        slope_g, intercept_g = np.polyfit(inv_t_global, log_at_global, 1)
        ea_final = float(abs(slope_g * 8.314 * np.log(10) / 1000))

        residuals = log_at_global - (slope_g * inv_t_global + intercept_g)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_at_global - np.mean(log_at_global))**2)
        r2_final = float(1 - ss_res / ss_tot)

        n_points = len(log_at_global)
        r2_adj = 1 - (1 - r2_final) * (n_points - 1) / max(n_points - 2, 1)

        def wlf_model(p, t, tr):
            return -p[0] * (t - tr) / (p[1] + (t - tr))

        def wlf_err(p):
            return np.sum((log_at_global - wlf_model(p, t_k_global, tr_k_global))**2)

        c2_init = max(50.0, ref_temp - tg_hint)
        res_wlf = minimize(wlf_err, x0=[17.4, c2_init], bounds=[(1, 50), (10, 200)])
        wlf_c1, wlf_c2 = res_wlf.x

        def vft_model(T, A, B, T0):
            return A + B / (T - T0)

        vft_success = False
        try:
            p0_vft = [-10, 500, (tg_hint + 273.15) - 50]
            lower_b = [-np.inf, 10, 50]
            upper_b = [np.inf, 5000, min(t_k_global) - 5]
            popt_vft, _ = curve_fit(vft_model, t_k_global, log_at_global,
                                    p0=p0_vft, bounds=(lower_b, upper_b), maxfev=10000)
            vft_success = True
        except:
            popt_vft = [np.nan, np.nan, np.nan]

        if vft_success:
            t_inf_c = popt_vft[2] - 273.15
            t_inf_info = "VFT gefitte Vogel temp"
        else:
            t_inf_c = ref_temp - wlf_c2
            t_inf_info = "Geschat (T_ref - C2)"

        eta0, gn0_raw, fit_params, fit_success = calculate_rheo_metrics(m_df)

        plateau_zone = m_df[m_df['Gp'] > 2 * m_df['Gpp']]
        if len(plateau_zone) > 3:
            gn0 = plateau_zone['Gp'].median()
            gn0_info = "Mediaan elastisch regime (G' > 2G'')"
        else:
            gn0 = m_df['Gp'].max()
            gn0_info = "Max G' (plateau niet bereikt)"

        cutoff_freq = m_df['w_s'].quantile(0.3)
        terminal_zone = m_df[(m_df['delta'] > 75) & (m_df['w_s'] <= cutoff_freq)]
        if len(terminal_zone) >= 3:
            slope_term = np.polyfit(np.log10(terminal_zone['w_s']),
                                    np.log10(terminal_zone['Gp']), 1)[0]
            slope_info = f"Berekend uit {len(terminal_zone)} punten (δ>75°, laagste 30% freq)"
        else:
            slope_term = np.nan
            slope_info = "Onvoldoende data voor terminal zone"

        co_list = []
        for t in selected_temps:
            d_t = df[df['T_group'] == t].sort_values('omega')
            crossovers = find_all_crossovers(d_t['omega'].values, d_t['Gp'].values, d_t['Gpp'].values)
            if crossovers:
                co_list.append({
                    'T (°C)': t,
                    'Crossover ω (rad/s)': round(crossovers[0]['omega'], 2),
                    'G=G\'\' (Pa)': round(crossovers[0]['modulus'], 0),
                    'Aantal crossovers': len(crossovers)
                })

        co_df = pd.DataFrame(co_list)
        all_cos_master = find_all_crossovers(m_df['w_s'].values, m_df['Gp'].values, m_df['Gpp'].values)
        num_cos = len(all_cos_master)

        t_smooth = np.linspace(min(selected_temps) - 10, max(selected_temps) + 10, 150)
        t_smooth_k = t_smooth + 273.15
        y_arr = slope_g * (1 / t_smooth_k) + intercept_g
        y_wlf = wlf_model([wlf_c1, wlf_c2], t_smooth_k, tr_k_global)
        diff = np.abs(y_arr - y_wlf)
        softening_idx = np.argmin(diff)
        t_softening = t_smooth[softening_idx]

        summ_df = pd.DataFrame([
            {'Parameter': 'Activatie Energie (Ea)', 'Waarde': f"{ea_final:.2f}", 'Eenheid': 'kJ/mol'},
            {'Parameter': 'Zero Shear Viscosity (η₀)', 'Waarde': f"{eta0:.2e}" if not np.isnan(eta0) else "N/A", 'Eenheid': 'Pa·s'},
            {'Parameter': 'Plateau Modulus (Gₙ⁰)', 'Waarde': f"{gn0:.2e}" if not np.isnan(gn0) else "N/A", 'Eenheid': 'Pa'},
            {'Parameter': 'WLF C1', 'Waarde': f"{wlf_c1:.2f}", 'Eenheid': '-'},
            {'Parameter': 'WLF C2', 'Waarde': f"{wlf_c2:.2f}", 'Eenheid': 'K'},
            {'Parameter': "Terminal Slope G'", 'Waarde': f"{slope_term:.2f}" if not np.isnan(slope_term) else "N/A", 'Eenheid': '-'},
            {'Parameter': 'Arrhenius R²', 'Waarde': f"{r2_final:.4f}", 'Eenheid': '-'},
            {'Parameter': 'Adjusted R²', 'Waarde': f"{r2_adj:.4f}", 'Eenheid': '-'}
        ])

        # ============================================================
        # TABS
        # ============================================================
        st.subheader(f"Sample: {sample_name}")
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            texts.get("tab1_name", "📈 Master Curve"),
            texts.get("tab2_name", "🧪 Structure (vGP)"),
            texts.get("tab3_name", "📉 tan δ Analysis"),
            texts.get("tab4_name", "🌡️ Thermal (Ea/WLF/VFT)"),
            texts.get("tab5_name", "🔬 TTS Validation"),
            texts.get("tab6_name", "🧬 Molecular Analysis"),
            texts.get("tab7_name", "📊 Dashboard"),
        ])

        # ============================================================
        # TAB 1: MASTER CURVE
        # fig1: G' en G'' vs verschoven frequentie (log-log)
        # fig_s: Smoothed η* vs verschoven frequentie
        # fig2: Shift factor trend (log aT vs T)
        # ============================================================
        with tab1:
            st.subheader(texts.get("tab1_title", "Master Curve at {temp}°C").format(temp=ref_temp))
            st.info(texts.get("tab1_info", ""))

            col_m1, col_m2 = st.columns([2, 1])

            with col_m1:
                # --- fig1: Master Curve G' en G'' ---
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                for t, color in zip(selected_temps, colors):
                    d = df[df['T_group'] == t].copy()
                    at = 10**st.session_state.shifts[t]
                    ax1.loglog(d['omega'] * at, d['Gp'], 'o-',
                               color=color, label=f"{int(t)}°C G'")
                    ax1.loglog(d['omega'] * at, d['Gpp'], 'x--',
                               color=color, alpha=0.5, label=f"{int(t)}°C G''")
                ax1.set_xlabel("ω·aT (rad/s)")
                ax1.set_ylabel("Modulus (Pa)")
                ax1.legend(ncol=2, title="Temperatuur")
                ax1.grid(True, which="both")
                st.pyplot(fig1)
                plt.close()

                st.subheader(texts.get("smooth_export", "💾 Smooth Export (Optional)"))
                st.caption(texts.get("smooth_caption", ""))

                s_val = st.slider(texts.get("smooth_strength", "Smoothing Strength"), 0.0, 2.0, 0.4)
                st.warning(texts.get("smooth_warning", ""))

                log_w = np.log10(m_df['w_s'])
                log_eta = np.log10(m_df['eta_s'])
                spl = UnivariateSpline(log_w, log_eta, s=s_val)
                w_new = np.logspace(log_w.min(), log_w.max(), 50)
                eta_new = 10**spl(np.log10(w_new))

                # --- fig_s: Smoothed viscositeit ---
                fig_s, ax_s = plt.subplots(figsize=(10, 5))
                ax_s.loglog(m_df['w_s'], m_df['eta_s'], 'o',
                            color='#aaaaaa', alpha=0.4, markersize=3, label='Raw data')
                ax_s.loglog(w_new, eta_new, '-',
                            color='#e63946', linewidth=2.5, label='Smoothed')
                ax_s.set_xlabel("ω·aT (rad/s)")
                ax_s.set_ylabel("η* (Pa·s)")
                ax_s.legend()
                ax_s.grid(True, which="both")
                st.pyplot(fig_s)
                plt.close()

            with col_m2:
                st.write(f"**{texts.get('shift_trend_title', '📊 Shift Factor Trend')}**")
                t_plot = sorted(selected_temps)
                s_plot = [st.session_state.shifts[t] for t in t_plot]

                # --- fig2: Shift factor trend ---
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                ax2.plot(t_plot, s_plot, 's-', color='#e63946', linewidth=2)
                ax2.axhline(0, color='#aaaaaa', linestyle='--', linewidth=1)
                ax2.set_xlabel("T (°C)")
                ax2.set_ylabel("log(aT)")
                ax2.grid(True)
                st.pyplot(fig2)
                plt.close()
                st.info(texts.get("shift_trend_info", ""))

                st.subheader(texts.get("quick_stats", "🎯 Quick Stats"))
                st.metric(texts.get("temperatures", "Temperatures"), len(selected_temps))
                st.metric(texts.get("data_points", "Data Points"), len(m_df))
                st.metric(texts.get("freq_range", "Freq Range"),
                          f"{m_df['w_s'].min():.2f} – {m_df['w_s'].max():.0f}")

        # ============================================================
        # TAB 2: VAN GURP-PALMEN
        # fig3: δ vs |G*| (semi-log)
        # ============================================================
        with tab2:
            st.subheader(texts.get("tab2_title", "Van Gurp-Palmen (vGP) Structure Analysis"))
            st.markdown(texts.get("tab2_intro", ""))

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
            ax3.grid(True, which="both")
            ax3.legend(title="T (°C)", ncol=2)
            st.pyplot(fig3)
            plt.close()

            if len(selected_temps) > 1:
                st.warning(texts.get("vgp_warning", ""))

            st.markdown(f"### {texts.get('morphology_title', '🔍 Morphological Diagnosis')}")
            morph_col1, morph_col2 = st.columns(2)
            with morph_col1:
                st.success(texts.get("morphology_good", ""))
            with morph_col2:
                st.error(texts.get("morphology_bad", ""))

        # ============================================================
        # TAB 3: TAN DELTA
        # fig_tan: tan δ vs frequentie (semi-log)
        # ============================================================
        with tab3:
            st.subheader(texts.get("tab3_title", "Loss Tangent (tan δ) - Relaxation Spectrum"))
            st.info(texts.get("tab3_info", ""))

            fig_tan, ax_tan = plt.subplots(figsize=(10, 5))
            for t, color in zip(selected_temps, colors):
                d = df[df['T_group'] == t]
                tan_d = d['Gpp'] / d['Gp']
                ax_tan.semilogx(d['omega'], tan_d, 'o-', color=color, label=f"{int(t)}°C")
            ax_tan.axhline(1, color='#e63946', linestyle='--', linewidth=1.5,
                           label="tan δ = 1 (G' = G'')")
            ax_tan.set_xlabel("ω (rad/s)")
            ax_tan.set_ylabel("tan δ (-)")
            ax_tan.legend(ncol=2, title="Temperatuur")
            ax_tan.grid(True)
            st.pyplot(fig_tan)
            plt.close()

            st.markdown(f"**{texts.get('tab3_table_title', '💡 Interpretation for TPU')}**")
            st.markdown(texts.get("tab3_table", ""))

        # ============================================================
        # TAB 4: THERMAL
        # fig_t: log(aT) vs T met Arrhenius, WLF, VFT fits
        # ============================================================
        with tab4:
            st.subheader(texts.get("tab4_title", "Thermal Characterization: Arrhenius, WLF & VFT"))

            col_t1, col_t2 = st.columns([2, 1])

            with col_t1:
                fig_t, ax_t = plt.subplots(figsize=(10, 6))
                ax_t.scatter(selected_temps, log_at_global, color='black',
                             label='Shift Factors (data)', s=80, zorder=5)
                ax_t.plot(t_smooth, y_arr, '--', color='#e63946',
                          linewidth=2, label='Arrhenius')
                ax_t.plot(t_smooth, y_wlf, '-', color='#457b9d',
                          linewidth=2.5, label='WLF')
                if vft_success:
                    ax_t.plot(t_smooth, vft_model(t_smooth_k, *popt_vft),
                              ':', color='#2a9d8f', linewidth=2.5, label='VFT')
                ax_t.axvline(t_softening, color='#f4a261', linestyle='-.',
                             linewidth=1.5, label=f'Softening (~{t_softening:.0f}°C)')
                ax_t.set_xlabel("T (°C)")
                ax_t.set_ylabel("log(aT)")
                ax_t.legend()
                ax_t.grid(True)
                st.pyplot(fig_t)
                plt.close()

            with col_t2:
                m = texts.get("tab4_metrics", {})
                st.metric(m.get("ea", "Ea (Arrhenius):"), f"{ea_final:.1f} kJ/mol")
                st.metric(m.get("softening", "Softening Point:"), f"{t_softening:.1f} °C")
                st.metric(
                    m.get("vft_t0", "VFT T₀:"),
                    f"{popt_vft[2] - 273.15:.1f} °C" if vft_success else m.get("vft_na", "VFT: N/A")
                )
                st.metric(m.get("wlf_c1", "WLF C1:"), f"{wlf_c1:.1f}")
                st.metric(m.get("wlf_c2", "WLF C2:"), f"{wlf_c2:.1f}")

                st.divider()
                st.write(f"**{texts.get('tab4_validation_title', '⚠️ Reference Temperature Validation')}**")

                if ref_temp < t_softening:
                    st.error(texts.get("tab4_critical_warning", "").format(
                        ref_temp=ref_temp, t_soft=t_softening,
                        t_max=max(selected_temps), t_req=t_softening + 10
                    ))
                else:
                    st.success(texts.get("tab4_success", "").format(
                        ref_temp=ref_temp, t_soft=t_softening
                    ))

                if r2_final > 0.98:
                    st.success(f"📈 R²={r2_final:.3f}")
                elif r2_final < 0.90:
                    st.warning(f"📉 R²={r2_final:.3f}")

        # ============================================================
        # TAB 5: TTS VALIDATION
        # fig_h: Han plot G' vs G'' (log-log)
        # fig_c: Cole-Cole η'' vs η'
        # ============================================================
        with tab5:
            st.subheader(texts.get("tab5_title", "TTS Validation via Han & Cole-Cole Plots"))

            cv1, cv2 = st.columns(2)

            with cv1:
                st.write(texts.get("tab5_han_title", "1️⃣ Han Plot: G' vs G''"))
                fig_h, ax_h = plt.subplots(figsize=(6, 5))
                for t, color in zip(selected_temps, colors):
                    d = df[df['T_group'] == t]
                    ax_h.loglog(d['Gpp'], d['Gp'], 'o-', color=color,
                                alpha=0.8, label=f"{int(t)}°C")
                ax_h.set_xlabel("G'' (Pa)")
                ax_h.set_ylabel("G' (Pa)")
                ax_h.legend(title="T (°C)", ncol=2)
                ax_h.grid(True, which="both")
                st.pyplot(fig_h)
                plt.close()
                st.caption(texts.get("tab5_han_caption", ""))

            with cv2:
                st.write(texts.get("tab5_cole_title", "2️⃣ Cole-Cole Plot: η'' vs η'"))
                fig_c, ax_c = plt.subplots(figsize=(6, 5))
                for t, color in zip(selected_temps, colors):
                    d = df[df['T_group'] == t]
                    ax_c.plot(d['Gpp'] / d['omega'], d['Gp'] / d['omega'],
                              'o-', color=color, label=f"{int(t)}°C")
                ax_c.set_xlabel("η' (Pa·s)")
                ax_c.set_ylabel("η'' (Pa·s)")
                ax_c.legend(title="T (°C)", ncol=2)
                ax_c.grid(True)
                st.pyplot(fig_c)
                plt.close()
                st.caption(texts.get("tab5_cole_caption", ""))

            st.divider()
            st.subheader(texts.get("tab5_quality_title", "⚖️ TTS Quality Control Summary"))

            q_col1, q_col2, q_col3 = st.columns(3)

            with q_col1:
                st.markdown(texts.get("tab5_r2_label", "**📊 Arrhenius R²**"))
                if r2_final > 0.98:
                    st.success(f"{texts.get('tab5_excellent', '✅ Excellent')} ({r2_final:.4f})")
                elif r2_final > 0.90:
                    st.warning(f"{texts.get('tab5_moderate', '⚠️ Moderate')} ({r2_final:.4f})")
                else:
                    st.error(f"{texts.get('tab5_weak', '❌ Weak')} ({r2_final:.4f})")

            with q_col2:
                st.markdown(texts.get("tab5_slope_label", "**📐 Terminal Slope**"))
                if not np.isnan(slope_term):
                    if slope_term >= 1.8:
                        st.success(f"{texts.get('tab5_good', '✅ Newtonian')} ({slope_term:.2f})")
                    elif slope_term >= 1.5:
                        st.warning(f"{texts.get('tab5_moderate', '⚠️ Moderate')} ({slope_term:.2f})")
                    else:
                        st.error(f"{texts.get('tab5_problem', '❌ Problem')} ({slope_term:.2f})")
                else:
                    st.info(texts.get("tab5_not_reached", "ℹ️ Not reached"))

            with q_col3:
                st.markdown(texts.get("tab5_cross_label", "**⚖️ Crossovers**"))
                if num_cos == 1:
                    st.success(f"{texts.get('tab5_single', '✅ Single')} (n={num_cos})")
                elif num_cos == 0:
                    st.warning(f"{texts.get('tab5_none', '⚠️ None')} (n={num_cos})")
                else:
                    st.error(f"{texts.get('tab5_multiple', '❌ Multiple')} (n={num_cos})")

        # ============================================================
        # TAB 6: MOLECULAR ANALYSIS
        # fig_ext: η* vs verschoven frequentie met Cross model fit
        # ============================================================
        with tab6:
            st.header(texts.get("tab6_title", "⚛️ Molecular Analysis & Process Parameters"))
            st.markdown(texts.get("tab6_intro", ""))

            m1, m2, m3 = st.columns(3)
            m1.metric(
                texts.get("tab6_eta0", "Zero Shear Viscosity (η₀)"),
                f"{eta0:.2e} Pa·s" if not np.isnan(eta0) else "N/A"
            )
            m2.metric(
                texts.get("tab6_gn0", "Plateau Modulus (Gₙ⁰)"),
                f"{gn0:.2e} Pa" if not np.isnan(gn0) else "N/A"
            )
            if fit_success:
                m3.metric(
                    texts.get("tab6_tau", "Relaxation Time (τ)"),
                    f"{fit_params[1]:.3f} s"
                )

            if not np.isnan(eta0):
                st.markdown(f"### {texts.get('tab6_mw_title', '🧬 Molecular Weight Relationship')}")
                mw_col1, mw_col2 = st.columns([2, 1])
                with mw_col1:
                    st.info(f"""
                    **η₀ ∝ M_w^3.4** (voor lineaire polymeren)

                    Dit betekent dat η₀ **extreem gevoelig** is voor Mw veranderingen:

                    | Δη₀ | ΔM_w (geschat) | Mogelijke Oorzaak |
                    |-----|----------------|-------------------|
                    | +15% | +4% | Langere ketens |
                    | -20% | -6% | **Hydrolyse!** |
                    | -50% | -15% | **Ernstige degradatie** |
                    """)
                with mw_col2:
                    st.success(f"""
                    **Huidige η₀:**
                    {eta0:.2e} Pa·s

                    **Typisch TPU:**
                    - 10⁴-10⁵: Laag Mw
                    - 10⁵-10⁶: Normaal
                    - > 10⁶: Hoog Mw
                    """)

            st.divider()
            st.subheader("Extrapolatie naar η₀ (Cross Model)")

            fig_ext, ax_ext = plt.subplots(figsize=(10, 5))
            ax_ext.loglog(m_df['w_s'], m_df['eta_s'], 'o',
                          color='#aaaaaa', alpha=0.4, markersize=3, label='Meetdata')
            if fit_success and not np.isnan(eta0):
                w_fit = np.logspace(np.log10(m_df['w_s'].min()) - 2,
                                    np.log10(m_df['w_s'].max()), 100)
                eta_fit = cross_model(w_fit, fit_params[0], fit_params[1], fit_params[2])
                ax_ext.loglog(w_fit, eta_fit, '--', color='#e63946',
                              linewidth=2.5, label='Cross Model Fit')
                ax_ext.axhline(eta0, color='#457b9d', linestyle=':',
                               linewidth=2, label=f'η₀ = {eta0:.1e} Pa·s')
                st.write(f"**Gevonden η₀:** {eta0:.2e} Pa·s | **Karakteristieke tijd (τ):** {fit_params[1]:.3f} s")
            else:
                st.warning("⚠️ η₀ extrapolatie mislukt.")
            ax_ext.set_xlabel("ω·aT (rad/s)")
            ax_ext.set_ylabel("η* (Pa·s)")
            ax_ext.legend()
            ax_ext.grid(True, which="both")
            st.pyplot(fig_ext)
            plt.close()

        # ============================================================
        # TAB 7: DASHBOARD
        # ============================================================
        with tab7:
            st.header(texts.get("tab7_title", "📊 Expert Dashboard - Consolidated Analysis"))
            st.markdown(texts.get("tab7_intro", ""))

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Flow Activation (Ea)", f"{ea_final:.1f} kJ/mol")
            col_b.metric("Zero Shear (η₀)", f"{eta0:.2e} Pa·s" if not np.isnan(eta0) else "N/A")
            col_c.metric("TTS Adj. R²", f"{r2_adj:.4f}")
            col_d.metric("Crossovers", f"{num_cos}", delta="Complex" if num_cos > 1 else "Simpel")

            st.divider()
            st.subheader(texts.get("tab7_parameters", "📋 Complete Parameter Overview"))

            dashboard_data = [
                {"Categorie": "Thermisch", "Parameter": "Activatie Energie (Ea)",
                 "Waarde": f"{ea_final:.2f}", "Eenheid": "kJ/mol", "Info": "Vloei-activatie energie"},
                {"Categorie": "Thermisch", "Parameter": "WLF C₁",
                 "Waarde": f"{wlf_c1:.2f}", "Eenheid": "-", "Info": "Vrije volume parameter"},
                {"Categorie": "Thermisch", "Parameter": "WLF C₂",
                 "Waarde": f"{wlf_c2:.2f}", "Eenheid": "K", "Info": "Temp-afstand tot Tg"},
                {"Categorie": "Thermisch", "Parameter": "VFT T∞ (Vogel Temp)",
                 "Waarde": f"{t_inf_c:.1f}", "Eenheid": "°C", "Info": t_inf_info},
                {"Categorie": "Thermisch", "Parameter": "Geschatte Tg",
                 "Waarde": f"{t_inf_c + 50:.1f}", "Eenheid": "°C", "Info": "T∞ + 50K regel voor TPU"},
                {"Categorie": "Viscositeit", "Parameter": "Zero Shear Viscosity (η₀)",
                 "Waarde": f"{eta0:.2e}" if not np.isnan(eta0) else "N/A", "Eenheid": "Pa·s",
                 "Info": "Processtabiliteits-indicator"},
                {"Categorie": "Viscositeit", "Parameter": "Relaxatietijd (τ)",
                 "Waarde": f"{fit_params[1]:.3f}" if fit_success else "N/A", "Eenheid": "s",
                 "Info": "Keten-ontwarringstijd uit Cross model"},
                {"Categorie": "Structuur", "Parameter": "Terminal Slope G'",
                 "Waarde": f"{slope_term:.2f}" if not np.isnan(slope_term) else "N/A", "Eenheid": "-",
                 "Info": slope_info + " (Ideaal: 2.0)"},
                {"Categorie": "Structuur", "Parameter": "Plateau Modulus (Gₙ⁰)",
                 "Waarde": f"{gn0:.2e}" if not np.isnan(gn0) else "N/A", "Eenheid": "Pa",
                 "Info": gn0_info},
                {"Categorie": "Structuur", "Parameter": "Crossover Punten",
                 "Waarde": f"{num_cos}", "Eenheid": "-", "Info": "Aantal G'=G'' kruisingen"},
                {"Categorie": "Validatie", "Parameter": "Arrhenius R²",
                 "Waarde": f"{r2_final:.4f}", "Eenheid": "-", "Info": "Lineaire fit kwaliteit"},
                {"Categorie": "Validatie", "Parameter": "Adjusted R²",
                 "Waarde": f"{r2_adj:.4f}", "Eenheid": "-", "Info": "R² gecorrigeerd voor # datapunten"},
            ]
            st.table(pd.DataFrame(dashboard_data))

            st.divider()
            st.subheader(texts.get("tab7_validation", "🔍 Model Reliability & Automatic Validation"))

            check_col1, check_col2 = st.columns(2)

            with check_col1:
                st.write(texts.get("tab7_thermal", "**Thermal Models:**"))
                if wlf_c1 < 0 or wlf_c2 < 0:
                    st.error(f"❌ **WLF Ongeldig:** Negatieve constanten (C₁={wlf_c1:.1f})")
                elif wlf_c1 < 5 or wlf_c1 > 30:
                    st.warning(f"⚠️ **WLF Atypisch:** C₁={wlf_c1:.1f}")
                else:
                    st.success(f"✅ **WLF Stabiel:** C₁={wlf_c1:.1f}, C₂={wlf_c2:.0f}K")

                if r2_adj > 0.98:
                    st.success(f"✅ **Arrhenius uitstekend:** Adj. R²={r2_adj:.4f}")
                elif r2_adj > 0.90:
                    st.info(f"ℹ️ **Arrhenius acceptabel:** Adj. R²={r2_adj:.4f}")
                else:
                    st.warning(f"⚠️ **Arrhenius zwak:** Adj. R²={r2_adj:.4f}")

                if vft_success:
                    estimated_tg = t_inf_c + 50
                    st.info(f"🌡️ **Geschatte Tg:** {estimated_tg:.1f}°C")

            with check_col2:
                st.write(texts.get("tab7_structural", "**Structural Quality:**"))
                if not np.isnan(slope_term):
                    if slope_term < 1.5:
                        st.error(f"❌ **Vloeiprobleem:** Slope={slope_term:.2f} << 2.0")
                    elif slope_term < 1.8:
                        st.warning(f"⚠️ **Afwijkende vloei:** Slope={slope_term:.2f}")
                    else:
                        st.success(f"✅ **Newtoniaans gedrag:** Slope={slope_term:.2f}")
                else:
                    st.info("ℹ️ Terminal zone niet bereikt")

                if num_cos == 0:
                    st.warning("⚠️ **Geen crossover:** G' > G'' over hele bereik")
                elif num_cos == 1:
                    st.success("✅ **Enkelvoudig crossover**")
                else:
                    st.error(f"❌ **{num_cos} crossovers:** Thermorheologisch complex!")

                if not np.isnan(eta0):
                    st.info(f"💧 **Hydrolyse Check:** η₀={eta0:.1e} Pa·s")

            st.divider()
            st.subheader(texts.get("tab7_crossovers", "⚖️ Crossover Points per Temperature"))
            if not co_df.empty:
                st.dataframe(co_df, use_container_width=True)
            else:
                st.info("Geen crossover punten gevonden.")

            st.divider()
            st.subheader(texts.get("tab7_export", "💾 Data Export - Download Your Results"))

            col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)

            col_ex1.download_button(
                texts.get("export_params", "📊 Parameters CSV"),
                summ_df.to_csv(index=False).encode('utf-8'),
                f"{sample_name}_Parameters.csv", "text/csv"
            )

            shift_export_df = pd.DataFrame({
                'T_C': selected_temps,
                'log_aT': [st.session_state.shifts[t] for t in selected_temps],
                'aT': [10**st.session_state.shifts[t] for t in selected_temps]
            })
            col_ex2.download_button(
                texts.get("export_shifts", "🕒 Shift Factors CSV"),
                shift_export_df.to_csv(index=False).encode('utf-8'),
                f"{sample_name}_ShiftFactors.csv", "text/csv"
            )

            if not co_df.empty:
                col_ex3.download_button(
                    texts.get("export_crossovers", "⚖️ Crossovers CSV"),
                    co_df.to_csv(index=False).encode('utf-8'),
                    f"{sample_name}_Crossovers.csv", "text/csv"
                )

            gewenste_kolommen = {
                'w_s': 'omega_shifted_rad_s', 'Gp': 'Gp_Pa', 'Gpp': 'Gpp_Pa',
                'eta_s': 'Complex_Visc_Pas', 'delta': 'PhaseAngle_deg', 'T_group': 'Original_T_C'
            }
            beschikbare_kolommen = [k for k in gewenste_kolommen if k in m_df.columns]
            master_export_df = m_df[beschikbare_kolommen].copy().rename(columns=gewenste_kolommen)
            if 'Gp_Pa' in master_export_df.columns and 'Gpp_Pa' in master_export_df.columns:
                master_export_df['tan_delta'] = master_export_df['Gpp_Pa'] / master_export_df['Gp_Pa']
                master_export_df['G_star_Pa'] = np.sqrt(master_export_df['Gp_Pa']**2 + master_export_df['Gpp_Pa']**2)

            col_ex4.download_button(
                texts.get("export_mastercurve", "📈 Master Curve CSV"),
                master_export_df.to_csv(index=False).encode('utf-8'),
                f"{sample_name}_MasterCurve.csv", "text/csv"
            )

    else:
        st.error(texts.get("no_data_error", "❌ No data found in file. Check file format."))

else:
    st.info(texts.get("upload_prompt", "👆 Upload a frequency sweep CSV/TXT file to begin."))
    with st.expander(texts.get("instructions_title", "ℹ️ User Instructions")):
        st.markdown(texts.get("instructions", ""))

# --- LANGUAGE SWITCHER IN SIDEBAR ---
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🇳🇱 NL", use_container_width=True,
                 type="primary" if st.session_state.lang == 'NL' else "secondary"):
        if st.session_state.lang != 'NL':
            st.session_state.lang = 'NL'
            st.rerun()
with col2:
    if st.button("🇬🇧 EN", use_container_width=True,
                 type="primary" if st.session_state.lang == 'EN' else "secondary"):
        if st.session_state.lang != 'EN':
            st.session_state.lang = 'EN'
            st.rerun()
st.sidebar.markdown("---")
st.set_page_config(page_title=texts.get("title", "RheoApp"), layout="wide")
st.sidebar.caption("The R.A.T. App - Rheology Analysis Tool - v1.0")