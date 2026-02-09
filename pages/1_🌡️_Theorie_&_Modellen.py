import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from io import BytesIO
import json
from pathlib import Path

# --- LANGUAGE SETUP ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'nl'

@st.cache_data
def load_translations(language):
    """Load unified translations from languages folder"""
    # Ga van pages/ naar root, dan naar languages/
    lang_file = Path(__file__).parent.parent / 'languages' / f'{language}.json'
    
    with open(lang_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load alle vertalingen
all_texts = load_translations(st.session_state.lang)

# Kies de juiste sectie voor deze page
# Page 1: 'theory_models'
# Page 2: 'interpretation_guide'  
# Page 3: 'data_troubleshooting'
texts = all_texts['theory_models']  # ← Verander per page!

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title=texts.get("page_title", "Theorie & Modellen - RheoApp"),
    page_icon=texts.get("page_icon", "🧬"),
    layout="wide"
)

# --- LANGUAGE SWITCHER IN SIDEBAR ---
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🇳🇱 NL", use_container_width=True,
                 type="primary" if st.session_state.lang == 'nl' else "secondary"):
        if st.session_state.lang != 'nl':
            st.session_state.lang = 'nl'
            st.rerun()

with col2:
    if st.button("🇬🇧 EN", use_container_width=True,
                 type="primary" if st.session_state.lang == 'en' else "secondary"):
        if st.session_state.lang != 'en':
            st.session_state.lang = 'en'
            st.rerun()

# --- HEADER ---
st.title(texts.get("main_title", "🧬 Theoretische Achtergrond & Modellen"))
st.markdown(texts.get("main_intro", "Deze pagina bevat de wetenschappelijke basis van de RheoApp."))

# --- TABS ---
tab_tts, tab_therm, tab_struc, tab_calc = st.tabs([
    texts.get("tab_tts_name", "🕒 Time-Temperature Superposition"),
    texts.get("tab_thermal_name", "🔥 Thermische Modellen"),
    texts.get("tab_structural_name", "🏗️ Structurele Parameters"),
    texts.get("tab_calc_name", "🧮 Snelle Calculators")
])

# ============================================================================
# TAB 1: TIME-TEMPERATURE SUPERPOSITION
# ============================================================================
with tab_tts:
    tts = texts.get("tts", {})
    
    st.header(tts.get("header", "Time-Temperature Superposition (TTS)"))
    st.markdown(tts.get("intro", "Het fundamentele principe achter TTS..."))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(tts.get("physical_basis_title", "🎯 Fysische basis"))
        st.info(tts.get("physical_basis_text", "Bij temperatuurverandering..."))
        
        st.markdown(tts.get("shifted_freq_label", "**Verschoven Frequentie:**"))
        st.latex(tts.get("shifted_freq_formula", r"\omega_{shifted} = \omega \cdot a_T"))
        
        st.markdown(tts.get("whereby_label", "**Waarbij:**"))
        st.markdown(tts.get("shift_factor_explanation", "- $a_T$ = Shift factor..."))
    
    with col2:
        st.subheader(tts.get("validity_title", "⚠️ Voorwaarde voor geldigheid"))
        st.warning(tts.get("validity_warning", "Het materiaal moet thermorheologisch simpel zijn..."))
        st.info(tts.get("protip", "**Pro Tip:** De Van Gurp-Palmen plot..."))
    
    st.divider()
    
    # Interactive Demo
    st.subheader(tts.get("demo_title", "📊 Interactieve TTS Demonstratie"))
    
    demo_col1, demo_col2 = st.columns([2, 1])
    
    with demo_col2:
        st.markdown(tts.get("demo_params_label", "**Speel met de parameters:**"))
        
        ref_temp = st.slider(
            tts.get("demo_ref_temp", "Referentie Temperatuur (°C)"),
            min_value=150, max_value=250, value=200, step=10,
            help=tts.get("demo_tref_info", "Dit is je 'anker' temperatuur")
        )
        
        test_temp = st.slider(
            tts.get("demo_test_temp", "Test Temperatuur (°C)"),
            min_value=150, max_value=250, value=170, step=10
        )
        
        ea = st.slider(
            tts.get("demo_ea", "Activatie Energie Ea (kJ/mol)"),
            min_value=40, max_value=150, value=100, step=5
        )
        
        # Calculate shift factor (Arrhenius)
        R = 8.314  # J/mol·K
        T_ref_K = ref_temp + 273.15
        T_test_K = test_temp + 273.15
        
        log_aT = (ea * 1000 / R) * (1/T_test_K - 1/T_ref_K) / 2.303
        aT = 10**log_aT
        
        st.markdown(tts.get("demo_shift_result", "**📌 Resultaat:**").format(test_temp=test_temp))
        st.code(tts.get("demo_shift_formula", "log(aT) = {log_at:.3f}  →  aT = {at:.4f}").format(
            log_at=log_aT, at=aT
        ))
        
        omega_equiv = 1.0 * aT
        st.info(tts.get("demo_interpretation", "**💡 Interpretatie:**\n...").format(
            test_temp=test_temp, ref_temp=ref_temp, omega_equiv=omega_equiv
        ))
    
    with demo_col1:
        # Simple demo plot
        fig, ax = plt.subplots(figsize=(8, 5))
        omega = np.logspace(-2, 2, 50)
        
        # Fake data
        G_ref = 1e5 * omega**0.5
        G_test = 1e5 * (omega / aT)**0.5
        G_test_shifted = 1e5 * (omega * aT)**0.5
        
        ax.loglog(omega, G_ref, 'k-', linewidth=2, 
                  label=tts.get("demo_legend_ref", "{ref_temp}°C (referentie)").format(ref_temp=ref_temp))
        ax.loglog(omega, G_test, 'r--', alpha=0.5, 
                  label=tts.get("demo_legend_test", "{test_temp}°C (ruw)").format(test_temp=test_temp))
        ax.loglog(omega * aT, G_test, 'b-', linewidth=2, 
                  label=tts.get("demo_legend_shifted", "{test_temp}°C (shifted)").format(test_temp=test_temp, at=aT))
        
        ax.set_xlabel(tts.get("demo_x_label", "ω (rad/s)"), fontsize=12, fontweight='bold')
        ax.set_ylabel(tts.get("demo_y_label", "G' (willekeurige eenheden)"), fontsize=12, fontweight='bold')
        ax.set_title(tts.get("demo_plot_title", "Effect van Shift Factor op Frequentie"), 
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
    
    st.divider()
    
    # Decision tree
    st.subheader(tts.get("decision_tree_title", "🌳 TTS Validatie Decision Tree"))
    st.code(tts.get("decision_tree", "```\nStart hier\n...\n```"), language='text')
    
    st.divider()

# Model comparison - INTERACTIVE
st.subheader(tts.get("model_comparison_title", "📈 Model Vergelijking"))
st.markdown(tts.get("model_comparison_intro", "Verschillende temperatuurmodellen..."))

# Interactive comparison plot
comp_col1, comp_col2 = st.columns([2, 1])

with comp_col2:
    st.markdown(tts.get("model_comparison_params_label", "**Vergelijk de modellen:**"))
    
    # Model parameters - interactive sliders
    T_ref_comp = st.number_input(
        tts.get("model_comparison_tref_label", "T_ref (°C)"),
        value=200, step=10, key="comp_tref"
    )
    
    Ea_comp = st.number_input(
        tts.get("model_comparison_ea_label", "Ea (kJ/mol) [Arrhenius]"),
        value=100, min_value=50, max_value=150, step=10, key="comp_ea"
    )
    
    C1_comp = st.number_input(
        tts.get("model_comparison_c1_label", "WLF C₁"),
        value=10.0, min_value=5.0, max_value=20.0, step=0.5, key="comp_c1"
    )
    
    C2_comp = st.number_input(
        tts.get("model_comparison_c2_label", "WLF C₂ (K)"),
        value=50, min_value=30, max_value=80, step=5, key="comp_c2"
    )
    
    st.caption(tts.get("model_comparison_adjust_tip", "💡 Pas parameters aan om het effect te zien!"))

with comp_col1:
    # Temperature range for comparison
    T_range = np.linspace(150, 230, 50)  # 150-230°C
    T_ref_K = T_ref_comp + 273.15
    R = 8.314  # J/mol·K
    
    # Calculate shift factors for each model
    log_aT_arrhenius = []
    log_aT_wlf = []
    
    for T in T_range:
        T_K = T + 273.15
        
        # Arrhenius model
        log_aT_arr = (Ea_comp * 1000 / R) * (1/T_K - 1/T_ref_K) / 2.303
        log_aT_arrhenius.append(log_aT_arr)
        
        # WLF model
        log_aT_w = (-C1_comp * (T - T_ref_comp)) / (C2_comp + (T - T_ref_comp))
        log_aT_wlf.append(log_aT_w)
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot both models
    ax.plot(T_range, log_aT_arrhenius, 'r-', linewidth=2.5, 
            label=tts.get("model_arrhenius", "Arrhenius"))
    ax.plot(T_range, log_aT_wlf, 'b-', linewidth=2.5, 
            label=tts.get("model_wlf", "WLF"))
    
    # Mark reference temperature
    ax.axvline(T_ref_comp, color='gray', linestyle='--', alpha=0.5, 
               label=tts.get("model_comparison_ref_line", "T_ref = {tref}°C").format(tref=T_ref_comp))
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    
    # Labels and styling
    ax.set_xlabel(tts.get("model_comparison_x_label", "Temperatuur (°C)"), 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(tts.get("model_comparison_y_label", "log(aT)"), 
                  fontsize=12, fontweight='bold')
    ax.set_title(tts.get("model_comparison_plot_title", "Model Vergelijking: Arrhenius vs WLF"), 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    annotation_text = tts.get("model_comparison_annotation", 
                               "Bij T_ref = {tref}°C: log(aT) = 0").format(tref=T_ref_comp)
    ax.text(0.02, 0.98, annotation_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    st.pyplot(fig)
    plt.close()
    
    # Explanation below plot
    st.caption(tts.get("model_comparison_interpretation", """**Interpretatie:**
- **Arrhenius** (rood): Lineair verband, simpel model
- **WLF** (blauw): Gebogen verloop, beter voor polymeren nabij Tg
- Bij T_ref kruisen beide modellen (log(aT) = 0)
- Verschil tussen modellen groter bij lage/hoge T"""))

# ============================================================================
# TAB 2: THERMAL MODELS
# ============================================================================
with tab_therm:
    thermal = texts.get("thermal", {})
    
    st.header(thermal.get("header", "Thermische Shift Factor Modellen"))
    st.markdown(thermal.get("intro", "Er zijn drie hoofdmodellen..."))
    
    st.divider()
    
    # ARRHENIUS
    st.subheader(thermal.get("arrhenius_title", "🔥 Arrhenius Model"))
    st.caption(thermal.get("arrhenius_subtitle", "Voor Simpele Vloeistoffen"))
    
    st.markdown(thermal.get("arrhenius_formula_label", "**Formule:**"))
    st.latex(thermal.get("arrhenius_formula", r"\log(a_T) = \frac{E_a}{R} ..."))
    
    st.markdown(thermal.get("arrhenius_linear_label", "**Lineaire vorm:**"))
    st.latex(thermal.get("arrhenius_linear", r"\log(a_T) = \text{slope} ..."))
    st.latex(thermal.get("arrhenius_ea_calc", r"E_a = ..."))
    
    st.markdown(thermal.get("arrhenius_params_label", "**Parameters:**"))
    st.markdown(thermal.get("arrhenius_params", "- **Ea**: Activatie-energie..."))
    
    st.markdown(thermal.get("arrhenius_when_label", "**Wanneer te gebruiken:**"))
    st.markdown(thermal.get("arrhenius_when", "✅ **Geschikt voor:**\n..."))
    
    st.markdown(thermal.get("arrhenius_typical_label", "**Typische waarden voor TPU:**"))
    st.markdown(thermal.get("arrhenius_typical", "- **Ea**: 80-120 kJ/mol..."))
    
    st.divider()
    
    # WLF
    st.subheader(thermal.get("wlf_title", "🧊 Williams-Landel-Ferry (WLF) Model"))
    st.caption(thermal.get("wlf_subtitle", "Voor Amorf Polymeren nabij Tg"))
    
    st.markdown(thermal.get("wlf_formula_label", "**Formule:**"))
    st.latex(thermal.get("wlf_formula", r"\log(a_T) = \frac{-C_1 ..."))
    
    st.markdown(thermal.get("wlf_params_label", "**Parameters:**"))
    st.markdown(thermal.get("wlf_params", "- **C₁**: Vrije volume parameter..."))
    
    st.markdown(thermal.get("wlf_universal_label", "**'Universele' constanten:**"))
    st.markdown(thermal.get("wlf_universal", "Als Tref = Tg:\n..."))
    
    st.markdown(thermal.get("wlf_typical_label", "**Typische bereiken:**"))
    st.markdown(thermal.get("wlf_typical", "| Parameter | Normaal Bereik | ...\n|---|---|---|"))
    
    st.warning(thermal.get("wlf_negative_warning", "⚠️ **KRITIEKE WAARSCHUWING: Negatieve C₁**\n..."))
    
    st.info(thermal.get("wlf_tg_estimation", "**🌡️ Tg Schatting via WLF:**\n..."))
    
    st.divider()
    
    # VFT
    st.subheader(thermal.get("vft_title", "🌀 Vogel-Fulcher-Tammann (VFT) Model"))
    st.caption(thermal.get("vft_subtitle", "Hybride Model"))
    
    st.markdown(thermal.get("vft_formula_label", "**Formule:**"))
    st.latex(thermal.get("vft_formula", r"\log(a_T) = A + \frac{B}{T - T_0}"))
    
    st.markdown(thermal.get("vft_params_label", "**Parameters:**"))
    st.markdown(thermal.get("vft_params", "- **A**: Constante offset..."))
    
    st.markdown(thermal.get("vft_relation_label", "**Relatie met Tg:**"))
    st.markdown(thermal.get("vft_relation", "Vuistregel:\n$$T_g \\approx T_0 + 50K$$\n..."))
    
    st.markdown(thermal.get("vft_when_label", "**Wanneer te gebruiken:**"))
    st.markdown(thermal.get("vft_when", "✅ **Voordelen:**\n..."))
    
    st.divider()
    
    # Model comparison table
    st.subheader(thermal.get("model_comparison_title", "⚖️ Model Vergelijking Samenvatting"))
    st.markdown(thermal.get("model_comparison_table", "| Aspect | Arrhenius | WLF | VFT |\n|---|---|---|---|"))
    
    st.subheader(thermal.get("model_interpretation_title", "🧠 Interpretatie van Parameters"))
    st.markdown(thermal.get("model_interpretation", "| Waarneming | Betekenis | Actie |\n|---|---|---|"))

# ============================================================================
# TAB 3: STRUCTURAL PARAMETERS
# ============================================================================
with tab_struc:
    structural = texts.get("structural", {})
    
    st.header(structural.get("header", "Structurele Rheologische Parameters"))
    st.markdown(structural.get("intro", "Deze parameters karakteriseren..."))
    
    st.divider()
    
    # ETA0
    st.subheader(structural.get("eta0_title", "🌊 Zero-Shear Viscosity (η₀)"))
    st.caption(structural.get("eta0_subtitle", "De Ultieme Vloei-indicator"))
    
    st.markdown(structural.get("eta0_definition_label", "**Definitie:**"))
    st.markdown(structural.get("eta0_definition", "De viscositeit bij oneindige lage..."))
    
    st.markdown(structural.get("eta0_cross_label", "**Cross Model:**"))
    st.latex(structural.get("eta0_cross_formula", r"\eta^*(\omega) = ..."))
    st.markdown(structural.get("eta0_cross_params", "- **η₀**: Zero-shear viscosity..."))
    
    st.markdown(structural.get("eta0_mw_label", "**Relatie met Molecuulgewicht:**"))
    st.markdown(structural.get("eta0_mw_relation", "Voor lineaire polymeren:\n$$\\eta_0 \\propto M_w^{3.4}$$\n..."))
    
    st.markdown(structural.get("eta0_typical_label", "**Typische waarden:**"))
    st.markdown(structural.get("eta0_typical_table", "| TPU Type | η₀ Range | ...\n|---|---|---|"))
    
    st.markdown(structural.get("eta0_process_label", "**Procesimplicaties:**"))
    st.markdown(structural.get("eta0_process", "| Proces | Wat bepaalt η₀ | ...\n|---|---|---|"))
    
    st.markdown(structural.get("eta0_qc_label", "**🎯 Gebruik als QC:**"))
    st.markdown(structural.get("eta0_qc", "**Hydrolyse Detectie:**\n..."))
    
    st.divider()
    
    # GN0
    st.subheader(structural.get("gn0_title", "🏗️ Plateau Modulus (G_N⁰)"))
    st.caption(structural.get("gn0_subtitle", "Maat voor Entanglement Dichtheid"))
    
    st.markdown(structural.get("gn0_definition_label", "**Definitie:**"))
    st.markdown(structural.get("gn0_definition", "De plateau modulus is..."))
    
    st.markdown(structural.get("gn0_molecular_label", "**Relatie met Moleculaire Architectuur:**"))
    st.markdown(structural.get("gn0_molecular", "$$G_N^0 \\approx \\frac{\\rho R T}{M_e}$$\n..."))
    
    st.markdown(structural.get("gn0_typical_label", "**Typische waarden:**"))
    st.markdown(structural.get("gn0_typical", "| TPU Hardheid | G_N⁰ (Pa) | ...\n|---|---|---|"))
    
    st.markdown(structural.get("gn0_process_label", "**Procesimplicaties:**"))
    st.markdown(structural.get("gn0_process", "**Hoge G_N⁰:**\n..."))
    
    st.divider()
    
    # TERMINAL SLOPE
    st.subheader(structural.get("terminal_slope_title", "📐 Terminal Slope"))
    st.caption(structural.get("terminal_slope_subtitle", "Validatie van Complete Smelt"))
    
    st.markdown(structural.get("terminal_slope_definition_label", "**Definitie:**"))
    st.markdown(structural.get("terminal_slope_definition", "In de terminal zone..."))
    
    st.markdown(structural.get("terminal_slope_ideal_label", "**Ideale waarde:**"))
    st.markdown(structural.get("terminal_slope_ideal", "**Slope = 2.0**\n..."))
    
    st.markdown(structural.get("terminal_slope_interpretation_label", "**Interpretatie:**"))
    st.markdown(structural.get("terminal_slope_interpretation", "| Slope | Diagnose | ...\n|---|---|---|"))
    
    st.markdown(structural.get("terminal_slope_tpu_label", "**TPU Specifiek:**"))
    st.markdown(structural.get("terminal_slope_tpu", "**Waarom is dit KRITIEK:**\n..."))
    
    st.divider()
    
    # CROSSOVER
    st.subheader(structural.get("crossover_title", "⚖️ Crossover Frequentie (ω_co)"))
    st.caption(structural.get("crossover_subtitle", "Waar Elasticiteit = Viscositeit"))
    
    st.markdown(structural.get("crossover_definition_label", "**Definitie:**"))
    st.markdown(structural.get("crossover_definition", "Het crossover punt is waar:\n..."))
    
    st.markdown(structural.get("crossover_relaxation_label", "**Relatie met Relaxatietijd:**"))
    st.markdown(structural.get("crossover_relaxation", "$$\\tau = \\frac{1}{\\omega_{co}}$$\n..."))
    
    st.markdown(structural.get("crossover_number_label", "**Aantal Crossovers:**"))
    st.markdown(structural.get("crossover_number", "| Aantal | Interpretatie | ...\n|---|---|---|"))
    
    st.markdown(structural.get("crossover_tpu_label", "**Meerdere Crossovers in TPU:**"))
    st.markdown(structural.get("crossover_tpu", "**Oorzaken:**\n..."))
    
    st.markdown(structural.get("crossover_process_label", "**Procesimplicaties:**"))
    st.markdown(structural.get("crossover_process", "**Hoge ω_co:**\n..."))

# ============================================================================
# TAB 4: CALCULATORS
# ============================================================================
with tab_calc:
    calculators = texts.get("calculators", {})
    
    st.header(calculators.get("header", "🧮 Snelle Calculators"))
    st.markdown(calculators.get("intro", "Handige tools voor snelle berekeningen."))
    
    st.divider()
    
    # CALC 1: MW CHANGE
    st.subheader(calculators.get("mw_calc_title", "📊 Calculator 1: Molecuulgewicht Verandering"))
    st.caption(calculators.get("mw_calc_subtitle", "Schat Mw verschil tussen batches"))
    
    col_mw1, col_mw2 = st.columns(2)
    
    with col_mw1:
        eta0_ref = st.number_input(
            calculators.get("mw_calc_eta0_ref_label", "η₀ Referentie Batch (Pa·s)"),
            value=3.5e5, format="%.2e"
        )
        eta0_new = st.number_input(
            calculators.get("mw_calc_eta0_new_label", "η₀ Nieuwe Batch (Pa·s)"),
            value=2.8e5, format="%.2e"
        )
        mw_exponent = st.number_input(
            calculators.get("mw_calc_exponent_label", "Mw Exponent"),
            value=3.4, step=0.1,
            help=calculators.get("mw_calc_exponent_help", "Voor lineaire polymeren is dit 3.4")
        )
    
    with col_mw2:
        if eta0_ref > 0:
            ratio = eta0_new / eta0_ref
            mw_change = ((ratio ** (1/mw_exponent)) - 1) * 100
            
            st.markdown(calculators.get("mw_calc_result_title", "**📌 Resultaat:**"))
            st.code(calculators.get("mw_calc_ratio", "η₀ Ratio: {ratio:.3f}").format(ratio=ratio))
            st.code(calculators.get("mw_calc_change", "ΔMw: {change:+.1f}%").format(change=mw_change))
    
    st.markdown(calculators.get("mw_calc_interpretation", "**💡 Interpretatie:**\n| Δη₀ | ΔMw | ...\n|---|---|---|"))
    
    st.divider()
    
    # CALC 2: TG ESTIMATION
    st.subheader(calculators.get("tg_calc_title", "🌡️ Calculator 2: Tg Schatting"))
    st.caption(calculators.get("tg_calc_subtitle", "Via WLF of VFT parameters"))
    
    method = st.radio(
        calculators.get("tg_calc_method_label", "Berekeningsmethod"),
        [calculators.get("tg_calc_method_wlf", "WLF"), 
         calculators.get("tg_calc_method_vft", "VFT")]
    )
    
    if method == calculators.get("tg_calc_method_wlf", "WLF"):
        tref = st.number_input(calculators.get("tg_calc_tref_label", "T_ref (°C)"), value=200)
        c2 = st.number_input(calculators.get("tg_calc_c2_label", "WLF C₂ (K)"), value=50)
        tg = tref - c2
        st.success(calculators.get("tg_calc_result_wlf", "**Geschatte Tg:** {tg:.1f}°C").format(tg=tg))
    else:
        t0 = st.number_input(calculators.get("tg_calc_t0_label", "VFT T₀ (°C)"), value=-90)
        tg = t0 + 50
        st.success(calculators.get("tg_calc_result_vft", "**Geschatte Tg:** {tg:.1f}°C").format(tg=tg))
    
    st.warning(calculators.get("tg_calc_validation", "**⚠️ Validatie Essentieel:**\n..."))
    
    st.divider()
    
    # CALC 3: PROCESS TEMP
    st.subheader(calculators.get("process_calc_title", "🏭 Calculator 3: Proces Temperatuur"))
    st.caption(calculators.get("process_calc_subtitle", "Bepaal ideaal T-venster"))
    
    tg_soft = st.number_input(calculators.get("process_calc_tg_label", "Tg Soft Segment (°C)"), value=-40)
    tm_hard = st.number_input(
        calculators.get("process_calc_tm_label", "Tm Hard Segment (°C)"), 
        value=190,
        help=calculators.get("process_calc_tm_help", "Typisch: 170-220°C")
    )
    
    t_min = tm_hard + 20
    t_max = min(230, tm_hard + 50)
    t_extr = t_min + 15
    t_inj_min = t_min + 10
    t_inj_max = t_max
    t_coat_min = t_min
    t_coat_max = t_min + 20
    t_comp_min = t_min
    t_comp_max = t_min + 30
    
    st.markdown(calculators.get("process_calc_recommendations", "**📋 Aanbevolen Procestemperaturen:**").format(
        tm=tm_hard, t_min=t_min, t_max=t_max, 
        t_extr=t_extr, t_inj_min=t_inj_min, t_inj_max=t_inj_max,
        t_coat_min=t_coat_min, t_coat_max=t_coat_max,
        t_comp_min=t_comp_min, t_comp_max=t_comp_max
    ))
    
    st.divider()
    
    # QUICK REFERENCE
    st.subheader(calculators.get("quick_ref_title", "📚 Snelle Referentie Tabel"))
    st.caption(calculators.get("quick_ref_subtitle", "Typische Waarden voor TPU"))
    
    st.markdown(calculators.get("quick_ref_table", "| Parameter | Symbool | Typisch Bereik | ...\n|---|---|---|---|"))

# --- FOOTER ---
st.sidebar.divider()
st.sidebar.caption("RheoApp - Theorie & Modellen v1.0")