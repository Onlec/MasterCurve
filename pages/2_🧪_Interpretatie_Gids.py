import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from io import BytesIO
import json

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
        return {"NL": {"interpretation_guide": {}}, "EN": {"interpretation_guide": {}}}

all_translations = load_translations()
texts = all_translations.get(st.session_state.lang, all_translations["NL"]).get("interpretation_guide", {})


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title=texts.get("page_title", "Interpretatie Gids - RAT"),
    page_icon=texts.get("page_icon", "🧪"),
    layout="wide"
)

# --- LANGUAGE SWITCHER IN SIDEBAR ---
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

# --- HEADER ---
st.title(texts.get("main_title", "ðŸ§ª Interpretatie & Validatie Gids"))
st.markdown(texts.get("main_intro", "Hoe lees je de grafieken in het dashboard?"))

# --- TABS ---
tab_vgp, tab_han, tab_cole, tab_cross, tab_scenarios = st.tabs([
    texts.get("tab_vgp_name", "ðŸ“Š Van Gurp-Palmen"),
    texts.get("tab_han_name", "ðŸ”¬ Han Plot"),
    texts.get("tab_cole_name", "ðŸŒ€ Cole-Cole Plot"),
    texts.get("tab_cross_name", "âš–ï¸ Crossover Analyse"),
    texts.get("tab_scenarios_name", "ðŸ’¼ Praktijk Scenario's")
])

# ============================================================================
# TAB 1: VAN GURP-PALMEN
# ============================================================================
with tab_vgp:
    vgp = texts.get("vgp", {})
    
    st.header(vgp.get("header", "ðŸ“Š Van Gurp-Palmen (vGP) Plot"))
    st.markdown(vgp.get("intro", "De vGP plot is de gouden standaard..."))
    
    st.info(vgp.get("why_powerful", "**Waarom is deze plot zo krachtig?**\n..."))
    
    st.divider()
    
    # Visual guide - 3 columns
    st.subheader(vgp.get("visual_guide_title", "ðŸ” Wat zie je? Interpretatie Gids"))
    
    v_col1, v_col2, v_col3 = st.columns(3)
    
    with v_col1:
        st.markdown(vgp.get("perfect_title", "### âœ… PERFECT: Superpositie"))
        st.success(vgp.get("perfect_description", "**Wat je ziet:**\n..."))
        # Placeholder image would go here
        st.caption(vgp.get("perfect_image_caption", "Ideaal: Alle temperaturen op Ã©Ã©n lijn"))
    
    with v_col2:
        st.markdown(vgp.get("moderate_title", "### âš ï¸ MATIG: Kleine Spreiding"))
        st.warning(vgp.get("moderate_description", "**Wat je ziet:**\n..."))
        st.caption(vgp.get("moderate_image_caption", "Acceptabel: Kleine spreiding"))
    
    with v_col3:
        st.markdown(vgp.get("problematic_title", "### âŒ PROBLEMATISCH"))
        st.error(vgp.get("problematic_description", "**Wat je ziet:**\n..."))
        st.caption(vgp.get("problematic_image_caption", "Probleem: Sterke spreiding"))
    
    st.divider()
    
    # Decision tree
    st.subheader(vgp.get("decision_tree_title", "ðŸŒ³ Diagnostische Beslisboom"))
    st.code(vgp.get("decision_tree", "```\nSTART...\n```"), language='text')
    
    st.divider()
    
    # TPU features
    st.subheader(vgp.get("tpu_features_title", "ðŸ”¬ TPU-Specifieke Kenmerken"))
    st.markdown(vgp.get("tpu_intro", "TPU heeft unieke kenmerken..."))
    
    st.markdown(vgp.get("tpu_step_pattern_title", "**1. 'Trap' Patroon**"))
    st.markdown(vgp.get("tpu_step_pattern", "**Wat je ziet:**\n..."))
    
    st.markdown(vgp.get("tpu_hook_pattern_title", "**2. 'Haak' bij Hoge Î´**"))
    st.markdown(vgp.get("tpu_hook_pattern", "**Wat je ziet:**\n..."))
    
    st.markdown(vgp.get("tpu_negative_wlf_title", "**3. Negatieve WLF Câ‚ & vGP**"))
    st.markdown(vgp.get("tpu_negative_wlf", "**De Link:**\n..."))
    
    st.divider()
    
    # Quick check
    st.subheader(vgp.get("quick_check_title", "âš¡ Quick Check: Is Mijn vGP OK?"))
    st.markdown(vgp.get("quick_check_intro", "Beantwoord deze vragen..."))
    
    st.markdown(vgp.get("quick_q1", "**Vraag 1:**..."))
    st.markdown("  " + vgp.get("quick_q1_yes", "âœ… JA â†’..."))
    st.markdown("  " + vgp.get("quick_q1_no", "âŒ NEE â†’..."))
    
    st.markdown(vgp.get("quick_q2", "**Vraag 2:**..."))
    st.markdown("  " + vgp.get("quick_q2_yes", "âš ï¸ JA â†’..."))
    st.markdown("  " + vgp.get("quick_q2_no", "ðŸš¨ NEE â†’..."))
    
    st.markdown(vgp.get("quick_result_title", "**ðŸ“Š RESULTAAT:**"))
    st.markdown(vgp.get("quick_result_all_yes", "**Alle âœ…?**\n..."))
    
    st.markdown(vgp.get("quick_action_title", "**ðŸ”§ DIRECTE ACTIES:**"))
    st.markdown(vgp.get("quick_actions", "| Observatie | Actie | ...\n|---|---|---|"))

# ============================================================================
# TAB 2: HAN PLOT
# ============================================================================
with tab_han:
    han = texts.get("han", {})
    
    st.header(han.get("header", "ðŸ”¬ Han Plot (G' vs G'')"))
    st.markdown(han.get("intro", "De Han plot detecteert chemische veranderingen..."))
    
    st.info(han.get("why_useful", "**Waarom is dit nuttig?**\n..."))
    
    st.divider()
    
    # Shifts interpretation
    st.subheader(han.get("shifts_title", "ðŸ” Wat betekenen verticale verschuivingen?"))
    
    st.markdown(han.get("upward_shift_title", "**â¬†ï¸ Opwaartse Verschuiving**"))
    st.markdown(han.get("upward_shift", "**Wat gebeurt er:**\n..."))
    
    st.markdown(han.get("downward_shift_title", "**â¬‡ï¸ Neerwaartse Verschuiving**"))
    st.markdown(han.get("downward_shift", "**Wat gebeurt er:**\n..."))
    
    st.divider()
    
    # Ideal overlap
    st.subheader(han.get("ideal_title", "âœ… Ideaal: Perfecte Overlap"))
    st.markdown(han.get("ideal_description", "**Wat je wilt zien:**\n..."))
    st.info(han.get("ideal_tpu_note", "**TPU Specifiek:**\n..."))
    
    st.divider()
    
    # Practical check
    st.subheader(han.get("practical_check_title", "âš¡ Praktische Check Procedure"))
    st.code(han.get("practical_check", "```\nSTAP 1...\n```"), language='text')
    
    st.markdown(han.get("han_vs_vgp_title", "ðŸ”„ Han vs Van Gurp-Palmen"))
    st.markdown(han.get("han_vs_vgp", "| Scenario | vGP | Han | Diagnose |\n|---|---|---|---|"))

# ============================================================================
# TAB 3: COLE-COLE
# ============================================================================
with tab_cole:
    cole = texts.get("cole", {})
    
    st.header(cole.get("header", "ðŸŒ€ Cole-Cole Plot (Î·'' vs Î·')"))
    st.markdown(cole.get("intro", "De Cole-Cole plot karakteriseert MWD..."))
    
    st.info(cole.get("why_useful", "**Wat vertelt de vorm?**\n..."))
    
    st.divider()
    
    # Shape guide
    st.subheader(cole.get("shape_guide_title", "ðŸ” Vorm-Interpretatie Gids"))
    
    st.markdown(cole.get("shape_circle_title", "**ðŸ”µ Perfecte Halve Cirkel**"))
    st.markdown(cole.get("shape_circle", "**Vorm:**\n```\n...\n```"))
    
    st.markdown(cole.get("shape_flattened_title", "**â¬­ Afgeplatte Boog**"))
    st.markdown(cole.get("shape_flattened", "**Vorm:**\n..."))
    
    st.markdown(cole.get("shape_asymmetric_title", "**ã€°ï¸ Asymmetrische Vorm**"))
    st.markdown(cole.get("shape_asymmetric", "**Vorm:**\n..."))
    
    st.divider()
    
    # Temperature effects
    st.subheader(cole.get("temp_changes_title", "ðŸŒ¡ï¸ Wat als Vorm Verandert met T?"))
    st.markdown(cole.get("temp_changes", "**Ideaal: Vorm blijft constant**\n..."))
    st.markdown(cole.get("temp_size_changes", "**Normale Schaal Verandering:**\n..."))
    
    st.divider()
    
    # Quantitative
    st.subheader(cole.get("quantitative_title", "ðŸ“ Kwantitatieve Analyse"))
    st.markdown(cole.get("quantitative_intro", "Voor experts: Kwantificeer MWD breedte."))
    
    st.markdown(cole.get("circle_diameter", "**Methode 1: Cirkel Diameter Ratio**\n..."))
    st.markdown(cole.get("shoulder_analysis", "**Methode 2: Shoulder Detectie**\n..."))
    st.markdown(cole.get("mwd_process_link", "**MWD â†” Procesgedrag:**\n| MWD | Cole-Cole | ..."))

# ============================================================================
# TAB 4: CROSSOVER ANALYSIS
# ============================================================================
with tab_cross:
    crossover = texts.get("crossover", {})
    
    st.header(crossover.get("header", "âš–ï¸ Crossover Analyse (G' = G'')"))
    st.markdown(crossover.get("intro", "Het crossover punt markeert..."))
    
    st.divider()
    
    # Number of crossovers
    st.subheader(crossover.get("number_title", "ðŸ”¢ Hoeveel Crossovers Heb Je?"))
    
    cross_col1, cross_col2, cross_col3 = st.columns(3)
    
    with cross_col1:
        st.markdown(crossover.get("one_crossover_title", "### 1ï¸âƒ£ EÃ©n Crossover"))
        st.success(crossover.get("one_crossover", "**Patroon:**\n..."))
        st.latex(crossover.get("one_crossover_tau_formula", r"\tau = \frac{1}{\omega_{co}}"))
        st.markdown(crossover.get("one_crossover_tau_meaning", "Dit is de tijd..."))
    
    with cross_col2:
        st.markdown(crossover.get("zero_crossover_title", "### 0ï¸âƒ£ Geen Crossover"))
        st.warning(crossover.get("zero_crossover", "**Patroon A:**\n..."))
    
    with cross_col3:
        st.markdown(crossover.get("multiple_crossover_title", "### 2ï¸âƒ£+ Meerdere"))
        st.error(crossover.get("multiple_crossover", "**Patroon:**\n..."))
    
    st.divider()
    
    # Frequency interpretation
    st.subheader(crossover.get("frequency_title", "ðŸ“Š Crossover Frequentie Interpretatie"))
    st.markdown(crossover.get("frequency_intro", "De positie vertelt je..."))
    
    # Create table from JSON data
    freq_data = crossover.get("frequency_table", {})
    if freq_data:
        df_freq = pd.DataFrame(
            freq_data.get("rows", []),
            columns=freq_data.get("columns", [])
        )
        st.table(df_freq)
    
    st.divider()
    
    # Quick check
    st.subheader(crossover.get("quick_check_title", "âš¡ Quick Crossover Check"))
    
    with st.expander(crossover.get("quick_check_expander", "ðŸ”§ Analyseer je Data")):
        n_cross = st.number_input(crossover.get("quick_check_input", "Hoeveel crossovers?"), 
                                   min_value=0, max_value=5, value=1, step=1)
        
        if n_cross == 0:
            cross_type = st.radio(
                crossover.get("zero_cross_question", "Wat zie je?"),
                [crossover.get("zero_cross_elastic", "G' > G'' overal"),
                 crossover.get("zero_cross_viscous", "G'' > G' overal")]
            )
            
            if cross_type == crossover.get("zero_cross_elastic", "G' > G'' overal"):
                st.error(crossover.get("zero_elastic_diagnosis", "âŒ Incomplete Smelt..."))
            else:
                st.error(crossover.get("zero_viscous_diagnosis", "âŒ Ernstige Degradatie..."))
        
        elif n_cross == 1:
            omega_co = st.number_input(crossover.get("one_cross_input", "Crossover frequentie (rad/s)"), 
                                        value=1.0, format="%.2f")
            tau = 1/omega_co if omega_co > 0 else 0
            
            st.success(crossover.get("one_cross_success", "âœ… Normaal\n\nÏ„ = {tau:.2f} s").format(tau=tau))
            
            if omega_co > 10:
                st.info(crossover.get("one_cross_fast", "Snelle relaxatie..."))
            elif omega_co < 0.1:
                st.warning(crossover.get("one_cross_slow", "Trage relaxatie..."))
            else:
                st.success(crossover.get("one_cross_typical", "Typisch TPU bereik"))
        
        else:
            st.error(crossover.get("multiple_cross_diagnosis", "âŒ {n_cross} Crossovers = Complex!").format(n_cross=n_cross))

# ============================================================================
# TAB 5: PRACTICAL SCENARIOS
# ============================================================================
with tab_scenarios:
    scenarios = texts.get("scenarios", {})
    
    st.header(scenarios.get("header", "ðŸ’¼ Praktijk Scenario's"))
    st.markdown(scenarios.get("intro", "Herken je deze situaties?"))
    
    # Scenario selector
    scenario = st.selectbox(
        scenarios.get("selector_label", "Kies je scenario:"),
        scenarios.get("selector_options", [
            "ðŸ”´ Mijn WLF Câ‚ is negatief!",
            "ðŸŸ  Master Curve 'trapt'",
            # ... etc
        ])
    )
    
    st.divider()
    
    # Scenario 1: Negative WLF C1
    if "negatief" in scenario.lower():
        st.error(scenarios.get("scenario1_title", "### ðŸš¨ Negatieve WLF Câ‚"))
        
        diag_col1, diag_col2 = st.columns([2, 1])
        
        with diag_col1:
            st.markdown(scenarios.get("scenario1_symptom", "**Symptoom:**\n```\nWLF Câ‚ < 0\n```"))
            st.markdown(scenarios.get("scenario1_root_cause", "**Root Cause:**\n..."))
            st.code(scenarios.get("scenario1_decision_tree", "```\nSTART...\n```"), language='text')
        
        with diag_col2:
            st.markdown(scenarios.get("scenario1_solutions", "**ðŸ”§ Oplossingen:**\n..."))
            st.info(scenarios.get("scenario1_prevention", "**ðŸ›¡ï¸ Preventie:**\n..."))
    
    # Scenario 2: Master curve stepping
    elif "trapt" in scenario.lower() or "trap" in scenario.lower():
        st.warning(scenarios.get("scenario2_title", "### âš ï¸ Master Curve Sluit Niet Aan"))
        st.markdown(scenarios.get("scenario2_symptom", "**Symptoom:** Trappen..."))
        st.markdown(scenarios.get("scenario2_three_types", "**Drie Patronen:**\n..."))
        st.markdown(scenarios.get("scenario2_checklist", "**âœ… Checklist:**\n..."))
    
    # Scenario 3: Low terminal slope
    elif "slope" in scenario.lower():
        st.warning(scenarios.get("scenario3_title", "### âš ï¸ Lage Terminal Slope"))
        st.markdown(scenarios.get("scenario3_symptom", "**Symptoom:**..."))
        st.markdown(scenarios.get("scenario3_diagnostics", "**ðŸ”¬ Diagnostiek:**\n..."))
    
    # Scenario 4-7: Other scenarios (similar pattern)
    # ...
    
    st.divider()
    
    # General checklist
    st.subheader(scenarios.get("general_checklist_title", "âœ… Algemene Checklist"))
    st.markdown(scenarios.get("general_checklist", "**ðŸ“‹ Before Every Measurement:**\n..."))

# --- FOOTER ---
# --- LANGUAGE SWITCHER IN SIDEBAR --

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
st.sidebar.caption("The R.A.T. App - Rheology Analysis Tool - v1.0")