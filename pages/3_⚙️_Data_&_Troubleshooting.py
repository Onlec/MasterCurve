import streamlit as st
import pandas as pd
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
        return {"NL": {"data_troubleshooting": {}}, "EN": {"data_troubleshooting": {}}}

all_translations = load_translations()
texts = all_translations.get(st.session_state.lang, all_translations["NL"]).get("data_troubleshooting", {})

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title=texts.get("page_title", "Data & Troubleshooting - RheoApp"),
    page_icon=texts.get("page_icon", "⚙️"),
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
st.title(texts.get("main_title", "⚙️ Data Specificaties & Troubleshooting"))
st.markdown(texts.get("main_intro", "Zorg ervoor dat je data correct is geformatteerd..."))

# ============================================================================
# SECTION 1: DATA FORMAT SPECIFICATIONS
# ============================================================================
format_section = texts.get("format", {})

st.header(format_section.get("header", "1. Data Format Specificaties"))
st.markdown(format_section.get("intro", "De app accepteert .csv of .txt bestanden..."))

# Example table
st.subheader(format_section.get("example_title", "Voorbeeld van de juiste structuur:"))

example_data = format_section.get("example_table", {})
if example_data:
    df_example = pd.DataFrame({
        col: [row[i] for row in example_data.get("rows", [])]
        for i, col in enumerate(example_data.get("columns", []))
    })
    st.table(df_example)
else:
    # Fallback to hardcoded example
    df_example = pd.DataFrame({
        "T (°C)": [170, 170, 180],
        "omega (rad/s)": [0.1, 0.25, 0.1],
        "G' (Pa)": [1250, 2100, 850],
        "G'' (Pa)": [3400, 4800, 2100],
        "eta* (Pa·s)": [36220, 21000, 22600]
    })
    st.table(df_example)

# Column names guide
with st.expander(format_section.get("column_guide_title", "Gids voor kolomnamen")):
    st.markdown(format_section.get("column_guide_text", "De app zoekt naar specifieke trefwoorden..."))

# Additional format details (if available)
if format_section.get("file_requirements"):
    st.markdown(format_section.get("file_requirements"))

if format_section.get("typical_sources"):
    st.markdown(format_section.get("typical_sources"))

if format_section.get("minimum_requirements"):
    st.markdown(format_section.get("minimum_requirements"))

st.divider()

# ============================================================================
# SECTION 2: TROUBLESHOOTING
# ============================================================================
trouble = texts.get("troubleshooting", {})

st.header(trouble.get("header", "2. Troubleshooting (Probleemoplosser)"))

col_err, col_sol = st.columns(2)

with col_err:
    st.subheader(trouble.get("column_problem_title", "Wat gaat er mis?"))
    
    st.error(trouble.get("error_negative_c1", "**Fout: 'Negative C1 found' (WLF)**"))
    st.warning(trouble.get("warning_low_r2", "**Waarschuwing: 'R² < 0.90'**"))
    st.error(trouble.get("error_fit_failed", "**Fout: 'Fit failed' bij Zero-Shear**"))
    st.info(trouble.get("problem_stepping", "**Probleem: Master Curve 'trapt'**"))

with col_sol:
    st.subheader(trouble.get("column_solution_title", "Hoe los ik het op?"))
    
    st.markdown(trouble.get("solution_negative_c1", "- **C1 Check**: Je referentietemperatuur..."))
    st.markdown(trouble.get("solution_low_r2", "- **R² Verbeteren**: Verwijder uiterste temperaturen..."))
    st.markdown(trouble.get("solution_fit_failed", "- **Fit Succes**: Het Cross-model heeft meer data..."))
    st.markdown(trouble.get("solution_stepping", "- **Trapping**: Gebruik de handmatige shift..."))

# Detailed troubleshooting sections (expandable)
with st.expander("📖 Uitgebreide Troubleshooting: Negatieve WLF C₁"):
    st.markdown(trouble.get("error_negative_c1_detailed", "**🚨 Negative WLF C₁ - Uitgebreid**\n..."))

with st.expander("📖 Uitgebreide Troubleshooting: Lage R²"):
    st.markdown(trouble.get("warning_low_r2_detailed", "**⚠️ Lage Arrhenius R²**\n..."))

with st.expander("📖 Uitgebreide Troubleshooting: η₀ Fit Failure"):
    st.markdown(trouble.get("error_fit_failed_detailed", "**❌ η₀ Fit Failure**\n..."))

with st.expander("📖 Uitgebreide Troubleshooting: Master Curve Stepping"):
    st.markdown(trouble.get("problem_stepping_detailed", "**🔧 Master Curve Stepping**\n..."))

# Common issues quick reference
if trouble.get("common_issues_title"):
    st.divider()
    st.subheader(trouble.get("common_issues_title", "📋 Quick Reference"))
    st.markdown(trouble.get("common_issues_table", "| Symptoom | Oorzaak | Fix |\n|---|---|---|"))

st.divider()

# ============================================================================
# SECTION 3: TPU MEASUREMENT TIPS
# ============================================================================
tips = texts.get("tpu_tips", {})

st.header(tips.get("header", "3. Tips voor TPU Metingen"))

st.info(tips.get("moisture_tip", "**Wist je dat?** TPU is zeer gevoelig voor vocht..."))

# Three columns of tips
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(tips.get("thermal_stability_title", "### 🌡️ Thermal Stability"))
    st.caption(tips.get("thermal_stability_text", "Voer altijd een 'Time Sweep' uit..."))
    
    # Extended protocol
    with st.expander("📖 Complete Time-Sweep Protocol"):
        st.markdown(tips.get("thermal_stability_extended", "**Complete Time-Sweep Protocol:**\n..."))

with col2:
    st.markdown(tips.get("hydrolysis_title", "### 💧 Hydrolyse"))
    st.caption(tips.get("hydrolysis_text", "Een snelle daling van G' wijst op vocht..."))
    
    # Extended protocol
    with st.expander("📖 Complete Hydrolyse Prevention"):
        st.markdown(tips.get("hydrolysis_extended", "**Complete Hydrolyse Prevention Protocol:**\n..."))

with col3:
    st.markdown(tips.get("strain_title", "### 🌀 Strain"))
    st.caption(tips.get("strain_text", "Zorg dat je binnen de LVE regio meet..."))
    
    # Extended protocol
    with st.expander("📖 Complete LVE Guide"):
        st.markdown(tips.get("strain_extended", "**Complete LVE (Linear Visco-Elastic) Guide:**\n..."))

# Additional TPU tips
if tips.get("additional_tips_title"):
    st.divider()
    st.subheader(tips.get("additional_tips_title", "🔬 Extra TPU Tips"))
    st.markdown(tips.get("additional_tips", "**1. Sample Loading:**\n..."))

# --- FOOTER ---
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
st.markdown("---")
st.sidebar.caption("RheoApp - v1.0")