"""
translations.py - Multi-language support for RheoApp
Contains all UI texts in Dutch (NL) and English (EN)
"""

def get_translations():
    """
    Returns a dictionary with translations for all UI elements.
    Structure: {"EN": {...}, "NL": {...}}
    """
    
    translations = {
        "EN": {
            # --- HEADER ---
            "title": "RheoApp - TPU Rheology Expert Tool",
            "caption": "💡 *Rheology is 50% measurement and 50% common sense.*",
            
            # --- DISCLAIMER ---
            "disclaimer_title": "⚠️ IMPORTANT DISCLAIMER",
            "disclaimer_text": """
**This is an experimental tool developed via AI-assisted coding (Claude.ai & Gemini).**

- ❌ **NO guarantee** on calculation correctness
- ❌ **NO formal validation** against industry standards  
- ❌ **NOT intended** for critical decisions without verification

**Use at your own risk.** Always validate with hand calculations and validated software.

See **README.md** for complete disclaimer and limitations.
""",
            
            # --- SIDEBAR ---
            "sidebar_title": "🎛️ Control Panel",
            "sidebar_caption": "*Configure your analysis here*",
            "upload_label": "📁 Upload Frequency Sweep",
            "data_loaded": "✅ Data loaded: **{name}**",
            "data_stats": "📊 {points} data points, {temps} temperatures",
            "select_temps": "🌡️ Select Temperatures",
            "select_temps_help": "Choose which temperatures to use for TTS analysis",
            "select_temps_warning": "⚠️ Select at least 3 temperatures for reliable TTS!",
            "ref_temp_label": "📌 Reference Temperature (°C)",
            "ref_temp_help": "Preferably choose the highest temperature (above softening point)",
            "colorscheme": "🎨 Color Scheme",
            "wlf_section": "⚙️ WLF Parameters",
            "expected_tg": "Expected Tg (°C)",
            "expected_tg_help": "For TPU soft segments typically -60°C to -20°C",
            "tg_tip": "💡 Tip: Check with DSC for accurate Tg",
            "reset_button": "🔄 Reset",
            "reset_help": "Reset all shift factors to 0",
            "auto_align": "🚀 Auto-Align",
            "auto_align_help": "Automatic optimization of shift factors",
            "manual_shifts": "🎚️ Manual Shift Factors",
            "help_section_title": "📚 Need Help?",
            "help_section_text": """
**Check the sidebar pages:**

- 🌡️ Theory & Models
- 🧪 Interpretation Guide  
- ⚙️ Data & Troubleshooting
""",
            
            # --- TAB NAMES ---
            "tab1_name": "📈 Master Curve",
            "tab2_name": "🧪 Structure (vGP)",
            "tab3_name": "📉 tan δ Analysis",
            "tab4_name": "🌡️ Thermal (Ea/WLF/VFT)",
            "tab5_name": "🔬 TTS Validation",
            "tab6_name": "🧬 Molecular Analysis",
            "tab7_name": "📊 Dashboard",
            
            # --- TAB 1: MASTER CURVE ---
            "tab1_title": "Master Curve at {temp}°C",
            "tab1_info": """
**💡 What do you see here?**

The Master Curve combines data from different temperatures by shifting them along the frequency axis.
If the curves **perfectly overlap**, your material is **thermorheologically simple** and TTS is valid.

→ For more explanation about TTS: See **🌡️ Theory & Models** in the sidebar
""",
            "shift_trend_title": "📊 Shift Factor Trend",
            "shift_trend_info": "💡 A linear trend indicates Arrhenius behavior; strong curvature indicates WLF.",
            "smooth_export": "💾 Smooth Export (Optional)",
            "smooth_caption": "Use spline smoothing for publication-quality plots",
            "smooth_strength": "Smoothing Strength",
            "smooth_warning": "⚠️ **Note:** Too much smoothing can hide real features. Use moderately!",
            "quick_stats": "🎯 Quick Stats",
            "temperatures": "Temperatures",
            "data_points": "Data Points",
            "freq_range": "Freq Range",
            
            # --- TAB 2: VAN GURP-PALMEN ---
            "tab2_title": "Van Gurp-Palmen (vGP) Structure Analysis",
            "tab2_intro": """
### 🎯 What is this?

The vGP plot is the **'fingerprint'** of your material structure. It plots the phase angle (δ) against 
the complex modulus (|G*|). Since both quantities are intrinsic (not dependent on frequency scale),
all temperature curves **MUST** collapse if your material is thermorheologically simple.

**💡 This is the ultimate TTS validation test!**

→ For detailed interpretation: See **🧪 Interpretation Guide** in the sidebar
""",
            "vgp_warning": """
⚠️ **TPU Reality Check:**

If you see clear **'steps'** or **shifts** between colors here:
- This explains why you might get **negative WLF C₁** values
- The material is **thermorheologically complex** in this T-range
- Hard segments don't melt uniformly

→ **Solution:** Choose T_ref > Softening Point (see Tab 4) or use Arrhenius only

→ **More info:** Go to **🧪 Interpretation Guide** → **Scenarios** → "Negative WLF C₁"
""",
            "morphology_title": "🔍 Morphological Diagnosis",
            "morphology_good": """
**✅ What you WANT to see:**

**Perfect Superposition**
- All curves collapse
- No spreading between colors
- Smooth, continuous line

**Meaning:**
- Homogeneous melt at all T
- TTS is 100% valid
- WLF/Arrhenius reliable
- Hard segments fully melted
""",
            "morphology_bad": """
**❌ Red Flags:**

**Spreading/Steps**
- Curves spread apart
- Clear 'hooks' or 'humps'
- Systematic shift with T

**Meaning:**
- **Thermorheologically COMPLEX**
- Phase heterogeneity active
- Hard-segment domains not uniformly melted
- TTS is mathematical approximation (not physical!)

**Action:**
1. Choose T_ref = highest T
2. Check Softening Point (Tab 4)
3. Possibly: remove lowest T's
""",
            
            # --- TAB 3: TAN DELTA ---
            "tab3_title": "Loss Tangent (tan δ) - Relaxation Spectrum",
            "tab3_info": """
**📖 What does tan δ measure?**

tan(δ) = G''/G' gives the **balance between viscous and elastic** behavior:
- **tan δ < 1**: Elastic dominates (rubber-like, shape-stable)
- **tan δ = 1**: Crossover point (G' = G'')
- **tan δ > 1**: Viscous dominates (flows easily)

**Peaks** in tan δ correspond to characteristic **relaxation times** (τ = 1/ω).
For TPU you often see multiple relaxations from soft vs hard segments.
""",
            "tab3_table_title": "💡 Interpretation for TPU",
            "tab3_table": """
| Observation | Meaning | Process Relevance |
|------------|---------|-------------------|
| **Peak at low ω** | Soft segment relaxation | Determines elastic recovery |
| **Peak shifts with T** | Temperature-dependent mobility | Set processing window |
| **tan δ @ 1 Hz** | Balance at typical process frequency | Coating: <0.3 = shape-stable |
| **Multiple peaks** | Bi-modal relaxations (soft/hard) | Typical for TPU (not problematic) |

→ For more details: See **🌡️ Theory & Models** → **Structural Parameters**
""",
            
            # --- TAB 4: THERMAL ---
            "tab4_title": "Thermal Characterization: Arrhenius, WLF & VFT",
            "tab4_metrics": {
                "ea": "**Ea (Arrhenius):**",
                "softening": "**Estimated Softening Point:**",
                "vft_t0": "**VFT T₀ (Vogel):**",
                "vft_na": "VFT: N/A",
                "wlf_c1": "**WLF C1:**",
                "wlf_c2": "**WLF C2:**",
            },
            "tab4_validation_title": "⚠️ Reference Temperature Validation",
            "tab4_critical_warning": """
🚨 **CRITICAL WARNING**

Your reference temperature ({ref_temp}°C) is **BELOW** the softening point ({t_soft:.1f}°C).

**What does this mean?**
- Hard segments are **not fully melted**
- You're measuring in a **thermorheologically complex** regime
- The Master Curve is a **mathematical approximation**, not physically correct
- WLF parameters may be **unreliable** (possibly negative C₁!)

**Action Required:**
1. Choose T_ref = highest temperature ({t_max}°C)
2. Or: Measure at higher temperatures (> {t_req:.0f}°C)
3. Check Van Gurp-Palmen (Tab 2) for spreading
""",
            "tab4_success": """
✅ **Reference Temperature OK**

You're building the Master Curve from the **homogeneous melt phase**.

- T_ref ({ref_temp}°C) > Softening Point ({t_soft:.1f}°C)
- Hard segments are fully melted ✅
- TTS is physically valid in this regime ✅
- WLF/Arrhenius parameters reliable ✅
""",
            
            # --- TAB 5: TTS VALIDATION ---
            "tab5_title": "TTS Validation via Han & Cole-Cole Plots",
            "tab5_han_title": "1️⃣ Han Plot: G' vs G''",
            "tab5_han_caption": "Danger: If lines spread, morphology changes and TTS is invalid.",
            "tab5_cole_title": "2️⃣ Cole-Cole Plot: η'' vs η'",
            "tab5_cole_caption": "Interpretation: A flattened arc indicates broad molecular weight distribution (MWD).",
            "tab5_quality_title": "⚖️ TTS Quality Control Summary",
            "tab5_r2_label": "**📊 Arrhenius R²**",
            "tab5_slope_label": "**📐 Terminal Slope**",
            "tab5_cross_label": "**⚖️ Crossovers**",
            "tab5_excellent": "✅ Excellent",
            "tab5_good": "✅ Newtonian",
            "tab5_single": "✅ Single",
            "tab5_moderate": "⚠️ Moderate",
            "tab5_weak": "❌ Weak",
            "tab5_problem": "❌ Problem",
            "tab5_none": "⚠️ None",
            "tab5_multiple": "❌ Multiple",
            "tab5_not_reached": "ℹ️ Not reached",
            
            # --- TAB 6: MOLECULAR ---
            "tab6_title": "⚛️ Molecular Analysis & Process Parameters",
            "tab6_intro": """
### 🎯 Why are these parameters crucial?

These molecular parameters are **directly linked** to processability and final product properties:
- **η₀** determines how the material flows at low shear rates (coating, extrusion)
- **Gₙ⁰** gives entanglement density (melt strength, shape stability)
- **τ** (relaxation time) predicts elastic memory effects

→ For complete theory: **🌡️ Theory & Models** → **Structural Parameters**
""",
            "tab6_eta0": "Zero Shear Viscosity (η₀)",
            "tab6_gn0": "Plateau Modulus (Gₙ⁰)",
            "tab6_tau": "Relaxation Time (τ)",
            "tab6_mw_title": "🧬 Molecular Weight Relationship",
            
            # --- TAB 7: DASHBOARD ---
            "tab7_title": "📊 Expert Dashboard - Consolidated Analysis",
            "tab7_intro": """
This dashboard consolidates **all critical parameters** and performs **automatic validation**.
It gives you at a glance the quality and reliability of your TTS analysis.

→ For complete parameter explanation: **🌡️ Theory & Models** → **Quick Calculators**
""",
            "tab7_parameters": "📋 Complete Parameter Overview",
            "tab7_validation": "🔍 Model Reliability & Automatic Validation",
            "tab7_thermal": "**Thermal Models:**",
            "tab7_structural": "**Structural Quality:**",
            "tab7_crossovers": "⚖️ Crossover Points per Temperature",
            "tab7_export": "💾 Data Export - Download Your Results",
            
            # --- ERROR MESSAGES ---
            "no_data_error": "❌ No data found in file. Check file format.",
            "upload_prompt": "👆 Upload a frequency sweep CSV/TXT file to begin.",
            
            # --- EXPORT LABELS ---
            "export_params": "📊 Parameters CSV",
            "export_shifts": "🕒 Shift Factors CSV",
            "export_crossovers": "⚖️ Crossovers CSV",
            "export_mastercurve": "📈 Master Curve CSV",
            
            # --- INSTRUCTIONS ---
            "instructions_title": "ℹ️ **User Instructions** - How to use RheoApp?",
            "instructions": """
## 🚀 Quick Start Guide

### 1. UPLOAD
- Click **"Browse files"** in the sidebar
- Select your frequency sweep data (CSV/TXT)
- Supported formats: TA Instruments, Anton Paar, simple CSV
- Sample name is automatically extracted

### 2. CONFIGURATION
- **Select Temperatures:** Choose which T's to use (minimum 3)
- **Choose Reference T:** Preferably **highest temperature** (above softening point!)
- **Colormap:** Visual preference for plots
- **Expected Tg:** For WLF hint (typical TPU: -40°C)

### 3. ALIGNMENT (Shift Factors)
- **Option A:** Click **"🚀 Auto-Align"** for automatic optimization
- **Option B:** Adjust **manually** with sliders (for fine-tuning)
- **Reset:** Click "🔄 Reset" to start over

### 4. ANALYSIS (7 Tabs)

| Tab | What to Check? | Key Validation |
|-----|---------------|----------------|
| **1. Master Curve** | Curve overlap | Visual TTS check |
| **2. Structure (vGP)** | Thermorheological simplicity | **CRITICAL: Curves must collapse** |
| **3. tan δ** | Relaxation spectrum | Crossover identification |
| **4. Thermal** | Ea, WLF, VFT models | Softening Point vs T_ref |
| **5. TTS Validation** | Han & Cole-Cole | Chemical stability & MWD |
| **6. Molecular** | η₀, G_N⁰ extraction | Processability parameters |
| **7. Dashboard** | All metrics + export | **START HERE for overview** |

### 5. EXPORT
- Go to **Tab 7 (Dashboard)**
- Click the **4 export buttons**:
  1. Parameters CSV (Ea, η₀, WLF, etc.)
  2. Shift Factors CSV (per temperature)
  3. Crossovers CSV
  4. Master Curve CSV (all points)

---

## 📚 Need Help?

**Check the sidebar pages for detailed explanations:**

- **🌡️ Theory & Models:** All formulas and physical background
- **🧪 Interpretation Guide:** How to read the graphs? (vGP, Han, Cole-Cole)
- **⚙️ Data & Troubleshooting:** File formats, error messages, TPU measurement tips

**When in trouble:**
1. Check **Interpretation Guide** → **Practice Scenarios**
2. Review **Data & Troubleshooting** → **Error Messages**
3. Validate with **Theory & Models** → **Quick Calculators**
""",
        },
        
        # --- NEDERLANDSE VERTALINGEN ---
        "NL": {
            # --- HEADER ---
            "title": "RheoApp - TPU Rheologie Expert Tool",
            "caption": "💡 *Rheologie is 50% meten en 50% gezond verstand.*",
            
            # --- DISCLAIMER ---
            "disclaimer_title": "⚠️ BELANGRIJKE DISCLAIMER",
            "disclaimer_text": """
**Dit is een experimentele tool ontwikkeld via AI-assisted coding (Claude.ai & Gemini).**

- ❌ **GEEN garantie** op correctheid van berekeningen
- ❌ **GEEN formele validatie** tegen industriestandaarden  
- ❌ **NIET bedoeld** voor kritische beslissingen zonder verificatie

**Gebruik op eigen risico.** Valideer altijd met handberekeningen en gevalideerde software.

Zie **README.md** voor volledige disclaimer en beperkingen.
""",
            
            # --- SIDEBAR ---
            "sidebar_title": "🎛️ Control Panel",
            "sidebar_caption": "*Configureer je analyse hier*",
            "upload_label": "📁 Upload Frequency Sweep",
            "data_loaded": "✅ Data geladen: **{name}**",
            "data_stats": "📊 {points} datapunten, {temps} temperaturen",
            "select_temps": "🌡️ Selecteer Temperaturen",
            "select_temps_help": "Kies welke temperaturen te gebruiken voor TTS analyse",
            "select_temps_warning": "⚠️ Selecteer minimaal 3 temperaturen voor betrouwbare TTS!",
            "ref_temp_label": "📌 Referentie Temperatuur (°C)",
            "ref_temp_help": "Bij voorkeur de hoogste temperatuur kiezen (boven softening point)",
            "colorscheme": "🎨 Kleurenschema",
            "wlf_section": "⚙️ WLF Parameters",
            "expected_tg": "Verwachte Tg (°C)",
            "expected_tg_help": "Voor TPU zachte segmenten typisch -60°C tot -20°C",
            "tg_tip": "💡 Tip: Check met DSC voor nauwkeurige Tg",
            "reset_button": "🔄 Reset",
            "reset_help": "Reset alle shift factors naar 0",
            "auto_align": "🚀 Auto-Align",
            "auto_align_help": "Automatische optimalisatie van shift factors",
            "manual_shifts": "🎚️ Handmatige Shift Factors",
            "help_section_title": "📚 Hulp Nodig?",
            "help_section_text": """
**Bekijk de sidebar pages:**

- 🌡️ Theorie & Modellen
- 🧪 Interpretatie Gids  
- ⚙️ Data & Troubleshooting
""",
            
            # --- TAB NAMEN ---
            "tab1_name": "📈 Master Curve",
            "tab2_name": "🧪 Structuur (vGP)",
            "tab3_name": "📉 tan δ Analyse",
            "tab4_name": "🌡️ Thermisch (Ea/WLF/VFT)",
            "tab5_name": "🔬 TTS Validatie",
            "tab6_name": "🧬 Moleculaire Analyse",
            "tab7_name": "📊 Dashboard",
            
            # --- TAB 1: MASTER CURVE ---
            "tab1_title": "Master Curve bij {temp}°C",
            "tab1_info": """
**💡 Wat zie je hier?**

De Master Curve combineert data van verschillende temperaturen door ze te verschuiven langs de frequentie-as.
Als de curves **perfect overlappen**, is je materiaal **thermorheologisch simpel** en is TTS geldig.

→ Voor meer uitleg over TTS: Zie **🌡️ Theorie & Modellen** in de sidebar
""",
            "shift_trend_title": "📊 Shift Factor Trend",
            "shift_trend_info": "💡 Een lineaire trend wijst op Arrhenius gedrag; een sterke kromming op WLF.",
            "smooth_export": "💾 Smooth Export (Optioneel)",
            "smooth_caption": "Gebruik spline smoothing voor publicatie-kwaliteit grafieken",
            "smooth_strength": "Smoothing Sterkte",
            "smooth_warning": "⚠️ **Let op:** Te veel smoothing kan echte features verbergen. Gebruik met mate!",
            "quick_stats": "🎯 Quick Stats",
            "temperatures": "Temperaturen",
            "data_points": "Datapunten",
            "freq_range": "Freq Bereik",
            
            # --- TAB 2: VAN GURP-PALMEN ---
            "tab2_title": "Van Gurp-Palmen (vGP) Structuuranalyse",
            "tab2_intro": """
### 🎯 Wat is dit?

De vGP plot is de **'vingerafdruk'** van je materiaalstructuur. Het plot de fasehoek (δ) tegen 
de complexe modulus (|G*|). Omdat deze beide grootheden intrinsiek zijn (niet afhankelijk van frequentie-schaal),
**MOETEN** alle temperatuur-curves samenvallen als je materiaal thermorheologisch simpel is.

**💡 Dit is de ultieme TTS validatie test!**

→ Voor gedetailleerde interpretatie: Zie **🧪 Interpretatie Gids** in de sidebar
""",
            "vgp_warning": """
⚠️ **TPU Realiteit Check:**

Als je hier duidelijke **'trappen'** of **verschuivingen** tussen de kleuren ziet:
- Dit verklaart waarom je mogelijk **negatieve WLF C₁** waarden krijgt
- Het materiaal is **thermorheologisch complex** in dit T-bereik
- Harde segmenten smelten niet uniform

→ **Oplossing:** Kies T_ref > Softening Point (zie Tab 4) of gebruik alleen Arrhenius model

→ **Meer info:** Ga naar **🧪 Interpretatie Gids** → **Scenario's** → "Negatieve WLF C₁"
""",
            "morphology_title": "🔍 Morfologische Diagnose",
            "morphology_good": """
**✅ Wat je WILT zien:**

**Perfecte Superpositie**
- Alle curves vallen samen
- Geen spreiding tussen kleuren
- Gladde, continue lijn

**Betekenis:**
- Homogene smelt bij alle T
- TTS is 100% geldig
- WLF/Arrhenius betrouwbaar
- Harde segmenten volledig gesmolten
""",
            "morphology_bad": """
**❌ Red Flags:**

**Spreiding/Trappen**
- Curves wijken af
- Duidelijke 'haken' of 'bulten'
- Systematische verschuiving met T

**Betekenis:**
- **Thermorheologisch COMPLEX**
- Fase-heterogeniteit actief
- Hard-segment domeinen niet uniform gesmolten
- TTS is wiskundige benadering (niet fysisch!)

**Actie:**
1. Kies T_ref = hoogste T
2. Check Softening Point (Tab 4)
3. Mogelijk: verwijder laagste T's
""",
            
            # --- TAB 3 EN VERDER (kopieer van EN voor nu) ---
            "tab3_title": "Loss Tangent (tan δ) - Relaxatie Spectrum",
            "tab3_info": """
**📖 Wat meet tan δ?**

tan(δ) = G''/G' geeft de **balans tussen visceus en elastisch** gedrag:
- **tan δ < 1**: Elastisch domineert (rubber-achtig, vorm-stabiel)
- **tan δ = 1**: Crossover punt (G' = G'')
- **tan δ > 1**: Visceus domineert (vloeit gemakkelijk)

**Peaks** in tan δ corresponderen met karakteristieke **relaxatietijden** (τ = 1/ω).
Voor TPU zie je vaak meerdere relaxaties van zachte vs harde segmenten.
""",
            "tab3_table_title": "💡 Interpretatie voor TPU",
            "tab3_table": """
| Observatie | Betekenis | Procesrelevantie |
|------------|-----------|------------------|
| **Peak bij lage ω** | Zachte segment relaxatie | Bepaalt elastische terugvering |
| **Peak verschuift met T** | Temperatuur-afhankelijke mobiliteit | Verwerkingsvenster instellen |
| **tan δ @ 1 Hz** | Balans bij typische proces-frequentie | Coating: <0.3 = vorm-stabiel |
| **Meerdere peaks** | Bi-modale relaxaties (soft/hard) | Typisch voor TPU (niet problematisch) |

→ Voor meer details: Zie **🌡️ Theorie & Modellen** → **Structurele Parameters**
""",
            
            # Rest kopiëren van EN
            "tab4_title": "Thermische Karakterisatie: Arrhenius, WLF & VFT",
            "tab4_metrics": {
                "ea": "**Ea (Arrhenius):**",
                "softening": "**Geschat Softening Point:**",
                "vft_t0": "**VFT T₀ (Vogel):**",
                "vft_na": "VFT: N/A",
                "wlf_c1": "**WLF C1:**",
                "wlf_c2": "**WLF C2:**",
            },
            "tab4_validation_title": "⚠️ Referentie Temperatuur Validatie",
            "tab4_critical_warning": """
🚨 **KRITIEKE WAARSCHUWING**

Je referentietemperatuur ({ref_temp}°C) ligt **ONDER** het softening point ({t_soft:.1f}°C).

**Wat betekent dit?**
- Harde segmenten zijn **niet volledig gesmolten**
- Je meet in een **thermorheologisch complex** regime
- De Master Curve is een **wiskundige benadering**, niet fysisch correct
- WLF parameters kunnen **onbetrouwbaar** zijn (mogelijk negatieve C₁!)

**Actie Vereist:**
1. Kies T_ref = hoogste temperatuur ({t_max}°C)
2. Of: Meet bij hogere temperaturen (> {t_req:.0f}°C)
3. Check Van Gurp-Palmen (Tab 2) voor spreiding
""",
            "tab4_success": """
✅ **Referentie Temperatuur OK**

Je bouwt de Master Curve vanuit de **homogene smeltfase**.

- T_ref ({ref_temp}°C) > Softening Point ({t_soft:.1f}°C)
- Harde segmenten zijn volledig gesmolten ✅
- TTS is fysisch geldig in dit regime ✅
- WLF/Arrhenius parameters betrouwbaar ✅
""",
            
            "tab5_title": "TTS Validatie via Han & Cole-Cole Plots",
            "tab5_han_title": "1️⃣ Han Plot: G' vs G''",
            "tab5_han_caption": "Gevaar: Als lijnen spreiden, verandert morfologie en is TTS ongeldig.",
            "tab5_cole_title": "2️⃣ Cole-Cole Plot: η'' vs η'",
            "tab5_cole_caption": "Interpretatie: Een afgeplatte boog duidt op brede molecuulgewichtsverdeling (MWD).",
            "tab5_quality_title": "⚖️ TTS Kwaliteitscontrole Samenvatting",
            "tab5_r2_label": "**📊 Arrhenius R²**",
            "tab5_slope_label": "**📐 Terminal Slope**",
            "tab5_cross_label": "**⚖️ Crossovers**",
            "tab5_excellent": "✅ Uitstekend",
            "tab5_good": "✅ Newtoniaans",
            "tab5_single": "✅ Enkelvoudig",
            "tab5_moderate": "⚠️ Matig",
            "tab5_weak": "❌ Zwak",
            "tab5_problem": "❌ Probleem",
            "tab5_none": "⚠️ Geen",
            "tab5_multiple": "❌ Meervoudig",
            "tab5_not_reached": "ℹ️ Niet bereikt",
            
            "tab6_title": "⚛️ Moleculaire Analyse & Procesparameters",
            "tab6_intro": """
### 🎯 Waarom zijn deze parameters cruciaal?

Deze moleculaire parameters zijn **direct gekoppeld** aan verwerkbaarheid en eindproduct eigenschappen:
- **η₀** bepaalt hoe het materiaal vloeit bij lage schuifsnelheden (coating, extrusion)
- **Gₙ⁰** geeft entanglement-dichtheid (melt strength, vorm-stabiliteit)
- **τ** (relaxatietijd) voorspelt elastische memory effecten

→ Voor complete theorie: **🌡️ Theorie & Modellen** → **Structurele Parameters**
""",
            "tab6_eta0": "Zero Shear Viscosity (η₀)",
            "tab6_gn0": "Plateau Modulus (Gₙ⁰)",
            "tab6_tau": "Relaxatietijd (τ)",
            "tab6_mw_title": "🧬 Molecuulgewicht Relatie",
            
            "tab7_title": "📊 Expert Dashboard - Geconsolideerde Analyse",
            "tab7_intro": """
Dit dashboard consolideert **alle kritieke parameters** en voert **automatische validatie** uit.
Het geeft je in één oogopslag de kwaliteit en betrouwbaarheid van je TTS analyse.

→ Voor volledige parameter uitleg: **🌡️ Theorie & Modellen** → **Snelle Calculators**
""",
            "tab7_parameters": "📋 Volledige Parameter Overzicht",
            "tab7_validation": "🔍 Model Betrouwbaarheid & Automatische Validatie",
            "tab7_thermal": "**Thermische Modellen:**",
            "tab7_structural": "**Structurele Kwaliteit:**",
            "tab7_crossovers": "⚖️ Crossover Punten per Temperatuur",
            "tab7_export": "💾 Data Export - Download je Resultaten",
            
            "no_data_error": "❌ Geen data gevonden in bestand. Controleer bestandsformaat.",
            "upload_prompt": "👆 Upload een frequency sweep CSV/TXT bestand om te beginnen.",
            
            "export_params": "📊 Parameters CSV",
            "export_shifts": "🕒 Shift Factors CSV",
            "export_crossovers": "⚖️ Crossovers CSV",
            "export_mastercurve": "📈 Master Curve CSV",
            
            "instructions_title": "ℹ️ **Gebruiksinstructies** - Hoe gebruik je RheoApp?",
            "instructions": """
## 🚀 Quick Start Guide

### 1. UPLOAD
- Klik **"Browse files"** in de sidebar
- Selecteer je frequency sweep data (CSV/TXT)
- Ondersteunde formaten: TA Instruments, Anton Paar, simpele CSV
- Sample naam wordt automatisch geëxtraheerd

### 2. CONFIGURATIE
- **Selecteer Temperaturen:** Kies welke T's te gebruiken (minimaal 3)
- **Kies Referentie T:** Bij voorkeur **hoogste temperatuur** (boven softening point!)
- **Colormap:** Visuele voorkeur voor plots
- **Verwachte Tg:** Voor WLF hint (typisch TPU: -40°C)

### 3. ALIGNMENT (Shift Factors)
- **Optie A:** Klik **"🚀 Auto-Align"** voor automatische optimalisatie
- **Optie B:** Pas **handmatig** aan met sliders (voor fine-tuning)
- **Reset:** Klik "🔄 Reset" om opnieuw te beginnen

### 4. ANALYSE (7 Tabs)

| Tab | Wat Check Je? | Belangrijkste Validatie |
|-----|---------------|-------------------------|
| **1. Master Curve** | Overlap van curves | Visuele TTS check |
| **2. Structuur (vGP)** | Thermorheologische eenvoud | **KRITIEK: Curves moeten samenvallen** |
| **3. tan δ** | Relaxatie spectrum | Crossover identificatie |
| **4. Thermisch** | Ea, WLF, VFT modellen | Softening Point vs T_ref |
| **5. TTS Validatie** | Han & Cole-Cole | Chemische stabiliteit & MWD |
| **6. Moleculair** | η₀, G_N⁰ extractie | Verwerkbaarheid parameters |
| **7. Dashboard** | Alle metrics + export | **START HIER voor overzicht** |

### 5. EXPORT
- Ga naar **Tab 7 (Dashboard)**
- Klik op de **4 export buttons**:
  1. Parameters CSV (Ea, η₀, WLF, etc.)
  2. Shift Factors CSV (per temperatuur)
  3. Crossovers CSV
  4. Master Curve CSV (alle punten)

---

## 📚 Hulp Nodig?

**Bekijk de sidebar pages voor gedetailleerde uitleg:**

- **🌡️ Theorie & Modellen:** Alle formules en fysische achtergrond
- **🧪 Interpretatie Gids:** Hoe lees je de grafieken? (vGP, Han, Cole-Cole)
- **⚙️ Data & Troubleshooting:** File formats, error messages, TPU meet-tips

**Bij problemen:**
1. Check **Interpretatie Gids** → **Praktijk Scenario's**
2. Bekijk **Data & Troubleshooting** → **Foutmeldingen**
3. Valideer met **Theorie & Modellen** → **Snelle Calculators**
"""
        }
    }
    
    return translations