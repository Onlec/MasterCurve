import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from translations import get_translations

# Altijd controleren of de taal er is
if 'lang' not in st.session_state:
    st.session_state.lang = 'EN'

texts = get_translations().get(st.session_state.lang, get_translations()["EN"])

# Pagina configuratie
st.set_page_config(
    page_title="Theorie & Modellen - RheoApp",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Theoretische Achtergrond & Modellen")
st.markdown("""
Deze pagina bevat de wetenschappelijke basis van de RheoApp. Hier vind je de achterliggende formules, 
interactieve visualisaties en de interpretatie van thermische en visco-elastische modellen.
""")

# We splitsen de README secties op in logische tabs
tab_tts, tab_therm, tab_struc, tab_calc = st.tabs([
    "🕒 Time-Temperature Superposition", 
    "🔥 Thermische Modellen (Arrhenius/WLF/VFT)", 
    "🏗️ Structurele Parameters",
    "🧮 Snelle Calculators"
])

with tab_tts:
    st.header("Time-Temperature Superposition (TTS)")
    st.markdown("""
    Het fundamentele principe achter TTS is dat de rheologische respons van een polymeer bij verschillende temperaturen **equivalent** is, 
    mits gecorrigeerd voor een verschuivingsfactor $a_T$.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Fysische basis")
        st.info("""
        **Bij temperatuurverandering:**
        * **Hoge T**: Ketens bewegen sneller → equivalent aan **lage frequentie**
        * **Lage T**: Ketens bewegen trager → equivalent aan **hoge frequentie**
        
        **Het magische inzicht:**
        Wat je bij 200°C en 1 rad/s meet, is hetzelfde als 180°C bij ~10 rad/s 
        (als het materiaal thermorheologisch simpel is!)
        """)
        
        st.markdown("**Verschoven Frequentie:**")
        st.latex(r"\omega_{shifted} = \omega \cdot a_T")
        
        st.markdown("**Waarbij:**")
        st.markdown("""
        - $a_T$ = Shift factor (dimensieloos)
        - $a_T > 1$: Verschuiving naar hogere frequentie (lagere T)
        - $a_T < 1$: Verschuiving naar lagere frequentie (hogere T)
        - Bij referentietemperatuur: $a_T = 1$ (geen verschuiving)
        """)
    
    with col2:
        st.subheader("⚠️ Voorwaarde voor geldigheid")
        st.warning("""
        Het materiaal moet **thermorheologisch simpel** zijn:
        
        ✅ **Vereist voor TPU:**
        - Harde segmenten volledig gesmolten (T > Tm + 20K)
        - Geen fasescheiding tijdens meting
        - Geen chemische veranderingen (crosslinking, degradatie)
        - Homogene morfologie bij alle meettemperaturen
        
        ❌ **TTS NIET geldig als:**
        - Hard-segment kristallisatie aanwezig
        - Fase-overgangen tijdens meting
        - Bi-modale molecuulgewichtsverdeling evolueert met T
        """)
        
        st.info("""
        **Pro Tip:** De **Van Gurp-Palmen plot** (Tab 2 in hoofdapp) is je beste vriend 
        om te controleren of TTS geldig is! Curves moeten samenvallen.
        """)

    st.divider()
    
    # Interactieve demonstratie
    st.subheader("📊 Interactieve TTS Demonstratie")
    
    demo_col1, demo_col2 = st.columns([2, 1])
    
    with demo_col2:
        st.markdown("**Speel met de parameters:**")
        ref_temp_demo = st.slider("Referentie T (°C)", 150, 220, 190, 10)
        target_temp = st.slider("Doeltemperatuur (°C)", 150, 220, 170, 10)
        shift_factor = st.slider("log(aT)", -3.0, 3.0, 0.0, 0.1)
        
        st.metric("Verschuivingsfactor aT", f"{10**shift_factor:.2f}")
        
        if target_temp < ref_temp_demo:
            st.info(f"Bij {target_temp}°C gedraagt het materiaal zich alsof de tijd **{10**shift_factor:.1f}x sneller** loopt (hogere frequentie)")
        elif target_temp > ref_temp_demo:
            st.info(f"Bij {target_temp}°C gedraagt het materiaal zich alsof de tijd **{10**(-shift_factor):.1f}x langzamer** loopt (lagere frequentie)")
    
    with demo_col1:
        # Simpel TTS voorbeeld
        omega_base = np.logspace(-2, 2, 50)
        gp_base = 1e5 * omega_base**2 / (1 + omega_base**2)
        
        omega_shifted = omega_base * (10**shift_factor)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(omega_base, gp_base, 'b-o', label=f'Data bij {ref_temp_demo}°C', markersize=4, alpha=0.7)
        ax.loglog(omega_shifted, gp_base, 'r-s', label=f'Data bij {target_temp}°C (shifted)', markersize=4, alpha=0.7)
        ax.set_xlabel("ω·aT (rad/s)", fontsize=11)
        ax.set_ylabel("G' (Pa)", fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("TTS Principe: Verschuiving langs frequentie-as")
        st.pyplot(fig)
        plt.close()
        
        st.caption("💡 Beweeg de slider en zie hoe data bij verschillende T op elkaar geschoven kan worden!")

    st.divider()
    
    # Beslisboom voor TTS validiteit
    st.subheader("🌳 Beslisboom: Is mijn TTS betrouwbaar?")
    
    col_tree1, col_tree2 = st.columns(2)
    
    with col_tree1:
        st.markdown("""
        ```
        Start hier
            │
            ├─→ Van Gurp-Palmen curves overlappen?
            │   ├─→ JA: ✅ Ga naar volgende check
            │   └─→ NEE: ❌ TTS ONBETROUWBAAR
            │            → Kies hogere T_ref
            │            → Of: accepteer thermorheologisch complex
            │
            ├─→ R² Arrhenius > 0.95?
            │   ├─→ JA: ✅ Ga naar volgende check  
            │   └─→ NEE: ⚠️ Matig betrouwbaar
            │
            ├─→ Terminal Slope ≈ 2.0?
            │   ├─→ JA: ✅ Volledige smelt
            │   └─→ NEE: ⚠️ Check incomplete smelt
            │
            └─→ Alle checks OK?
                └─→ JA: ✅✅✅ TTS BETROUWBAAR!
        ```
        """)
    
    with col_tree2:
        st.success("""
        **Als alles OK is:**
        - Je master curve is fysisch correct
        - η₀ extrapolatie is betrouwbaar
        - WLF/Arrhenius parameters zijn geldig
        - Je kunt voorspellingen doen buiten meetbereik (met voorzichtigheid!)
        """)
        
        st.error("""
        **Als niet OK:**
        - Gebruik TTS ALLEEN voor trend-analyse
        - Vertrouw NIET op absolute waarden (Ea, η₀)
        - Blijf binnen gemeten T-bereik voor voorspellingen
        - Overweeg andere metingen (DMA, DSC) voor morfologie-info
        """)

with tab_therm:
    st.header("🔥 Thermische Modellen")
    
    st.info("""
    **Waarom 3 modellen?** Elk model beschrijft het gedrag in een ander temperatuurregime:
    - **Arrhenius**: Hoge T (homogene smelt, ver boven Tg)
    - **WLF**: Mid-range T (nabij Tg tot ~Tg+100K)
    - **VFT**: Breed bereik (beide regimes combineren)
    """)
    
    # Arrhenius
    st.subheader("1️⃣ Arrhenius Vergelijking")
    st.markdown("**Gebruikt voor:** Homogene smelten ver boven de overgangstemperatuur.")
    
    arr_col1, arr_col2 = st.columns([3, 2])
    
    with arr_col1:
        st.latex(r"\log_{10}(a_T) = \frac{-E_a}{2.303 \cdot R} \cdot \left(\frac{1}{T} - \frac{1}{T_{ref}}\right)")
        
        st.markdown("""
        **Lineaire vorm (voor fitting):**
        """)
        st.latex(r"\log_{10}(a_T) = \text{slope} \cdot \frac{1}{T} + \text{intercept}")
        st.latex(r"E_a = -\text{slope} \times 8.314 \times 2.303 / 1000 \text{ [kJ/mol]}")
        
        st.markdown("""
        **Fysische betekenis:**
        - Ea = Energie-barrière voor moleculaire beweging
        - Hoge Ea → Materiaal is zeer gevoelig voor temperatuur
        - Lineaire relatie tussen log(aT) en 1/T
        """)
    
    with arr_col2:
        st.markdown("**Interpretatie:**")
        st.success("**Ea < 50 kJ/mol**  \nLage T-gevoeligheid (olie-achtig)")
        st.info("**Ea 50-150 kJ/mol**  \nTypisch voor polymeren (TPU meestal 80-120)")
        st.error("**Ea > 150 kJ/mol**  \nZEER gevoelig! Kritisch procesvenster")
        
        st.warning("""
        **Let op bij TPU:**
        Als Ea > 140 kJ/mol:
        - ±5°C variatie in extruder = ±50% viscositeit!
        - Strikte temperatuurcontrole nodig
        - Risico op batch-variabiliteit
        """)

    st.divider()

    # WLF
    st.subheader("2️⃣ WLF (Williams-Landel-Ferry)")
    st.markdown("**Gebruikt voor:** Beschrijving van vrije-volume effecten nabij de glasovergang (Tg).")
    
    wlf_col1, wlf_col2 = st.columns([3, 2])
    
    with wlf_col1:
        st.latex(r"\log_{10}(a_T) = \frac{-C_1(T - T_{ref})}{C_2 + (T - T_{ref})}")
        
        st.markdown("""
        **Universele constanten (bij T_ref = Tg + 50K):**
        - C₁ᵘ = 17.44
        - C₂ᵘ = 51.6 K
        
        **Relatie met Tg:**
        """)
        st.latex(r"T_g \approx T_{ref} - C_2")
        
        st.markdown("""
        **Fysische betekenis:**
        - **C₁**: Maat voor vrije-volume effecten
        - **C₂**: Temperatuurafstand tot Tg (mobiliteitsgrens)
        - Niet-lineaire (gebogen) vorm in Arrhenius plot
        """)
    
    with wlf_col2:
        st.markdown("**Validatie Criteria:**")
        
        st.success("""
        ✅ **Normaal bereik:**
        - C₁: 8-17
        - C₂: 40-60K
        - C₂ ≈ (T_ref - Tg_known)
        """)
        
        st.error("""
        ❌ **Red Flags:**
        - C₁ < 0 → Fysisch onmogelijk!
        - C₁ > 30 → Zeer onwaarschijnlijk
        - C₂ < 20K → Check data
        
        **Oorzaak:** Vaak thermorheologisch complex materiaal
        """)
        
        st.info("""
        **TPU Specifiek:**
        - Zachte segmenten: C₂ ≈ 50-70K
        - Harde segmenten: N/A (kristallijn)
        - Negatieve C₁? → T_ref te laag!
        """)

    st.divider()

    # VFT
    st.subheader("3️⃣ VFT (Vogel-Fulcher-Tammann)")
    st.markdown("**Gebruikt voor:** Flexibel model dat zowel rubber- als smelt-regime kan beschrijven.")
    
    vft_col1, vft_col2 = st.columns([3, 2])
    
    with vft_col1:
        st.latex(r"\log_{10}(a_T) = A + \frac{B}{T - T_0}")
        
        st.markdown("""
        **Parameters:**
        - **A**: Constante (dimensieloos)
        - **B**: Temperatuur-coëfficiënt [K]
        - **T₀ (Vogel temp)**: Theoretische "freeze" temperatuur [K]
        
        **Relatie met Tg:**
        """)
        st.latex(r"T_g \approx T_0 + 50K \text{ (vuistregel voor TPU)}")
        
        st.markdown("""
        **Voordeel boven WLF:**
        - Kan breed temperatuurbereik beschrijven
        - Werkt vaak beter bij complexe systemen
        - Automatische T₀ bepaling (geen Tg input nodig)
        """)
    
    with vft_col2:
        st.markdown("**Interpretatie T₀:**")
        
        st.success("""
        ✅ **Normaal voor TPU:**
        - T₀: -100°C tot -50°C
        - (= zachte segmenten Tg - 50K)
        """)
        
        st.warning("""
        ⚠️ **Let op:**
        - T₀ > 0°C → Zeer onwaarschijnlijk
        - T₀ te dicht bij meettemps → Fit divergeert
        - Check of T₀ < min(T_data) - 10K
        """)
        
        st.info("""
        **Geschatte Tg:**
        Als VFT fit succesvol:
        ```
        Tg ≈ T₀ + 50K
        ```
        Voor TPU zachte segmenten is dit vaak
        rond -40°C tot -20°C
        """)

    st.divider()
    
    # Model vergelijking visualisatie
    st.subheader("📊 Model Vergelijking Visualisatie")
    
    vis_col1, vis_col2 = st.columns([2, 1])
    
    with vis_col2:
        st.markdown("**Stel parameters in:**")
        ea_demo = st.slider("Ea (kJ/mol)", 50, 150, 100, 10)
        c1_demo = st.slider("WLF C₁", 5, 25, 15, 1)
        c2_demo = st.slider("WLF C₂ (K)", 30, 80, 50, 5)
        
        st.markdown("**Observaties:**")
        st.caption("🔵 Arrhenius = lineair in 1/T plot")
        st.caption("🔴 WLF = gebogen vorm")
        st.caption("🟢 VFT = flexibel tussen beide")
    
    with vis_col1:
        T_range = np.linspace(150, 220, 50) + 273.15
        T_ref = 190 + 273.15
        
        # Arrhenius
        log_at_arr = -(ea_demo * 1000) / (8.314 * 2.303) * (1/T_range - 1/T_ref)
        
        # WLF
        log_at_wlf = -c1_demo * (T_range - T_ref) / (c2_demo + (T_range - T_ref))
        
        # VFT (simplified)
        T0_demo = T_ref - c2_demo - 50
        log_at_vft = -3 + 500 / (T_range - T0_demo) - 500/(T_ref - T0_demo)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(T_range - 273.15, log_at_arr, 'b-', linewidth=2, label='Arrhenius')
        ax.plot(T_range - 273.15, log_at_wlf, 'r--', linewidth=2, label='WLF')
        ax.plot(T_range - 273.15, log_at_vft, 'g:', linewidth=2, label='VFT')
        ax.axhline(0, color='black', linestyle=':', alpha=0.3)
        ax.set_xlabel("Temperatuur (°C)", fontsize=11)
        ax.set_ylabel("log(aT)", fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("Vergelijking van Thermische Modellen")
        st.pyplot(fig)
        plt.close()
        
        st.info("""
        **Interpretatie:**
        - Arrhenius en WLF kunnen elkaar kruisen → **Softening Point**
        - VFT probeert beide gedragingen te combineren
        - Bij TPU: gebruik WLF onder Tm, Arrhenius erboven
        """)

with tab_struc:
    st.header("🏗️ Structurele Parameters")
    
    st.markdown("""
    Deze parameters beschrijven de **moleculaire architectuur** van je TPU en zijn direct gekoppeld aan verwerkbaarheid en eindproduct eigenschappen.
    """)
    
    # Plateau Modulus
    st.subheader("1️⃣ Plateau Modulus (G_N⁰)")
    
    gn_col1, gn_col2 = st.columns([3, 2])
    
    with gn_col1:
        st.markdown("**Definitie:** De modulus in het elastische plateau waar ketens verstrikt zijn maar nog niet relaxeren.")
        
        st.latex(r"G_N^0 \propto \frac{\rho R T}{M_e}")
        
        st.markdown("""
        **Waarbij:**
        - ρ = Dichtheid [kg/m³]
        - R = Gasconstante [J/(mol·K)]
        - T = Absolute temperatuur [K]
        - M_e = Entanglement molecuulgewicht [g/mol]
        
        **Fysische betekenis:**
        - Maat voor **netwerkdichtheid**
        - Hoe meer entanglements → hogere G_N⁰
        - Bepaalt elastische terugvering
        """)
    
    with gn_col2:
        st.markdown("**Typische Waarden:**")
        
        st.success("**G_N⁰ = 10⁵-10⁶ Pa**  \nGoed verstrikt polymeer (typisch TPU)")
        
        st.info("**G_N⁰ < 10⁴ Pa**  \nWeinig entanglements  \n(laag Mw of veel zachte segmenten)")
        
        st.warning("**G_N⁰ > 10⁶ Pa**  \nZeer sterk verstrikt  \n(mogelijk crosslinking!)")
        
        st.markdown("---")
        
        st.info("""
        **Procesrelevantie:**
        - Hoge G_N⁰ → Goede melt strength
        - Laag G_N⁰ → Makkelijk te verwerken
        - Verandering tussen batches → Let op Mw drift
        """)

    st.divider()

    # Zero Shear Viscosity
    st.subheader("2️⃣ Zero-Shear Viscosity (η₀)")
    
    eta_col1, eta_col2 = st.columns([3, 2])
    
    with eta_col1:
        st.markdown("**Definitie:** De viscositeit bij oneindige lage afschuifsnelheid (Newtoniaans plateau).")
        
        st.latex(r"\eta_0 \propto M_w^{3.4}")
        
        st.markdown("""
        **Waarom zo belangrijk?**
        
        1. **Meest gevoelige Mw indicator**
           - 15% toename η₀ = ~4% toename Mw
           - Degradatie detector (hydrolyse!)
        
        2. **Directe procesrelevantie**
           - Coating: druipen vs egaal
           - Extrusion: drukopbouw
           - Injection: vultijd
        
        3. **Kwaliteitscontrole**
           - Batch-to-batch referentie
           - Vroege waarschuwing vocht/degradatie
        """)
        
        st.markdown("**Cross Model (voor fitting):**")
        st.latex(r"\eta(\omega) = \frac{\eta_0}{1 + (\lambda \omega)^n}")
    
    with eta_col2:
        st.markdown("**Typische Waarden TPU:**")
        
        st.success("**η₀ = 10⁴-10⁶ Pa·s**  \nNormaal procesvenster")
        
        st.info("**η₀ < 10³ Pa·s**  \nZeer laag Mw  \n(risico op mechanische zwakte)")
        
        st.error("**η₀ > 10⁶ Pa·s**  \nVerwerkingsproblemen  \n(hoge druk, lange cycli)")
        
        st.markdown("---")
        
        st.warning("""
        **Hydrolyse Alarm:**
        ```
        Δη₀ < -20% vs vorige batch
        →
        Mogelijk vocht in granulaat!
        Check droogtijd (3h @ 80°C)
        ```
        """)
        
        st.info("""
        **Mw Schatting:**
        ```
        η₀_nieuw / η₀_oud = 1.15
        →
        Mw ↑ ≈ 4%
        ```
        """)

    st.divider()

    # Terminal Slope
    st.subheader("3️⃣ Terminal Slope (Vloeigedrag)")
    
    slope_col1, slope_col2 = st.columns([3, 2])
    
    with slope_col1:
        st.markdown("""
        **Definitie:** De helling van log(G') vs log(ω) in de terminal zone (lage frequentie, vloeiend regime).
        
        **Theoretische waarden voor lineaire polymeren:**
        """)
        st.latex(r"\frac{d \log G'}{d \log \omega} = 2.0 \text{ (ideaal)}")
        st.latex(r"\frac{d \log G''}{d \log \omega} = 1.0 \text{ (ideaal)}")
        
        st.markdown("""
        **Fysische betekenis:**
        - Slope = 2.0 → Perfect Newtoniaans vloeigedrag
        - Materiaal relaxeert volledig in terminal zone
        - Ketens zijn volledig ontward bij lage ω
        """)
        
        st.markdown("**Detectie in RheoApp:**")
        st.code("""
        # Selectiecriteria:
        1. Delta > 75° (tan δ > 3.73 → visceus domineert)
        2. Laagste 30% frequentiebereik
        3. Minimaal 3 datapunten
        
        → Linear fit in log-log ruimte
        """)
    
    with slope_col2:
        st.markdown("**Interpretatie:**")
        
        st.success("**Slope = 1.8-2.2**  \n✅ Volledige smelt  \nNewtoniaans gedrag")
        
        st.warning("**Slope = 1.5-1.7**  \n⚠️ Licht afwijkend  \nCheck T_ref vs softening point")
        
        st.error("**Slope < 1.5**  \n❌ PROBLEEM:  \n• Incomplete smelt  \n• Crosslinking  \n• Harde segmenten nog kristallijn")
        
        st.markdown("---")
        
        st.info("""
        **Diagnostiek:**
        
        Als slope te laag:
        1. Check Van Gurp-Palmen (fase-overgangen?)
        2. Verhoog T_ref met 10-20°C
        3. Controleer time-sweep (stabiel?)
        4. DSC check: Tm harde segmenten?
        """)

    st.divider()
    
    # Crossover frequentie
    st.subheader("4️⃣ Crossover Frequentie (G' = G'')")
    
    co_col1, co_col2 = st.columns([3, 2])
    
    with co_col1:
        st.markdown("""
        **Definitie:** Het punt waar elastische en visceuze moduli gelijk zijn.
        
        **Karakteristieke relaxatietijd:**
        """)
        st.latex(r"\tau_{relax} = \frac{1}{\omega_{crossover}}")
        
        st.markdown("""
        **Fysische betekenis:**
        - τ_relax = tijd die ketens nodig hebben om te ontwarren
        - Korte τ (hoge ω_co) → Snelle relaxatie (laag Mw)
        - Lange τ (lage ω_co) → Trage relaxatie (hoog Mw)
        
        **Aantal crossovers:**
        - **1x**: Normaal polymeergedrag ✅
        - **0x**: Puur elastisch (gel) of puur visceus (olie)
        - **2+x**: Thermorheologisch complex! ⚠️
        """)
    
    with co_col2:
        st.markdown("**Typische Waarden:**")
        
        st.success("**ω_co ≈ 0.1-10 rad/s**  \nTypisch TPU  \nτ ≈ 0.1-10 seconden")
        
        st.info("**ω_co > 100 rad/s**  \nSnelle relaxatie  \nLaag Mw of dunne olie-achtig")
        
        st.warning("**ω_co < 0.01 rad/s**  \nZeer trage relaxatie  \nHoog Mw of beginnende gel-vorming")
        
        st.markdown("---")
        
        st.error("""
        **Meerdere Crossovers:**
        
        Mogelijke oorzaken:
        • Bi-modale Mw verdeling
        • Hard-segment kristallisatie
        • Fase-scheiding tijdens meting
        
        → Check Van Gurp-Palmen!
        """)

with tab_calc:
    st.header("🧮 Snelle Calculators & Vuistregels")
    
    st.markdown("""
    Handige tools voor snelle berekeningen tijdens je analyse.
    """)
    
    # Calculator 1: Mw verandering schatten
    st.subheader("1️⃣ Molecuulgewicht Verandering Schatten")
    
    calc1_col1, calc1_col2 = st.columns(2)
    
    with calc1_col1:
        st.markdown("**Van η₀ → Mw verandering:**")
        
        eta0_oud = st.number_input("η₀ oude batch (Pa·s)", value=1e5, format="%.2e")
        eta0_nieuw = st.number_input("η₀ nieuwe batch (Pa·s)", value=1.2e5, format="%.2e")
        
        if eta0_oud > 0:
            delta_eta = (eta0_nieuw - eta0_oud) / eta0_oud * 100
            delta_mw = ((1 + delta_eta/100)**(1/3.4) - 1) * 100
            
            st.markdown("---")
            st.metric("Verandering η₀", f"{delta_eta:+.1f}%")
            st.metric("Geschatte Mw verandering", f"{delta_mw:+.1f}%")
            
            if abs(delta_mw) > 5:
                st.warning(f"⚠️ Significante verandering! Check procesparameters en grondstof kwaliteit.")
            else:
                st.success("✅ Binnen normale batch variatie")
    
    with calc1_col2:
        st.markdown("**Formule:**")
        st.latex(r"\eta_0 \propto M_w^{3.4}")
        st.latex(r"\frac{M_{w,nieuw}}{M_{w,oud}} = \left(\frac{\eta_{0,nieuw}}{\eta_{0,oud}}\right)^{1/3.4}")
        
        st.info("""
        **Interpretatie:**
        
        De η₀ is ZEER gevoelig voor Mw:
        - 10% Mw toename → ~38% η₀ toename
        - 5% Mw afname → ~17% η₀ afname
        
        Daarom is η₀ perfect voor hydrolyse detectie!
        """)

    st.divider()

    # Calculator 2: Tg schatting
    st.subheader("2️⃣ Glasovergangstemperatuur (Tg) Schatting")
    
    calc2_col1, calc2_col2 = st.columns(2)
    
    with calc2_col1:
        st.markdown("**Via WLF Constanten:**")
        
        t_ref = st.number_input("Referentie Temp (°C)", value=190.0, step=10.0)
        c2_input = st.number_input("WLF C₂ (K)", value=50.0, step=5.0)
        
        tg_wlf = t_ref - c2_input
        
        st.markdown("---")
        st.metric("Geschatte Tg (zachte segmenten)", f"{tg_wlf:.1f}°C")
        
        if -60 < tg_wlf < -20:
            st.success("✅ Typisch voor TPU zachte segmenten")
        else:
            st.warning("⚠️ Ongebruikelijke waarde - check WLF fit")
    
    with calc2_col2:
        st.markdown("**Via VFT T₀:**")
        
        t0_input = st.number_input("VFT T₀ (°C)", value=-80.0, step=5.0)
        
        tg_vft = t0_input + 50
        
        st.markdown("---")
        st.metric("Geschatte Tg (VFT regel)", f"{tg_vft:.1f}°C")
        
        st.info("""
        **Formules:**
        
        WLF: $T_g \\approx T_{ref} - C_2$
        
        VFT: $T_g \\approx T_0 + 50K$
        
        **Validatie:** Check met DSC!
        """)

    st.divider()

    # Calculator 3: Procestemperatuur voorspelling
    st.subheader("3️⃣ Optimale Procestemperatuur Bepalen")
    
    calc3_col1, calc3_col2 = st.columns(2)
    
    with calc3_col1:
        st.markdown("**Doel: Vind T waarbij η* = target waarde**")
        
        target_visc = st.number_input("Doel viscositeit (Pa·s)", value=1000.0, format="%.0f")
        process_omega = st.number_input("Proces freq (rad/s)", value=10.0, step=1.0)
        
        st.info("""
        **Typische waarden:**
        - Coating: η* ≈ 500-2000 Pa·s bij ~1 rad/s
        - Extrusion: η* ≈ 1000-5000 Pa·s bij ~10 rad/s  
        - Injection: η* ≈ 100-500 Pa·s bij ~100 rad/s
        """)
    
    with calc3_col2:
        st.markdown("**Workflow:**")
        st.code("""
        1. In Master Curve tab:
           - Zoek waar η*(ω_shifted) = target
           
        2. Bereken:
           ω_actual = ω_shifted / aT
           
        3. Los aT op voor gewenste T:
           - Via Arrhenius: T = f(aT, Ea)
           - Via WLF: T = f(aT, C1, C2)
           
        4. Valideer:
           - Check T > softening point
           - Check T < degradatie temp
        """, language="python")
        
        st.success("""
        💡 **Pro Tip:**
        
        Bouw altijd een veiligheidsmarge in:
        - T_proces = T_optimaal + 10°C
        - Vermijdt grensgedrag bij fluctuaties
        """)

    st.divider()
    
    # Quick Reference Tabel
    st.subheader("📋 Quick Reference: Typische TPU Waarden")
    
    ref_data = {
        "Parameter": [
            "Zero Shear Viscosity (η₀)",
            "Plateau Modulus (G_N⁰)",
            "Terminal Slope",
            "Crossover Freq (ω_co)",
            "Activatie Energie (Ea)",
            "WLF C₁",
            "WLF C₂",
            "VFT T₀",
            "Tg (zachte segmenten)",
            "Tm (harde segmenten)"
        ],
        "Typisch Bereik": [
            "10⁴ - 10⁶ Pa·s",
            "10⁵ - 10⁶ Pa",
            "1.8 - 2.2",
            "0.1 - 10 rad/s",
            "80 - 120 kJ/mol",
            "10 - 17",
            "40 - 60 K",
            "-100°C tot -50°C",
            "-60°C tot -20°C",
            "150°C - 220°C"
        ],
        "Alarm als": [
            "< 10³ of > 10⁷",
            "< 10⁴ of > 10⁷",
            "< 1.5",
            "> 100 of < 0.01",
            "> 150 kJ/mol",
            "< 5 of > 25",
            "< 20 of > 100",
            "> 0°C",
            "> 0°C",
            "N/A (check DSC)"
        ]
    }
    
    df_ref = pd.DataFrame(ref_data)
    st.table(df_ref)

# Sidebar
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Gebruik deze pagina als naslagwerk tijdens je analyse in de hoofdapp!")
st.sidebar.markdown("""
**Snelkoppelingen:**
- 🕒 TTS basis & validatie
- 🔥 Thermische model theorie
- 🏗️ Structurele parameters
- 🧮 Handige calculators
""")