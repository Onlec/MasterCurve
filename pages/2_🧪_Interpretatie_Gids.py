import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from io import BytesIO

from translations import get_translations
# Altijd controleren of de taal er is
if 'lang' not in st.session_state:
    st.session_state.lang = 'EN'

texts = get_translations().get(st.session_state.lang, get_translations()["EN"])

# Pagina configuratie
st.set_page_config(
    page_title="Interpretatie Gids - RheoApp",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Interpretatie & Validatie Gids")
st.markdown("""
Hoe lees je de grafieken in het dashboard? Gebruik deze gids om te bepalen of je meting betrouwbaar is 
en wat de curves vertellen over de structuur van je TPU. **Deze pagina is je 'cheat sheet' voor rheologie!**
""")

# Tabs voor verschillende validatie aspecten
tab_vgp, tab_han, tab_cole, tab_cross, tab_scenarios = st.tabs([
    "📊 Van Gurp-Palmen", 
    "🔬 Han Plot", 
    "🌀 Cole-Cole Plot",
    "⚖️ Crossover Analyse",
    "💼 Praktijk Scenario's"
])

with tab_vgp:
    st.header("📊 Van Gurp-Palmen (vGP) Plot")
    st.markdown("""
    De vGP plot is de **gouden standaard** voor het valideren van Time-Temperature Superposition. 
    Het plot de fasehoek (δ) tegen de complexe modulus (|G*|).
    """)
    
    st.info("""
    **Waarom is deze plot zo krachtig?**
    
    Omdat δ(|G*|) een **intrinsieke eigenschap** is van de moleculaire structuur.
    Als de structuur niet verandert met temperatuur, moeten alle curves samenvallen.
    Als ze niet samenvallen → structuur verandert → TTS is ongeldig!
    """)
    
    st.divider()
    
    # Interpretatie sectie
    st.subheader("🔍 Wat zie je? Interpretatie Gids")
    
    v_col1, v_col2, v_col3 = st.columns(3)
    
    with v_col1:
        st.markdown("### ✅ PERFECT: Superpositie")
        st.success("""
        **Wat je ziet:**
        - Alle curves (verschillende T) vallen op één lijn
        - Geen spreiding, geen 'trappen'
        - Gladde, continue curve
        
        **Betekenis:**
        - Materiaal is thermorheologisch SIMPEL
        - Homogene smelt bij alle T
        - Harde segmenten volledig gesmolten
        
        **Consequentie:**
        - ✅ Master Curve is fysisch correct
        - ✅ WLF/Arrhenius parameters betrouwbaar
        - ✅ η₀ extrapolatie geldig
        - ✅ Voorspellingen buiten meetbereik OK
        """)
        
        st.image("https://via.placeholder.com/300x200/28a745/ffffff?text=Perfect+Overlap", 
                 caption="Ideaal: Alle temperaturen op één lijn", use_container_width=True)
    
    with v_col2:
        st.markdown("### ⚠️ MATIG: Kleine Spreiding")
        st.warning("""
        **Wat je ziet:**
        - Curves liggen dicht bij elkaar
        - Kleine spreiding (< 5° bij zelfde |G*|)
        - Geen extreme afwijkingen
        
        **Betekenis:**
        - Licht thermorheologisch complex
        - Kleine morfologische verschillen
        - Acceptabel voor praktisch gebruik
        
        **Consequentie:**
        - ⚠️ TTS bruikbaar met voorzichtigheid
        - ⚠️ Parameters zijn benaderingen
        - ❌ NIET extrapoleren ver buiten bereik
        - ✅ OK voor proces-optimalisatie
        """)
        
        st.image("https://via.placeholder.com/300x200/ffc107/000000?text=Slight+Spread", 
                 caption="Acceptabel: Kleine spreiding", use_container_width=True)
    
    with v_col3:
        st.markdown("### ❌ SLECHT: Grote Spreiding")
        st.error("""
        **Wat je ziet:**
        - Curves wijken sterk af
        - Duidelijke 'trappen' of 'haken'
        - Systematische verschuiving met T
        
        **Betekenis:**
        - Thermorheologisch COMPLEX
        - Fase-overgangen tijdens meting
        - Hard-segment smelten/kristalliseren
        
        **Consequentie:**
        - ❌ TTS is ONBETROUWBAAR
        - ❌ Ea, η₀ zijn approximaties
        - ❌ Kies hogere T_ref
        - ❌ Accepteer dat materiaal complex is
        """)
        
        st.image("https://via.placeholder.com/300x200/dc3545/ffffff?text=Large+Spread", 
                 caption="Problematisch: Grote spreiding", use_container_width=True)
    
    st.divider()
    
    # Diagnostische beslisboom
    st.subheader("🌳 Diagnostische Beslisboom")
    
    st.code("""
    START: Bekijk Van Gurp-Palmen plot
        │
        ├─→ Alle curves overlappen perfect?
        │   └─→ JA: ✅✅✅ TTS IS GELDIG
        │       ├─→ Ga door met analyse
        │       ├─→ Alle parameters betrouwbaar
        │       └─→ Master Curve is fysisch correct
        │
        ├─→ Curves liggen dicht bij elkaar (< 5° spreiding)?
        │   └─→ JA: ⚠️ TTS MATIG BETROUWBAAR
        │       ├─→ Gebruik voor trend-analyse
        │       ├─→ Wees voorzichtig met absolute waarden
        │       └─→ Niet extrapoleren ver buiten bereik
        │
        └─→ Curves wijken sterk af (> 10° spreiding)?
            └─→ JA: ❌ TTS ONGELDIG IN DIT BEREIK
                ├─→ Mogelijke oorzaken:
                │   ├─ Harde segmenten nog kristallijn
                │   ├─ T_ref < Softening Point
                │   ├─ Fase-scheiding actief
                │   └─ Bi-modale Mw verdeling
                │
                └─→ ACTIES:
                    ├─ Kies hogere T_ref (+20°C)
                    ├─ Verwijder laagste T uit analyse
                    ├─ Check DSC: Tm harde segmenten?
                    └─ Accepteer thermorheologisch complex
    """, language="text")
    
    st.divider()
    
    # Speciale kenmerken voor TPU
    st.subheader("🔬 TPU-Specifieke Kenmerken")
    
    tpu_col1, tpu_col2 = st.columns(2)
    
    with tpu_col1:
        st.markdown("**Typische TPU Patronen:**")
        
        st.info("""
        **Patroon 1: "Trappen" bij Lage |G*|**
        ```
        Curves scheiden bij δ > 70°
        (lage frequentie, vloeizone)
        ```
        **Oorzaak:** Harde segmenten beginnen te kristalliseren
        bij lagere temperaturen
        
        **Oplossing:** Gebruik alleen T > Tm + 20K
        """)
        
        st.info("""
        **Patroon 2: "Bult" of Omlaag Duiken**
        ```
        δ neemt af bij zeer lage |G*|
        (in plaats van constant te blijven)
        ```
        **Oorzaak:** Elastisch netwerk door incomplete smelt
        
        **Oplossing:** Verhoog alle meettemperaturen
        """)
    
    with tpu_col2:
        st.markdown("**Negatieve WLF C₁ Verklaard:**")
        
        st.warning("""
        **Symptoom:**
        Je krijgt C₁ < 0 uit de WLF fit
        
        **Diagnose via vGP:**
        - Check vGP plot
        - Zie je duidelijke spreiding?
        - Dan is je materiaal thermorheologisch complex!
        
        **Waarom gebeurt dit?**
        ```
        T_ref ligt onder softening point
            ↓
        Morfologie verandert met T
            ↓
        Shift factors volgen geen WLF
            ↓
        Fit geeft onzinnige parameters
        ```
        
        **Oplossing:**
        1. Kies T_ref = hoogste meettemperatuur
        2. Of: Gebruik alleen Arrhenius (boven Tm)
        """)
    
    st.divider()
    
    # Quick Check Tool
    st.subheader("⚡ Quick Check: Is Mijn vGP OK?")
    
    with st.expander("🔧 Interactieve Check Tool"):
        st.markdown("**Beantwoord deze vragen over je vGP plot:**")
        
        q1 = st.radio(
            "1. Hoe zien de curves eruit?",
            ["Alle curves vallen samen", "Kleine spreiding (< 5°)", "Grote spreiding (> 10°)"],
            key="vgp_q1"
        )
        
        q2 = st.radio(
            "2. Zie je 'trappen' of 'haken' in de curves?",
            ["Nee, gladde curves", "Ja, bij lage |G*| (δ > 70°)", "Ja, over het hele bereik"],
            key="vgp_q2"
        )
        
        q3 = st.radio(
            "3. Wat is je referentietemperatuur?",
            ["Hoogste meettemperatuur", "Midden van het bereik", "Laagste meettemperatuur"],
            key="vgp_q3"
        )
        
        if st.button("🔍 Analyseer mijn vGP"):
            score = 0
            
            if q1 == "Alle curves vallen samen":
                score += 3
            elif q1 == "Kleine spreiding (< 5°)":
                score += 1
            
            if q2 == "Nee, gladde curves":
                score += 2
            elif q2 == "Ja, bij lage |G*| (δ > 70°)":
                score += 1
            
            if q3 == "Hoogste meettemperatuur":
                score += 1
            
            st.markdown("---")
            
            if score >= 5:
                st.success("""
                ✅✅✅ **UITSTEKEND!**
                
                Je vGP plot ziet er perfect uit. TTS is geldig en betrouwbaar.
                Alle parameters (Ea, η₀, WLF) zijn fysisch correct.
                """)
            elif score >= 3:
                st.warning("""
                ⚠️ **MATIG**
                
                Je vGP toont enige complexiteit. TTS is bruikbaar maar met voorzichtigheid.
                
                **Aanbevelingen:**
                - Gebruik voor trend-analyse
                - Vertrouw niet blind op absolute waarden
                - Overweeg hogere T_ref als je trappen ziet
                """)
            else:
                st.error("""
                ❌ **PROBLEMATISCH**
                
                Je vGP wijst op thermorheologisch complex gedrag.
                
                **Verplichte Acties:**
                1. Kies T_ref = hoogste meting
                2. Verwijder temperaturen waar trappen ontstaan
                3. Check DSC voor Tm hard-segmenten
                4. Overweeg: Materiaal is inherent complex
                """)

with tab_han:
    st.header("🔬 Han Plot (G' vs G'')")
    st.markdown("""
    De Han plot detecteert **chemische veranderingen** tijdens de meting, zoals crosslinking of degradatie.
    Als de moleculaire structuur constant blijft, moeten alle temperatuur-curves samenvallen.
    """)
    
    st.info("""
    **Principe:**
    Voor thermorheologisch simpele materialen is er een unieke relatie:
    ```
    G' = f(G'')
    ```
    Deze relatie hangt NIET af van temperatuur, alleen van de moleculaire architectuur.
    """)
    
    st.divider()
    
    # Interpretatie
    st.subheader("🔍 Wat betekenen verticale verschuivingen?")
    
    han_col1, han_col2 = st.columns(2)
    
    with han_col1:
        st.markdown("### ⬆️ Opwaartse Shift (Bij Hogere T)")
        
        st.error("""
        **Symptoom:**
        ```
        Bij zelfde G'':
        G'(T_hoog) > G'(T_laag)
        
        Curves bij hogere T liggen BOVEN
        ```
        
        **Diagnose: THERMAL CROSSLINKING** 🚨
        
        **Mechanisme:**
        - Vrije NCO groepen reageren na bij hoge T
        - Polymeer wordt stijver tijdens meting
        - Elastisch netwerk ontstaat
        
        **Voor TPU specifiek:**
        - Typisch bij T > 200-220°C
        - Overmatige hard-segment reactie
        - Urea-vorming (H₂O in systeem?)
        
        **ACTIE VEREIST:**
        1. ❌ Reduceer max temperatuur (-20°C)
        2. ❌ Verkort meettijd (snelle sweep)
        3. ✅ Check time-sweep stabiliteit
        4. ✅ Gebruik N₂ purge (oxidatie?)
        """)
        
        st.image("https://via.placeholder.com/300x200/dc3545/ffffff?text=Upward+Shift", 
                 caption="Gevaar: Crosslinking tijdens meting!", use_container_width=True)
    
    with han_col2:
        st.markdown("### ⬇️ Neerwaartse Shift (Bij Hogere T)")
        
        st.warning("""
        **Symptoom:**
        ```
        Bij zelfde G'':
        G'(T_hoog) < G'(T_laag)
        
        Curves bij hogere T liggen ONDER
        ```
        
        **Diagnose: DEGRADATIE** ⚠️
        
        **Mechanisme:**
        - Thermische ketenbreuk
        - Hydrolyse (vocht aanwezig)
        - Oxidatieve afbraak
        
        **Voor TPU specifiek:**
        - Urethaan bindingen breken
        - Mw daalt tijdens meting
        - G' daalt harder dan G'' (elasticiteit verliest eerst)
        
        **ACTIE VEREIST:**
        1. ⚠️ Check vochtgehalte granulaat
        2. ⚠️ Droog minimaal 3h @ 80°C vacuüm
        3. ⚠️ Reduceer max temperatuur
        4. ⚠️ Gebruik N₂ atmosphere (geen lucht!)
        """)
        
        st.image("https://via.placeholder.com/300x200/ffc107/000000?text=Downward+Shift", 
                 caption="Waarschuwing: Degradatie actief", use_container_width=True)
    
    st.divider()
    
    # Ideale situatie
    st.subheader("✅ Ideaal: Perfecte Overlap")
    
    st.success("""
    **Gewenst Patroon:**
    ```
    Alle curves vallen samen
    Geen verticale verschuiving
    Temperatuur heeft GEEN effect op G'(G'') relatie
    ```
    
    **Betekenis:**
    - Materiaal is chemisch stabiel tijdens meting
    - Moleculaire structuur blijft constant
    - Geen ongewenste reacties
    - TTS is geldig ✅
    
    **Dit is wat je ALTIJD wilt zien!**
    """)
    
    st.divider()
    
    # Praktische Check
    st.subheader("⚡ Praktische Check Procedure")
    
    check_col1, check_col2 = st.columns(2)
    
    with check_col1:
        st.markdown("**Stap-voor-Stap:**")
        
        st.code("""
        1. Open Han Plot in hoofdapp (Tab 5)
        
        2. Focus op HET MIDDEN van de plot
           (niet de extremen)
        
        3. Trek een verticale lijn
           bij een willekeurige G'' waarde
        
        4. Lees G' af voor elke temperatuur
        
        5. Vergelijk:
           - Gelijk? → ✅ Stabiel
           - Hoger bij hoge T? → ❌ Crosslinking
           - Lager bij hoge T? → ⚠️ Degradatie
        
        6. Herhaal voor meerdere G'' waarden
        """, language="text")
    
    with check_col2:
        st.markdown("**Beslisboom:**")
        
        st.code("""
        Han Plot Inspectie
            │
            ├─→ Curves vallen samen?
            │   └─→ JA: ✅ Chemisch stabiel
            │
            ├─→ Opwaartse shift bij hoge T?
            │   └─→ JA: ❌ CROSSLINKING
            │       ├─ Reduceer T_max (-20°C)
            │       ├─ Verkort meettijd
            │       └─ Check NCO index
            │
            └─→ Neerwaartse shift bij hoge T?
                └─→ JA: ⚠️ DEGRADATIE
                    ├─ Droog materiaal goed
                    ├─ Gebruik N₂ purge
                    └─ Check stabilisators
        """, language="text")

with tab_cole:
    st.header("🌀 Cole-Cole Plot (η'' vs η')")
    st.markdown("""
    De Cole-Cole plot onthult de **molecuulgewichtsverdeling (MWD)** van je polymeer.
    De vorm van de boog vertelt je of je een smal of breed spectrum aan ketenlengte hebt.
    """)
    
    st.info("""
    **Achtergrond:**
    
    Voor een **monodispers** polymeer (alle ketens even lang):
    - Cole-Cole plot = perfecte halve cirkel
    
    Voor een **polydispers** polymeer (brede MWD):
    - Boog wordt afgeplat
    - Hoe platter → hoe breder de verdeling
    """)
    
    st.divider()
    
    # Interpretatie van vormen
    st.subheader("🔍 Vorm-Interpretatie Gids")
    
    cc_col1, cc_col2, cc_col3 = st.columns(3)
    
    with cc_col1:
        st.markdown("### ⭕ Halve Cirkel")
        
        st.success("""
        **Vorm:**
        Perfecte halve cirkel
        
        **Betekenis:**
        - **Zeer smalle MWD**
        - Bijna monodispers
        - Alle ketens ongeveer even lang
        
        **Typisch voor:**
        - Gefractioneerd polymeer
        - Living polymerisatie
        - Niet voor commerciële TPU!
        
        **Proces-implicatie:**
        - Scherpe viscositeits-T relatie
        - Klein procesvenster
        - Gevoelig voor kleine Mw veranderingen
        """)
        
        st.image("https://via.placeholder.com/250x250/28a745/ffffff?text=Perfect+Circle", 
                 caption="Monodispers: Smalle MWD", use_container_width=True)
    
    with cc_col2:
        st.markdown("### 🌙 Afgeplatte Boog")
        
        st.info("""
        **Vorm:**
        Afgeplatte halve cirkel
        Bredere boog
        
        **Betekenis:**
        - **Brede MWD**
        - Mix van korte en lange ketens
        - Polydispers polymeer
        
        **Typisch voor:**
        - ✅ Commerciële TPU
        - ✅ Step-growth polymeren
        - ✅ Normaal verwachtbaar!
        
        **Proces-implicatie:**
        - ✅ Stabiel procesvenster
        - ✅ Goede melt strength
        - ✅ Minder gevoelig voor fluctuaties
        - ⚠️ Wel: meer lot-variatie mogelijk
        """)
        
        st.image("https://via.placeholder.com/250x250/0dcaf0/000000?text=Flattened+Arc", 
                 caption="Polydispers: Brede MWD (normaal TPU)", use_container_width=True)
    
    with cc_col3:
        st.markdown("### 🎭 Asymmetrisch")
        
        st.warning("""
        **Vorm:**
        Twee bulten of asymmetrisch
        Niet-gladde boog
        
        **Betekenis:**
        - **Bi-modale MWD**
        - Twee populaties ketens
        - Mogelijk blend
        
        **Typisch voor:**
        - Re-grind blends
        - Multi-reactor product
        - Incompatibele menging
        
        **Proces-implicatie:**
        - ⚠️ Onvoorspelbaar gedrag
        - ⚠️ Mogelijk fasescheiding
        - ⚠️ Check grondstof kwaliteit
        - ❌ Kan problemen geven in verwerking
        """)
        
        st.image("https://via.placeholder.com/250x250/ffc107/000000?text=Asymmetric", 
                 caption="Bi-modaal: Twee populaties", use_container_width=True)
    
    st.divider()
    
    # Temperatuur effecten
    st.subheader("🌡️ Wat als de Vorm Verandert met Temperatuur?")
    
    temp_col1, temp_col2 = st.columns(2)
    
    with temp_col1:
        st.error("""
        **Symptoom: Vorm Verandert Systematisch**
        
        Bij hogere T:
        - Boog wordt platter OF
        - Boog wordt ronder
        
        **Diagnose:**
        Dit is een **RODE VLAG** 🚩
        
        **Betekenis:**
        1. **Thermische degradatie actief**
           - MWD wordt breder (ketenbreuk)
           - Boog platter bij hoge T
        
        2. **Crosslinking/Na-reactie**
           - MWD wordt smaller (netwerk)
           - Boog ronder bij hoge T
        
        3. **Fase-scheiding**
           - Morfologie verandert met T
           - Compatibiliteit lost op
        """)
    
    with temp_col2:
        st.success("""
        **Ideaal: Vorm Blijft Constant**
        
        Alle temperaturen:
        - Zelfde boogvorm
        - Alleen positie schuift (door viscositeit)
        - Vorm-factor constant
        
        **Betekenis:**
        - ✅ MWD stabiel met T
        - ✅ Geen chemische verandering
        - ✅ TTS geldig
        - ✅ Betrouwbare metingen
        
        **Dit wil je zien!**
        
        Cole-Cole is een extra validatie naast
        Van Gurp-Palmen en Han Plot.
        """)
    
    st.divider()
    
    # Kwantificatie
    st.subheader("📐 Kwantitatieve Analyse (Geavanceerd)")
    
    st.markdown("""
    **Polydispersiteits Index (PDI) Schatting:**
    
    Hoewel Cole-Cole geen directe PDI geeft, kan de vorm hints geven:
    """)
    
    quant_data = {
        "Boogvorm": [
            "Perfecte cirkel",
            "Licht afgeplat",
            "Matig afgeplat",
            "Sterk afgeplat",
            "Bi-modaal"
        ],
        "Geschatte PDI (Mw/Mn)": [
            "1.0 - 1.2",
            "1.2 - 1.8",
            "1.8 - 2.5",
            "2.5 - 4.0",
            "> 4.0 (multi-modal)"
        ],
        "Verwachting voor TPU": [
            "Zeer zeldzaam",
            "Ongebruikelijk",
            "Normaal",
            "Mogelijk",
            "Problematisch"
        ]
    }
    
    df_quant = pd.DataFrame(quant_data)
    st.table(df_quant)
    
    st.caption("""
    💡 **Let op:** Dit zijn ruwe schattingen! Voor nauwkeurige PDI gebruik GPC/SEC.
    Cole-Cole geeft een kwalitatieve indicatie, geen absolute waarde.
    """)

with tab_cross:
    st.header("⚖️ Crossover Analyse (G' = G'')")
    st.markdown("""
    Het crossover punt (waar G' = G'') markeert de overgang tussen elastisch en visceus gedrag.
    Het aantal crossovers onthult de complexiteit van je materiaal.
    """)
    
    st.divider()
    
    # Aantal crossovers interpretatie
    st.subheader("🔢 Hoeveel Crossovers Heb Je?")
    
    cross_col1, cross_col2, cross_col3 = st.columns(3)
    
    with cross_col1:
        st.markdown("### 1️⃣ Eén Crossover")
        
        st.success("""
        **Patroon:**
        ```
        Lage ω: G'' > G' (visceus)
        Crossover bij ω_co
        Hoge ω: G' > G'' (elastisch)
        ```
        
        **Betekenis:**
        - ✅ **Klassiek polymeergedrag**
        - ✅ Thermorheologisch simpel
        - ✅ Volledige relaxatie bij lage ω
        - ✅ TTS perfect geldig
        
        **Karakteristieke Tijd:**
        """)
        st.latex(r"\tau = \frac{1}{\omega_{co}}")
        st.markdown("""
        Dit is de tijd die ketens nodig
        hebben om volledig te ontwarren.
        
        **Voor TPU:**
        - ω_co ≈ 0.1-10 rad/s (normaal)
        - τ ≈ 0.1-10 seconden
        """)
    
    with cross_col2:
        st.markdown("### 0️⃣ Geen Crossover")
        
        st.warning("""
        **Patroon A: G' > G'' overal**
        ```
        Materiaal is ALTIJD elastisch
        Zelfs bij lage frequentie
        ```
        
        **Betekenis:**
        - Gel-achtig gedrag
        - Crosslinked netwerk
        - Incomplete smelt
        - Kristallijn netwerk actief
        
        **Voor TPU:**
        - ⚠️ Hard-segmenten nog kristallijn
        - ⚠️ Meet bij hogere T
        - ⚠️ Check DSC: Tm bereikt?
        
        ---
        
        **Patroon B: G'' > G' overal**
        ```
        Materiaal is ALTIJD visceus
        Zelfs bij hoge frequentie
        ```
        
        **Betekenis:**
        - Zeer laag Mw (olie-achtig)
        - Degradatie
        - Geen entanglements
        
        **Voor TPU:**
        - ❌ Ernstige hydrolyse
        - ❌ Mw < kritisch Mw
        - ❌ Check materiaal historie
        """)
    
    with cross_col3:
        st.markdown("### 2️⃣+ Meerdere Crossovers")
        
        st.error("""
        **Patroon:**
        ```
        G' en G'' kruisen MEERDERE keren
        Niet-monotone curves
        ```
        
        **Betekenis:**
        - ❌ **Thermorheologisch COMPLEX**
        - ❌ Multi-fase systeem
        - ❌ Relaxatie-processen overlappen
        
        **Mogelijke Oorzaken bij TPU:**
        
        1. **Bi-modale Mw Verdeling**
           - Korte en lange ketens
           - Twee relaxatie-tijden
        
        2. **Hard-Segment Kristallisatie**
           - Kristallen smelten tijdens meting
           - Morfologie verandert met T
        
        3. **Fase-Scheiding**
           - Soft/hard segment incompatibiliteit
           - Twee Tg's actief
        
        4. **Meetfouten (Check!)**
           - Strain te hoog (LVE geschonden)
           - Sample niet equilibreerd
           - Geometrie fout (gap setting)
        
        **ACTIE:**
        1. Check Van Gurp-Palmen (spreiding?)
        2. Verhoog T_ref boven Tm
        3. Verwijder complexe temperaturen
        4. Herhaal meting (validatie)
        """)
    
    st.divider()
    
    # Crossover frequentie interpretatie
    st.subheader("📊 Crossover Frequentie (ω_co) Interpretatie")
    
    st.markdown("""
    De **positie** van het crossover punt vertelt je over de relaxatiedynamiek:
    """)
    
    freq_data = {
        "ω_crossover": [
            "> 100 rad/s",
            "10 - 100 rad/s",
            "0.1 - 10 rad/s",
            "0.01 - 0.1 rad/s",
            "< 0.01 rad/s"
        ],
        "Relaxatietijd τ": [
            "< 0.01 s",
            "0.01 - 0.1 s",
            "0.1 - 10 s",
            "10 - 100 s",
            "> 100 s"
        ],
        "Interpretatie": [
            "Zeer snelle relaxatie (laag Mw, dunne smelt)",
            "Snelle relaxatie (typisch bij hoge T)",
            "✅ Normaal TPU bereik",
            "Trage relaxatie (hoog Mw of lage T)",
            "Zeer trage relaxatie (gel-achtig, crosslinked)"
        ],
        "Proces Implicatie": [
            "Vloeit snel, weinig elastische memory",
            "Goede verwerkbaarheid",
            "Balans elasticiteit/vloei",
            "Moeilijk te verwerken, hoge elastische memory",
            "Zeer moeilijk, risico op defecten"
        ]
    }
    
    df_freq = pd.DataFrame(freq_data)
    st.table(df_freq)
    
    st.divider()
    
    # Praktische check tool
    st.subheader("⚡ Quick Crossover Check")
    
    with st.expander("🔧 Analyseer je Crossover Data"):
        n_cross = st.number_input("Hoeveel crossovers zie je?", min_value=0, max_value=5, value=1, step=1)
        
        if n_cross == 0:
            cross_type = st.radio(
                "Wat zie je?",
                ["G' > G'' overal (altijd elastisch)", "G'' > G' overal (altijd visceus)"]
            )
            
            if cross_type == "G' > G'' overal (altijd elastisch)":
                st.error("""
                ❌ **Incomplete Smelt / Gel Gedrag**
                
                **Diagnose:**
                - Hard-segmenten zijn nog kristallijn
                - Of: Materiaal is crosslinked
                
                **Acties:**
                1. Verhoog alle meettemperaturen (+20°C)
                2. Check DSC: Is Tm bereikt?
                3. Controleer sample historie (geen re-grind?)
                4. Herhaal meting met verse sample
                """)
            else:
                st.error("""
                ❌ **Ernstige Degradatie / Zeer Laag Mw**
                
                **Diagnose:**
                - Mw < entanglement Mw
                - Of: Hydrolyse heeft ketens gebroken
                
                **Acties:**
                1. Check vochtgehalte (DRYING!)
                2. GPC/SEC meting voor Mw
                3. Vergelijk met standaard batch
                4. Check processhistorie
                """)
        
        elif n_cross == 1:
            omega_co = st.number_input("Crossover frequentie (rad/s)", value=1.0, format="%.2f", min_value=0.001)
            tau = 1/omega_co
            
            st.success(f"""
            ✅ **Normaal Gedrag**
            
            Relaxatietijd: τ = {tau:.2f} seconden
            """)
            
            if omega_co > 10:
                st.info("Snelle relaxatie → Materiaal vloeit gemakkelijk")
            elif omega_co < 0.1:
                st.warning("Trage relaxatie → Materiaal heeft sterke elastische memory")
            else:
                st.success("Typisch TPU bereik → Goede balans")
        
        else:
            st.error(f"""
            ❌ **{n_cross} Crossovers = Thermorheologisch Complex**
            
            **Dit is een red flag!**
            
            Mogelijke oorzaken:
            1. Bi-modale Mw verdeling
            2. Fase-scheiding tijdens meting
            3. Hard-segment kristallisatie-effecten
            4. Meetfout (check strain, equilibratie)
            
            **Verplichte Acties:**
            1. Bekijk Van Gurp-Palmen plot
            2. Als spreiding zichtbaar → TTS ongeldig
            3. Verhoog T_ref significant
            4. Overweeg DMA/DSC voor morfologie-info
            """)

with tab_scenarios:
    st.header("💼 Praktijk Scenario's: Van Symptoom naar Oplossing")
    st.markdown("""
    Herken je deze situaties? Gebruik deze gids om snel de oorzaak en oplossing te vinden.
    """)
    
    # Scenario selector
    scenario = st.selectbox(
        "Kies je scenario:",
        [
            "🔴 Mijn WLF C₁ is negatief!",
            "🟠 Master Curve 'trapt' en sluit niet aan",
            "🟡 Terminal Slope is veel lager dan 2.0",
            "🟢 η₀ extrapolatie faalt (fit convergeert niet)",
            "🔵 R² is laag (< 0.90) voor Arrhenius",
            "🟣 Han Plot toont opwaartse shift",
            "⚫ Meerdere crossover punten gevonden"
        ]
    )
    
    st.divider()
    
    # Scenario uitwerkingen
    if scenario == "🔴 Mijn WLF C₁ is negatief!":
        st.error("### 🚨 Negatieve WLF C₁")
        
        diag_col1, diag_col2 = st.columns([2, 1])
        
        with diag_col1:
            st.markdown("""
            **Symptoom:**
            ```
            WLF fit geeft C₁ < 0
            Fysisch onmogelijk!
            ```
            
            **Root Cause Analyse:**
            
            1️⃣ **Check Van Gurp-Palmen**
            ```
            Vraag: Vallen curves samen?
            
            NEE → Thermorheologisch complex!
            └─→ Dit is de oorzaak van negatieve C₁
            ```
            
            2️⃣ **Check T_ref vs Softening Point**
            ```
            In Dashboard Tab 4:
            - Kijk naar "Estimated Softening Point"
            - Is T_ref < Softening Point?
            
            JA → Hard-segmenten nog niet volledig gesmolten
            └─→ Morfologie verandert met T
                └─→ WLF formule niet geldig
            ```
            
            3️⃣ **Check Shift Factor Trend**
            ```
            In Tab 1, rechter kolom:
            - Is de curve sterk gebogen (niet lineair)?
            - Keert de kromming om?
            
            JA → Dit gedrag volgt geen WLF
            └─→ Mogelijk fase-overgang in meetbereik
            ```
            """)
        
        with diag_col2:
            st.markdown("""
            **Oplossingsplan:**
            """)
            
            st.info("""
            **Actie 1: Verhoog T_ref**
            ```
            Kies: Hoogste meettemperatuur
            Of: T > Softening + 20°C
            
            Herhaal Auto-Align
            Check WLF opnieuw
            ```
            """)
            
            st.success("""
            **Actie 2: Gebruik Arrhenius**
            ```
            Boven Tm werkt Arrhenius beter
            Negeer WLF in dit regime
            
            Focus op:
            - Ea waarde
            - R² validatie
            ```
            """)
            
            st.warning("""
            **Actie 3: Accepteer Complexiteit**
            ```
            Als niets helpt:
            
            TPU is inherent
            thermorheologisch complex
            in dit T-bereik
            
            Gebruik TTS alleen voor
            trend-analyse, niet voor
            absolute voorspellingen
            ```
            """)
        
        st.divider()
        
        st.code("""
        BESLISBOOM:
        
        Negatieve WLF C₁
            │
            ├─→ Van Gurp-Palmen spreiding?
            │   └─→ JA: Thermorheologisch complex
            │       └─→ Verhoog T_ref naar hoogste T
            │           └─→ Nog steeds negatief?
            │               └─→ Gebruik alleen Arrhenius
            │
            └─→ T_ref < Softening Point?
                └─→ JA: Hard-segmenten nog aanwezig
                    └─→ Kies T_ref > Softening + 20°C
                        └─→ Nog problemen?
                            └─→ Meet bij hogere temperaturen
        """, language="text")
    
    elif scenario == "🟠 Master Curve 'trapt' en sluit niet aan":
        st.warning("### ⚠️ Master Curve Sluit Niet Aan")
        
        mc_col1, mc_col2 = st.columns(2)
        
        with mc_col1:
            st.markdown("""
            **Symptoom:**
            ```
            Curves van verschillende T
            overlappen niet goed
            
            Duidelijke 'trappen' zichtbaar
            Vooral bij lage frequenties
            ```
            
            **Diagnose Stappen:**
            
            1. **Visuele Inspectie**
               - Waar zitten de trappen?
               - Bij lage ω (terminal zone)?
               - Bij hoge ω (glasachtig plateau)?
               - Over hele bereik?
            
            2. **Van Gurp-Palmen Check**
               - Ga naar Tab 2 in hoofdapp
               - Zie je dezelfde spreiding?
               - JA → Thermorheologisch complex
            
            3. **Auto-Align Resultaat**
               - Probeer opnieuw "🚀 Auto-Align"
               - Verbetert het?
               - NEE → Fysisch onmogelijk om te alignen
            """)
        
        with mc_col2:
            st.markdown("""
            **Oplossingen:**
            
            **Scenario A: Trappen bij LAGE ω**
            ```
            Oorzaak:
            - Hard-segmenten kristalliseren
              bij lage T in terminal zone
            
            Oplossing:
            1. Verhoog T_ref naar hoogste T
            2. Verwijder laagste 1-2 temperaturen
            3. Focus overlap op terminal zone
               (belangrijker voor proces!)
            ```
            
            **Scenario B: Trappen bij HOGE ω**
            ```
            Oorzaak:
            - Glasachtige relaxaties
            - Minder belangrijk voor proces
            
            Oplossing:
            - Accepteer kleine mismatch hoge ω
            - Zolang terminal zone goed overlapt
              is TTS bruikbaar
            ```
            
            **Scenario C: Overal Trappen**
            ```
            Oorzaak:
            - Fundamenteel complex materiaal
            - Bi-modale Mw verdeling
            
            Oplossing:
            - Check Cole-Cole (vorm verandert?)
            - Mogelijk: materiaal is niet geschikt
              voor TTS analyse
            - Gebruik data per temperatuur apart
            ```
            """)
        
        st.success("""
        **💡 Pro Tip: Handmatige Fine-Tuning**
        
        Workflow voor beste alignment:
        
        1. Start met Auto-Align (basis)
        2. Focus op terminal zone (lage ω, hoge T)
           → Dit gebied bepaalt η₀ en procesgedrag
        3. Verschuif handmatig met sliders:
           - Optimaliseer overlap bij G' minimum (tan δ maximum)
        4. Hoge frequentie mag iets minder perfect
        5. Valideer met Van Gurp-Palmen
        
        **Doel:** 90% overlap in terminal zone
        **Niet:** 100% overlap overal (vaak onmogelijk voor TPU)
        """)
    
    elif scenario == "🟡 Terminal Slope is veel lager dan 2.0":
        st.warning("### ⚠️ Lage Terminal Slope")
        
        ts_col1, ts_col2 = st.columns(2)
        
        with ts_col1:
            st.markdown("""
            **Symptoom:**
            ```
            Dashboard toont:
            Terminal Slope G' = 1.3
            (ideaal: 2.0)
            ```
            
            **Wat betekent dit?**
            
            Materiaal vloeit NIET als een normale vloeistof
            in de terminal zone. Er is een structuur die
            relaxatie belemmert.
            
            **Mogelijke Oorzaken:**
            
            1️⃣ **Incomplete Smelt** (meest waarschijnlijk)
            ```
            - Hard-segmenten nog gedeeltelijk kristallijn
            - T_ref < Tm + 20K
            - Netwerk van kristallieten blijft aanwezig
            ```
            
            2️⃣ **Crosslinking**
            ```
            - Chemische na-reactie tijdens meting
            - Check Han Plot (opwaartse shift?)
            - NCO groepen actief bij hoge T
            ```
            
            3️⃣ **Meetfout**
            ```
            - Terminal zone niet bereikt
            - Laagste ω nog te hoog
            - Strain te hoog (LVE geschonden)
            ```
            
            4️⃣ **Inherent Long-Chain Branching**
            ```
            - Vertakte structuur (zeldzaam bij TPU)
            - Check materiaal specificaties
            ```
            """)
        
        with ts_col2:
            st.markdown("""
            **Diagnostiek Workflow:**
            """)
            
            st.code("""
            STAP 1: Check Meetbereik
            ├─→ Dashboard: "Terminal Slope Info"
            ├─→ Hoeveel punten gebruikt (N)?
            │   ├─→ N < 3: ❌ ONVOLDOENDE DATA
            │   │   └─→ Meet bij lagere ω
            │   └─→ N ≥ 3: ✅ Data OK
            │
            STAP 2: Check Temperatuur
            ├─→ Tab 4: Softening Point?
            ├─→ Is T_ref > Softening + 20°C?
            │   ├─→ NEE: ❌ INCOMPLETE SMELT
            │   │   └─→ Verhoog T_ref
            │   └─→ JA: Ga naar Stap 3
            │
            STAP 3: Check Han Plot
            ├─→ Tab 5: Verticale shift?
            │   ├─→ JA (omhoog): ❌ CROSSLINKING
            │   │   └─→ Reduceer T_max
            │   └─→ NEE: Ga naar Stap 4
            │
            STAP 4: Check Sample
            └─→ Herhaal meting met:
                ├─ Hogere temperaturen
                ├─ Lagere strain (1%)
                └─ Verse sample (niet re-grind)
            """, language="text")
            
            st.success("""
            **Meest Waarschijnlijke Fix:**
            
            Voor TPU:
            ```
            Verhoog T_ref met 20-30°C
            
            Typisch:
            - Was: 170°C → Slope = 1.4
            - Nu: 200°C → Slope = 1.9
            
            Hard-segmenten nu volledig gesmolten
            → Newtoniaans vloeigedrag hersteld
            ```
            """)
    
    # Voeg andere scenario's toe (verkorte versies)
    elif scenario == "🟢 η₀ extrapolatie faalt (fit convergeert niet)":
        st.info("### ℹ️ Zero-Shear Viscosity Fit Probleem")
        
        st.markdown("""
        **Symptoom:** Cross model fit faalt, η₀ = N/A
        
        **Oorzaken:**
        1. Te weinig data in terminal zone (ω < 1 rad/s)
        2. Newtoniaans plateau niet bereikt
        3. Ruis in data
        
        **Oplossing:**
        - Meet bij lagere frequenties (0.01-0.1 rad/s)
        - Check of G' vlakt af bij lage ω
        - Verhoog smoothing in Tab 1
        - Als blijft falen: schat η₀ ≈ G'/ω bij laagste ω
        """)
    
    elif scenario == "🔵 R² is laag (< 0.90) voor Arrhenius":
        st.info("### ℹ️ Slechte Arrhenius Fit")
        
        st.markdown("""
        **Symptoom:** R² of Adj. R² < 0.90
        
        **Betekenis:** Shift factors volgen geen lineaire Arrhenius relatie
        
        **Oorzaken:**
        1. Temperatuurbereik te dicht bij Tg (WLF gedrag)
        2. Fase-overgangen in bereik
        3. Te weinig temperaturen (n < 5)
        
        **Oplossing:**
        - Check of WLF beter fit (gebogen vorm = WLF territory)
        - Meet meer temperaturen (minimaal 5)
        - Gebruik alleen T > Softening Point + 30K
        - Overweeg VFT model (beide regimes)
        """)
    
    elif scenario == "🟣 Han Plot toont opwaartse shift":
        st.error("### 🚨 Crosslinking Gedetecteerd!")
        
        st.markdown("""
        **Symptoom:** Bij hoge T liggen curves BOVEN die van lage T
        
        **Diagnose:** THERMAL CROSSLINKING (zeer ernstig!)
        
        **Mechanisme bij TPU:**
        - Vrije NCO groepen reageren na bij T > 200°C
        - Urea/biuret vorming
        - Netwerk ontstaat tijdens meting
        
        **ONMIDDELLIJKE ACTIES:**
        1. ❌ Stop metingen > 200°C
        2. ✅ Reduceer max T met 20-30°C
        3. ✅ Verkort meettijd (snelle sweep)
        4. ✅ Check NCO index materiaal (< 0.5%?)
        5. ✅ Gebruik N₂ purge (geen lucht!)
        6. ✅ Time-sweep check bij hoge T (stabiel?)
        
        **Data Validiteit:**
        ❌ Huidige data is NIET representatief
        ❌ Materiaal is veranderd tijdens meting
        ❌ Herhaal metingen bij lagere T!
        """)
    
    elif scenario == "⚫ Meerdere crossover punten gevonden":
        st.error("### ⚫ Multi-Crossover Systeem")
        
        st.markdown("""
        **Symptoom:** Dashboard toont 2+ crossovers
        
        **Betekenis:** Thermorheologisch complex materiaal
        
        **Mogelijke Oorzaken:**
        1. Bi-modale Mw verdeling
        2. Hard-segment kristallisatie tijdens meting
        3. Fase-scheiding soft/hard domeinen
        4. Meetfout (check strain, equilibratie)
        
        **Diagnostiek:**
        - Van Gurp-Palmen: Spreiding? → Fase-heterogeniteit
        - Cole-Cole: Asymmetrisch? → Bi-modaal
        - Han Plot: Shift? → Chemische verandering
        
        **Actie Plan:**
        1. Verhoog T_ref significant (+30°C)
        2. Verwijder complexe temperaturen uit analyse
        3. Check materiaal historie (blend? re-grind?)
        4. Overweeg DSC/DMA voor morfologie-info
        5. Accepteer: TTS niet perfect geldig voor dit materiaal
        """)
    
    st.divider()
    
    # Algemene troubleshooting checklist
    st.subheader("✅ Algemene Troubleshooting Checklist")
    
    checklist_data = {
        "Check": [
            "Sample Voorbereiding",
            "Meetcondities",
            "Temperatuur Range",
            "Referentie Keuze",
            "Data Kwaliteit",
            "Validatie Plots",
            "Fysische Consistentie"
        ],
        "Wat te Controleren": [
            "Droogtijd (3h @ 80°C vacuüm), geen re-grind mix, homogeen sample",
            "Strain < 5% (LVE!), 5 min equilibratie per T, N₂ purge actief",
            "Minimaal 5 temperaturen, span > 40°C, T_max < degradatie temp",
            "T_ref > Softening Point + 20K, bij voorkeur hoogste T",
            "Reproduceerbaar?, Geen drift in time-sweep?, G' en G'' smooth?",
            "Van Gurp-Palmen overlap?, Han Plot geen shift?, Cole-Cole consistent?",
            "η₀ realistisch (10⁴-10⁶)?, Slope ≈ 2?, Crossovers logisch?"
        ],
        "Red Flags": [
            "Vocht bellen, kleur verandering, inhomogeniteit",
            "Strain > 10%, Drift > 5%, Geen N₂",
            "n < 4, span < 30°C, T_max > 230°C",
            "T_ref < Softening, Laagste T gekozen",
            "Variatie > 20%, Time-sweep daalt, Spikes in data",
            "Spreiding, Verticale shift, Vorm verandert",
            "η₀ < 100, Slope < 1.5, Crossovers > 2"
        ]
    }
    
    df_checklist = pd.DataFrame(checklist_data)
    st.table(df_checklist)

# Sidebar
st.sidebar.markdown("---")
st.sidebar.info("""
**🧭 Navigatie Tips:**

Gebruik deze pagina als je:
- Onverwachte resultaten ziet
- Wilt begrijpen wat plots betekenen
- Snel een probleem wilt oplossen
- Validatie wilt uitvoeren

**Workflow:**
1. Bekijk plots in hoofdapp
2. Kom hier voor interpretatie
3. Gebruik scenario gids voor fixes
""")
st.sidebar.success("""
**💡 Pro Tip:**

Van Gurp-Palmen is je eerste check!
Als die faalt, zijn alle andere
analyses ook verdacht.
""")