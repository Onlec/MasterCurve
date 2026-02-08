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
    page_title="Data & Troubleshooting - RheoApp",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Data Specificaties & Troubleshooting")
st.markdown("""
Zorg ervoor dat je data correct is geformatteerd om fouten in de berekeningen te voorkomen. 
Deze pagina helpt je bij het voorbereiden van je bestanden, het oplossen van veelvoorkomende problemen,
en geeft TPU-specifieke meetadvies.
""")

# Tabs voor verschillende aspecten
tab_format, tab_errors, tab_tpu, tab_prep = st.tabs([
    "📄 Data Formaat",
    "🔧 Foutmeldingen & Oplossingen",
    "🧪 TPU Meetadvies",
    "📋 Meetvoorbereiding Checklist"
])

with tab_format:
    st.header("📄 Data Format Specificaties")
    
    st.info("""
    **Ondersteunde Bestanden:**
    - `.csv` (comma-separated values)
    - `.txt` (tab-separated values)
    
    **Encodings:**
    - UTF-16 LE (met BOM)
    - UTF-8 (met/zonder BOM)
    - Latin-1 (ISO-8859-1)
    - Decimaal komma → punt conversie automatisch
    """)
    
    st.divider()
    
    # Vereiste kolommen
    st.subheader("🔢 Vereiste Kolommen")
    
    col_req1, col_req2 = st.columns(2)
    
    with col_req1:
        st.markdown("**Minimale Vereisten:**")
        
        required_data = {
            "Kolom": [
                "Temperature",
                "Angular Frequency",
                "Storage Modulus",
                "Loss Modulus"
            ],
            "Alias Namen": [
                "Temp, T, Temp (°C)",
                "omega, ω, Freq (rad/s)",
                "G', Gp, G' (Pa)",
                "G'', Gpp, G'' (Pa)"
            ],
            "Eenheid": [
                "°C",
                "rad/s",
                "Pa",
                "Pa"
            ],
            "Bereik TPU": [
                "150-230°C",
                "0.01-100 rad/s",
                "10²-10⁷ Pa",
                "10²-10⁷ Pa"
            ]
        }
        
        df_required = pd.DataFrame(required_data)
        st.table(df_required)
    
    with col_req2:
        st.markdown("**Optioneel (Berekend indien Afwezig):**")
        
        optional_data = {
            "Kolom": [
                "Complex Viscosity",
                "Phase Angle",
                "tan δ"
            ],
            "Formule (als berekend)": [
                "η* = √(G'² + G''²) / ω",
                "δ = arctan(G''/G')",
                "tan δ = G''/G'"
            ],
            "Alternatieve Namen": [
                "eta*, Eta_star, |η*|",
                "delta, δ, Phase (°)",
                "tan_delta, Loss Factor"
            ]
        }
        
        df_optional = pd.DataFrame(optional_data)
        st.table(df_optional)
        
        st.success("""
        💡 **Tip:** Je hoeft alleen G', G'', ω, en T aan te leveren.
        De rest berekent RheoApp automatisch!
        """)
    
    st.divider()
    
    # Bestandsstructuur voorbeeld
    st.subheader("📋 Voorbeeld Bestandsstructuur")
    
    st.markdown("**Optie 1: Eenvoudig CSV (aanbevolen)**")
    
    st.code("""
T (°C),omega (rad/s),G' (Pa),G'' (Pa)
170,0.1,1250,3400
170,0.25,2100,4800
170,0.63,3800,6200
180,0.1,850,2100
180,0.25,1450,3200
...
    """, language="csv")
    
    st.markdown("**Optie 2: TA Instruments / Anton Paar Export (native)**")
    
    st.code("""
<Header info>
Sample Name	TPU_Sample_001
Date	2024-02-08
Interval data:	Point No.	Temperature	Angular Frequency	Storage Modulus	Loss Modulus
		1	170	0.1	1250	3400
		2	170	0.25	2100	4800
		...
    """, language="text")
    
    st.info("""
    **Native Instrument Formats:**
    
    RheoApp kan direct bestanden lezen van:
    - ✅ TA Instruments TRIOS (export als TXT)
    - ✅ Anton Paar RheoCompass (export als CSV)
    - ✅ Thermo HAAKE (tab-separated TXT)
    
    Selecteer in je software: "Export All Data" of "Export Interval Data"
    """)
    
    st.divider()
    
    # Veelvoorkomende formatfouten
    st.subheader("⚠️ Veelvoorkomende Format Fouten")
    
    error_col1, error_col2 = st.columns(2)
    
    with error_col1:
        st.markdown("**Fout 1: Decimaal Komma**")
        
        st.error("""
        ❌ **Probleem:**
        ```
        G' (Pa)
        1250,50
        2100,30
        ```
        Komma wordt als duizendtal-scheidingsteken gezien!
        """)
        
        st.success("""
        ✅ **Oplossing:**
        ```
        G' (Pa)
        1250.50
        2100.30
        ```
        Of: RheoApp converteert automatisch komma→punt
        """)
    
    with error_col2:
        st.markdown("**Fout 2: Spaties in Headers**")
        
        st.error("""
        ❌ **Probleem:**
        ```
        Storage Modulus (Pa)
        ```
        Spaties kunnen parsing verstoren
        """)
        
        st.success("""
        ✅ **Oplossing:**
        ```
        Storage_Modulus_Pa
        OF
        G' (Pa)
        ```
        Gebruik korte, duidelijke namen
        """)
    
    st.divider()
    
    # Sample Name Extractie
    st.subheader("🏷️ Sample Naam Extractie")
    
    st.markdown("""
    RheoApp probeert automatisch de sample naam te vinden in je bestand.
    
    **Zoeklocaties (in volgorde):**
    1. Regel 3, Kolom 2 (TA Instruments formaat)
    2. Header met "Sample" of "Name"
    3. Fallback: "Onbekend_Sample"
    
    **Voorbeeld:**
    """)
    
    st.code("""
<Header>
Sample ID	TPU_Batch_42A    ← Deze naam wordt gebruikt
Date	2024-02-08
Interval data: ...
    """, language="text")
    
    st.info("""
    💡 **Tip:** Zet de sample naam altijd in de eerste 3 regels van je bestand
    voor automatische detectie.
    """)

with tab_errors:
    st.header("🔧 Foutmeldingen & Oplossingen")
    
    st.markdown("""
    Herken je één van deze foutmeldingen? Hier vind je de oorzaak en oplossing.
    """)
    
    # Error selector
    error_type = st.selectbox(
        "Selecteer je foutmelding:",
        [
            "❌ 'Geen data gevonden in het bestand'",
            "⚠️ 'VFT fit niet succesvol'",
            "📉 'Terminal Slope = N/A'",
            "🔄 'η₀ extrapolatie mislukt'",
            "⚖️ 'Geen crossover punten gevonden'",
            "🌡️ 'Negatieve WLF C1 waarde'",
            "📊 'R² < 0.90 - Zwakke fit'",
            "🔴 'Master Curve overlapt niet goed'"
        ]
    )
    
    st.divider()
    
    # Error uitwerkingen
    if error_type == "❌ 'Geen data gevonden in het bestand'":
        st.error("### ❌ Geen Data Gevonden")
        
        diag_col1, diag_col2 = st.columns(2)
        
        with diag_col1:
            st.markdown("**Mogelijke Oorzaken:**")
            
            st.markdown("""
            1️⃣ **Verkeerde Delimiter**
            ```
            Bestand gebruikt spaties ipv tabs
            Of: puntkomma's ipv komma's
            ```
            
            2️⃣ **Missende Header "Interval data:"**
            ```
            TA Instruments formaat vereist
            deze specifieke regel
            ```
            
            3️⃣ **Encoding Probleem**
            ```
            Exotische karakterset
            Niet-standaard line endings
            ```
            
            4️⃣ **Lege Kolommen**
            ```
            Vereiste kolommen bevatten
            geen data of alleen nullen
            ```
            
            5️⃣ **Verkeerde Kolomnamen**
            ```
            Headers komen niet overeen
            met verwachte namen
            ```
            """)
        
        with diag_col2:
            st.markdown("**Oplossingen:**")
            
            st.success("""
            **Stap 1: Check Bestand in Notepad/TextEdit**
            
            ```
            - Open bestand in teksteditor
            - Zoek naar "Interval data:"
            - Check of tabs (\t) of komma's aanwezig zijn
            - Kijk naar eerste 10 regels
            ```
            """)
            
            st.info("""
            **Stap 2: Export Opnieuw**
            
            ```
            In je rheometer software:
            1. Selecteer "Export All Data"
            2. Kies formaat: "Tab-delimited text (.txt)"
            3. Zorg dat headers included zijn
            4. Save en probeer opnieuw
            ```
            """)
            
            st.warning("""
            **Stap 3: Handmatig Aanpassen**
            
            ```
            Maak nieuw bestand met structuur:
            
            T (°C)	omega (rad/s)	G' (Pa)	G'' (Pa)
            170	0.1	1250	3400
            170	0.25	2100	4800
            ...
            
            Let op: Gebruik TAB tussen kolommen!
            ```
            """)
        
        st.code("""
        DEBUG CHECKLIST:
        
        ☐ Bestand bevat "Interval data:" regel (voor TA format)
          OF bevat directe header rij
        
        ☐ Kolommen gescheiden door tabs of komma's (niet spaties)
        
        ☐ Headers bevatten woorden: "Temperature", "Frequency", "Modulus"
        
        ☐ Minimaal 4 kolommen aanwezig
        
        ☐ Data rijen bevatten numerieke waarden (geen tekst)
        
        ☐ Minimaal 5 datapunten per temperatuur
        
        ☐ Geen lege rijen in de data
        """, language="text")
    
    elif error_type == "⚠️ 'VFT fit niet succesvol'":
        st.warning("### ⚠️ VFT Fit Mislukt")
        
        st.markdown("""
        **Symptoom:** VFT model convergeert niet, T₀ = N/A in dashboard
        
        **Oorzaken & Oplossingen:**
        """)
        
        vft_col1, vft_col2 = st.columns(2)
        
        with vft_col1:
            st.markdown("**Waarom faalt VFT?**")
            
            st.info("""
            **Oorzaak 1: Te Weinig Temperaturen**
            ```
            n < 4 temperaturen
            
            VFT heeft 3 parameters (A, B, T₀)
            Minimaal 4 punten nodig voor fit
            ```
            
            **Oplossing:**
            - Meet minimaal 5 verschillende temperaturen
            - Span minimaal 40°C range
            """)
            
            st.info("""
            **Oorzaak 2: Smalle T-Range**
            ```
            ΔT < 30°C
            
            Onvoldoende kromming om
            T₀ nauwkeurig te bepalen
            ```
            
            **Oplossing:**
            - Vergroot temperatuurbereik naar 50-80°C
            - Meet bij extremen (hoge én lage T)
            """)
        
        with vft_col2:
            st.info("""
            **Oorzaak 3: T₀ Bounds Conflict**
            ```
            Fit probeert T₀ boven min(T_data)
            te zetten → bounds violation
            ```
            
            **Oplossing:**
            - Pas Tg hint aan (dichter bij werkelijke Tg)
            - Accepteer: VFT werkt niet altijd
            - Gebruik WLF als alternatief
            """)
            
            st.success("""
            **Oplossing 4: VFT is Optioneel!**
            ```
            VFT fit falen is GEEN probleem
            
            Gebruik in plaats daarvan:
            - Arrhenius (boven Tm)
            - WLF (rond Tg+50K)
            
            Deze zijn vaak stabieler en
            voldoende voor TPU analyse
            ```
            """)
        
        st.warning("""
        **💡 Praktische Tip:**
        
        Voor standaard TPU analyse:
        ```
        T < 180°C: Gebruik WLF (als stabiel)
        T > 180°C: Gebruik Arrhenius
        VFT: Alleen als je BEIDE regimes wilt combineren
        ```
        
        VFT is 'nice to have', niet 'must have'!
        """)
    
    elif error_type == "📉 'Terminal Slope = N/A'":
        st.info("### 📉 Terminal Slope Niet Berekend")
        
        st.markdown("""
        **Symptoom:** Dashboard toont "N/A" voor Terminal Slope
        
        **Betekenis:** Onvoldoende datapunten in de vloeizone
        """)
        
        ts_col1, ts_col2 = st.columns(2)
        
        with ts_col1:
            st.markdown("**Diagnose:**")
            
            st.code("""
            RheoApp zoekt punten met:
            1. Delta (δ) > 75°
            2. Omega (ω) in laagste 30%
            3. Minimaal 3 punten
            
            Als één voorwaarde faalt:
            → Terminal Slope = N/A
            """, language="text")
            
            st.error("""
            **Meest Voorkomende Oorzaak:**
            ```
            Frequentiebereik stopt te vroeg!
            
            Laagste gemeten ω = 1 rad/s
            Maar terminal zone begint vaak
            pas bij ω < 0.1 rad/s
            
            G" domineert nog niet genoeg
            → δ blijft onder 75°
            ```
            """)
        
        with ts_col2:
            st.markdown("**Oplossingen:**")
            
            st.success("""
            **Oplossing 1: Meet bij Lagere Frequenties**
            ```
            Huidige range: 0.1 - 100 rad/s
            Nieuwe range: 0.01 - 100 rad/s
            
            Dit geeft meer punten waar:
            - G" > G' (vloeigedrag)
            - Delta > 75°
            - Terminal zone bereikt
            ```
            """)
            
            st.info("""
            **Oplossing 2: Check Visueel**
            ```
            In Tab 1 (Master Curve):
            
            - Zie je G' afvlakken bij lage ω?
            - Is er een Newtoniaans plateau?
            
            NEE → Meet lagere frequenties
            JA → Mogelijk meetfout (strain?)
            ```
            """)
            
            st.warning("""
            **Oplossing 3: Accepteer Limitatie**
            ```
            Als lagere ω niet mogelijk:
            
            - Terminal Slope blijft N/A
            - Gebruik andere validaties:
              * Van Gurp-Palmen
              * Han Plot
              * R² checks
            
            Terminal Slope is hulpmiddel,
            geen vereiste voor TTS!
            ```
            """)
        
        st.info("""
        **Typische TPU Meetbereiken:**
        
        | Doel | Optimale ω Range |
        |------|------------------|
        | Terminal Slope bepalen | 0.01 - 100 rad/s |
        | Alleen Master Curve | 0.1 - 100 rad/s |
        | Snelle screening | 1 - 100 rad/s |
        
        Investeer in lage frequenties voor volledige karakterisatie!
        """)
    
    # Compacte versies voor overige errors
    elif error_type == "🔄 'η₀ extrapolatie mislukt'":
        st.info("""
        **Probleem:** Cross model fit convergeert niet
        
        **Oorzaken:**
        - Te weinig data in terminal zone (ω < 0.1 rad/s)
        - Newtoniaans plateau niet bereikt
        - Ruis in viscositeitsdata
        
        **Oplossingen:**
        1. Meet bij lagere frequenties (0.01-0.1 rad/s)
        2. Verhoog smoothing in Tab 1
        3. Check of η* afvlakt bij lage ω
        4. Gebruik alternatief: η₀ ≈ η*(ω_min) als rough estimate
        """)
    
    elif error_type == "⚖️ 'Geen crossover punten gevonden'":
        st.warning("""
        **Probleem:** G' en G'' kruisen nergens in meetbereik
        
        **Scenario A: G' > G'' overal**
        - Materiaal is sterk elastisch (gel-achtig)
        - Mogelijke oorzaken:
          * Hard-segmenten nog kristallijn
          * Crosslinked netwerk
          * Meettemperatuur te laag
        - Oplossing: Meet bij hogere temperaturen (+20-30°C)
        
        **Scenario B: G'' > G' overal**
        - Materiaal is puur visceus (olie-achtig)
        - Mogelijke oorzaken:
          * Zeer laag molecuulgewicht
          * Ernstige degradatie/hydrolyse
          * Meettemperatuur te hoog
        - Oplossing: Check sample kwaliteit, GPC meting
        
        **Scenario C: Crossover buiten bereik**
        - Crossover bij ω > 100 rad/s (buiten range)
        - Oplossing: Extend frequency range if possible
        """)
    
    elif error_type == "🌡️ 'Negatieve WLF C1 waarde'":
        st.error("""
        **Probleem:** WLF C₁ < 0 (fysisch onmogelijk!)
        
        **Root Cause:** Thermorheologisch complex materiaal
        
        **Diagnose Stappen:**
        1. Check Van Gurp-Palmen (Tab 2) → Spreiding?
        2. Check T_ref vs Softening Point (Tab 4) → Onder softening?
        3. Check shift factor plot → Kromming keert om?
        
        **Oplossingen:**
        1. Kies T_ref = hoogste meettemperatuur
        2. Verwijder laagste 1-2 temperaturen
        3. Gebruik alleen Arrhenius (negeer WLF)
        4. Accepteer: Materiaal is inherent complex
        
        → Zie "Interpretatie Gids" tab voor gedetailleerde uitleg
        """)
    
    elif error_type == "📊 'R² < 0.90 - Zwakke fit'":
        st.warning("""
        **Probleem:** Arrhenius R² of Adj. R² < 0.90
        
        **Betekenis:** Shift factors volgen geen lineaire wet
        
        **Oorzaken:**
        1. **WLF Gedrag:** T-bereik te dicht bij Tg
           - Oplossing: Gebruik WLF model
        2. **Fase-overgangen:** Kristallisatie in range
           - Oplossing: Verwijder problematische T's
        3. **Te Weinig Data:** n < 5 temperaturen
           - Oplossing: Meet meer temperaturen
        4. **Complex Materiaal:** Inherente heterogeniteit
           - Oplossing: Accepteer lage R², focus op trends
        
        **Beslisboom:**
        ```
        R² < 0.90
        ├→ WLF R² beter? → Gebruik WLF
        ├→ Meer T's mogelijk? → Herhaal meting
        └→ Anders: Accepteer, gebruik TTS voorzichtig
        ```
        """)
    
    elif error_type == "🔴 'Master Curve overlapt niet goed'":
        st.error("""
        **Probleem:** Curves van verschillende T sluiten niet aan
        
        **Quick Fixes:**
        
        1. **Reset & Auto-Align:**
           - Klik "🔄 Reset" in sidebar
           - Klik "🚀 Auto-Align"
           - Wacht tot optimalisatie klaar is
        
        2. **Verander T_ref:**
           - Kies hoogste temperatuur als referentie
           - Herhaal Auto-Align
        
        3. **Handmatige Fine-Tuning:**
           - Focus op terminal zone (lage ω)
           - Verschuif met sliders tot best overlap
           - Hoge ω mag iets minder perfect
        
        4. **Verwijder Probleem-Temperaturen:**
           - Deselect laagste 1-2 temperaturen
           - Deze zijn vaak thermorheologisch complex
        
        **Als Niets Helpt:**
        → Materiaal is fundamenteel thermorheologisch complex
        → Check Van Gurp-Palmen (moet ook spreiding tonen)
        → Accepteer: TTS werkt niet perfect voor dit TPU
        """)

with tab_tpu:
    st.header("🧪 TPU-Specifiek Meetadvies")
    
    st.info("""
    **Thermoplastische Polyurethanen (TPU) zijn NIET zoals normale polymeren!**
    
    De gesegmenteerde blokcopolymeer structuur (zachte + harde segmenten) vereist
    speciale aandacht bij rheologische metingen.
    """)
    
    st.divider()
    
    # TPU Challenges
    st.subheader("⚠️ TPU Meet-Uitdagingen")
    
    challenge_col1, challenge_col2 = st.columns(2)
    
    with challenge_col1:
        st.markdown("### 1️⃣ Vocht Gevoeligheid")
        
        st.error("""
        **Probleem:**
        TPU is **extreem hygroscopisch**
        
        ```
        Urethaan bindingen: R-NH-CO-O-R'
                           ↓ + H₂O (hydrolyse)
        R-NH₂ + CO₂ + HO-R'
        ```
        
        **Gevolgen:**
        - Mw daalt tijdens verwerking/meting
        - η₀ keldert (soms 50% daling!)
        - Schuimen in smelt (CO₂ vorming)
        - Niet-reproduceerbare resultaten
        
        **Herkenning:**
        - η₀ lager dan verwacht
        - Bellen in sample tijdens meting
        - Time-sweep toont daling G'
        - Batch-to-batch variatie > 30%
        """)
        
        st.success("""
        **VERPLICHTE DROOG PROCEDURE:**
        
        ```
        1. Pre-dry: 80°C, 4-6 uur, VACUUM oven
           (niet circulatie oven - te langzaam!)
        
        2. Check: Vochtmeter < 0.02% (200 ppm)
        
        3. Direct verwerken: < 30 min na drogen
        
        4. Opslag: Desiccator met P₂O₅
        ```
        
        **Vuistregel:**
        Als in twijfel: droog extra 2 uur!
        Te droog bestaat niet voor TPU.
        """)
    
    with challenge_col2:
        st.markdown("### 2️⃣ Thermische Instabiliteit")
        
        st.warning("""
        **Probleem:**
        Vrije NCO groepen reageren na bij T > 200°C
        
        ```
        NCO + NCO → Carbodiimide (crosslink)
        NCO + H₂O → CO₂ + Amine
        Amine + NCO → Urea (hard segment)
        ```
        
        **Gevolgen:**
        - Crosslinking tijdens meting
        - G' stijgt in time-sweep
        - Han Plot: Opwaartse shift
        - Sample verkleurt (bruinig)
        
        **Herkenning:**
        - Time-sweep niet stabiel (G' ↑)
        - Han Plot curves divergeren
        - Sample wordt stijver tijdens meting
        - Visuele kleurverandering
        """)
        
        st.success("""
        **PREVENTIE STRATEGIE:**
        
        ```
        1. Max T: < 220°C (bij voorkeur < 210°C)
        
        2. N₂ Purge: ALTIJD (voorkomt oxidatie)
        
        3. Time-Sweep Check:
           - 10 min @ max T, 1 Hz
           - G' mag max ±5% driften
        
        4. Snelle Meting:
           - Gebruik preset frequency sweep
           - Minimale tijd @ hoge T
        
        5. Fresh Sample:
           - Niet hergebruiken
           - Nieuwe pellets per meting
        ```
        
        **Vuistregel:**
        Snelheid > Nauwkeurigheid bij T > 200°C
        Liever 5 punten/decade snel, dan 10 punten/decade traag
        """)
    
    st.divider()
    
    # Optimale Meetcondities
    st.subheader("✅ Optimale TPU Meetcondities")
    
    cond_col1, cond_col2, cond_col3 = st.columns(3)
    
    with cond_col1:
        st.markdown("### 🌡️ Temperatuur")
        
        st.info("""
        **Range Bepaling:**
        
        ```
        1. Check DSC:
           - Tm hard-segment = ?
           - Tg soft-segment = ?
        
        2. Temperatuur Planning:
           Min T = Tm + 20K
           Max T = Min(230°C, Degradatie - 10K)
           Stappen = 10-20°C
        
        3. Aantal Temperaturen:
           - Minimaal: 5
           - Optimaal: 6-8
           - Span: 50-80°C
        ```
        
        **Voorbeeld:**
        ```
        Hard-segment Tm = 180°C
        
        Meetplan:
        170°C (check TTS grenzen)
        180°C
        190°C ← Goede T_ref
        200°C
        210°C
        220°C (max, kort)
        ```
        """)
    
    with cond_col2:
        st.markdown("### 📊 Frequentie")
        
        st.info("""
        **Range Optimalisatie:**
        
        ```
        Voor VOLLEDIGE karakterisatie:
        ω_min = 0.01 rad/s (terminal zone!)
        ω_max = 100 rad/s (glasachtig plateau)
        
        Punten/decade = 5-7
        Totaal ≈ 25-35 punten
        ```
        
        **Prioriteiten:**
        
        1. **Terminal Zone (0.01-1 rad/s)**
           - Kritisch voor η₀
           - Nodig voor Terminal Slope
           - Proces-relevant
        
        2. **Mid-Range (1-10 rad/s)**
           - Crossover gebied
           - TTS overlap check
        
        3. **Hoog (10-100 rad/s)**
           - Nice-to-have
           - Minder proces-relevant
        
        **Quick Scan:**
        ```
        Tijd-kritisch? Gebruik:
        0.1 - 100 rad/s
        3 punten/decade
        Minder data, maar 3x sneller
        ```
        """)
    
    with cond_col3:
        st.markdown("### 🎚️ Strain")
        
        st.error("""
        **LVE Regime = LAW!**
        
        ```
        Lineair Visco-Elastisch gebied:
        - G' en G'' onafhankelijk van strain
        - Materiaal niet permanent vervormd
        - Nodig voor TTS geldigheid
        ```
        
        **TPU Typisch:**
        ```
        LVE Limiet = 1-10% strain
        (afhankelijk van T en Mw)
        
        Safe value: 5% strain
        Conservative: 1-3% strain
        ```
        
        **VERPLICHTE STRAIN SWEEP:**
        ```
        Voor Elke Nieuwe Batch:
        
        1. Strain Sweep @ T_mid, ω = 1 rad/s
           Range: 0.1% - 100%
        
        2. Bepaal LVE limiet:
           G' daalt < 5% = nog lineair
           G' daalt > 5% = niet-lineair
        
        3. Kies Werk-Strain:
           = 50% van LVE limiet
        
        Voorbeeld:
        LVE limiet = 8% → Gebruik 4%
        ```
        """)
    
    st.divider()
    
    # Sample Voorbereiding
    st.subheader("🔬 Sample Voorbereiding Protocol")
    
    st.markdown("""
    **Stap-voor-Stap voor Reproduceerbare Resultaten:**
    """)
    
    prep_steps = {
        "Stap": [
            "1. Sample Selectie",
            "2. Drogen",
            "3. Loading",
            "4. Equilibratie",
            "5. Geometrie Check",
            "6. LVE Validatie",
            "7. Meting"
        ],
        "Procedure": [
            "Virgin pellets (GEEN re-grind!), Homogeen lot, Visuele inspectie (geen kleur variatie)",
            "80-90°C vacuum oven, 4-6 uur, Vochtmeter < 0.02%, Direct verwerken (< 30 min)",
            "Pre-heat platen tot T_start, Load sample (net genoeg), Trimming NA sluiten (niet voor!)",
            "Wacht 5-10 min @ elke T, Check torque stabiliteit (< 2% drift), Gebruik N₂ purge",
            "Normal force < 1N (te hoog = squeeze flow), Gap correct (bijv. 1mm voor 25mm platen)",
            "Strain sweep @ T_mid, Kies strain in LVE (typisch 5%), Bevestig met amplitude sweep",
            "Frequency sweep bij elke T, Start bij hoogste T (reset thermal history), 5-10 min equilibratie tussen T"
        ],
        "Kritiek (⚠️)": [
            "Re-grind heeft vaak degradatie!",
            "Vocht = #1 oorzaak slechte data",
            "Trimmen voor sluiten = lucht insluiting",
            "Onvoldoende equilibratie = T-gradiënten",
            "Te hoge N-force = squeeze flow artefacten",
            "Niet-LVE = geen TTS validiteit!",
            "Verkeerde volgorde = thermal history effecten"
        ]
    }
    
    df_prep = pd.DataFrame(prep_steps)
    st.table(df_prep)
    
    st.divider()
    
    # Common Mistakes
    st.subheader("🚫 Top 5 TPU Meet-Fouten (en hoe te vermijden)")
    
    mistake_col1, mistake_col2 = st.columns(2)
    
    with mistake_col1:
        st.error("""
        **Fout #1: Onvoldoende Gedroogd**
        
        Symptoom: Batch-to-batch variatie > 30%
        
        ❌ Fout: "3 uur in lucht oven is genoeg"
        ✅ Goed: "Minimaal 4 uur VACUUM @ 80°C"
        
        Check: Vochtmeter voor EN na drogen
        """)
        
        st.error("""
        **Fout #2: Te Lage Meettemperaturen**
        
        Symptoom: Negatieve WLF C₁, vGP spreiding
        
        ❌ Fout: "Ik meet vanaf 150°C (safety margin)"
        ✅ Goed: "Minimaal Tm + 20K (bijv. 190°C)"
        
        Check: DSC Tm eerst bepalen!
        """)
        
        st.error("""
        **Fout #3: Sample Hergebruiken**
        
        Symptoom: Tweede meting geeft andere curve
        
        ❌ Fout: "Ik gebruik dezelfde sample voor 3 T-ranges"
        ✅ Goed: "Verse sample voor elke volledige sweep"
        
        Reden: Thermal history, degradatie, crosslinking
        """)
    
    with mistake_col2:
        st.error("""
        **Fout #4: Geen Time-Sweep Check**
        
        Symptoom: Han Plot opwaartse shift, crosslinking
        
        ❌ Fout: "Direct frequency sweep zonder validatie"
        ✅ Goed: "Altijd 10 min time-sweep @ max T eerst"
        
        Criterium: G' drift < ±5% = stabiel
        """)
        
        st.error("""
        **Fout #5: Strain Te Hoog**
        
        Symptoom: TTS werkt niet, curves wijken af
        
        ❌ Fout: "Ik gebruik altijd 10% strain (default)"
        ✅ Goed: "Strain sweep eerst, dan 50% van LVE limiet"
        
        TPU vaak: 1-5% strain voor LVE!
        """)

with tab_prep:
    st.header("📋 Meetvoorbereiding Checklist")
    
    st.markdown("""
    Print deze checklist en vink af tijdens je meetvoorbereiding voor consistente, reproduceerbare resultaten.
    """)
    
    st.divider()
    
    # Pre-Measurement Checklist
    st.subheader("✅ PRE-MEASUREMENT CHECKLIST")
    
    checklist_data = {
        "Item": [
            "📦 Sample Voorbereiding",
            "▫️ Virgin pellets (geen re-grind)",
            "▫️ Homogeen lot (zelfde batch)",
            "▫️ Visuele inspectie OK (kleur, vorm)",
            "▫️ Voldoende sample (50-100 g reserve)",
            "",
            "💧 Drogen",
            "▫️ Vacuum oven pre-heat 80°C",
            "▫️ Sample 4-6 uur gedroogd",
            "▫️ Vochtmeter check < 0.02%",
            "▫️ Sample in desiccator tot gebruik",
            "▫️ Max 30 min tussen drogen en meting",
            "",
            "🔬 Instrument Setup",
            "▫️ Rheometer geometrie schoon (solvent + kimwipe)",
            "▫️ Gap zero check uitgevoerd",
            "▫️ N₂ purge aangesloten en getest",
            "▫️ Oven temperatuur kalibratie < 6 maanden oud",
            "▫️ Environmental chamber ingesteld",
            "",
            "📊 Meetprotocol",
            "▫️ DSC Tm bepaald (voor T_min keuze)",
            "▫️ Temperaturen gepland (min 5, span > 50°C)",
            "▫️ Frequentie range: 0.01-100 rad/s",
            "▫️ Strain sweep protocol klaar",
            "▫️ Time-sweep check ingepland @ max T",
            "",
            "💾 Data Management",
            "▫️ Sample ID uniek gekozen",
            "▫️ Logboek entry gemaakt (batch, datum, operator)",
            "▫️ Export instellingen gecontroleerd (tabs, headers)",
            "▫️ Backup locatie ingesteld"
        ],
        "Status": ["☐"] * 30,
        "Opmerkingen": [""] * 30
    }
    
    df_checklist = pd.DataFrame(checklist_data)
    
    # Editable checklist (voor interactief gebruik)
    st.info("💡 **Tip:** Screenshot deze checklist of print als PDF voor gebruik bij de rheometer!")
    
    st.table(df_checklist)
    
    st.divider()
    
    # During Measurement
    st.subheader("🔄 TIJDENS METING - Monitor Punten")
    
    monitor_col1, monitor_col2 = st.columns(2)
    
    with monitor_col1:
        st.markdown("### Real-Time Checks")
        
        st.code("""
        Na elke temperatuur:
        
        ☐ Normal force < 1N
          → Hoger? Sample squeeze flow!
        
        ☐ Torque in range (10-90% max)
          → Te laag = ruis, te hoog = overload
        
        ☐ No visible bubbles
          → Bellen? Vocht of degradatie!
        
        ☐ G' en G'' smooth curves
          → Spikes? Trim fout of gap issue
        
        ☐ Consistent trend met vorige T
          → Discontinuïteit? Check sample
        """, language="text")
    
    with monitor_col2:
        st.markdown("### Abort Criteria")
        
        st.error("""
        **STOP METING ALS:**
        
        ❌ Time-sweep drift > 10% (crosslinking!)
        
        ❌ Sample verkleurt (degradatie)
        
        ❌ Torque pieken / oscillaties (slip)
        
        ❌ G' plotseling stijgt tussen T's (thermal crosslink)
        
        ❌ Bellen zichtbaar in sample (vocht/degradatie)
        
        → Start opnieuw met verse sample
        → Check droogtijd
        → Reduceer max T
        """)
    
    st.divider()
    
    # Post-Measurement
    st.subheader("📝 POST-MEASUREMENT - Data Validatie")
    
    st.code("""
    Direct na meting:
    
    1. ☐ Export data naar veilige locatie
       → Minimaal 2 kopieën (server + USB)
    
    2. ☐ Visuele inspectie plots in software
       → G' > G'' bij hoge ω? ✅
       → Smooth curves? ✅
       → Geen discontinuïteiten? ✅
    
    3. ☐ Sample conditie check
       → Kleur veranderd? → Noteer
       → Bellen? → Noteer (vocht!)
       → Homogeen? → OK
    
    4. ☐ Logboek update
       → Meetcondities
       → Afwijkingen
       → Opvallende observaties
    
    5. ☐ Upload naar RheoApp
       → Check auto-naam extractie
       → Bekijk Van Gurp-Palmen direct
       → Red flags? → Herhaal meting
    
    6. ☐ Geometrie reiniging
       → Warm solvent (tolueen, THF)
       → Droog met N₂
       → Bewaar proper
    """, language="text")
    
    st.divider()
    
    # Troubleshooting Guide
    st.subheader("🆘 Snel Troubleshooting During Measurement")
    
    trouble_data = {
        "Symptoom": [
            "Normal Force stijgt (> 5N)",
            "Torque te laag (< 10%)",
            "Torque pieken/oscillaties",
            "G' daalt in time-sweep",
            "G' stijgt in time-sweep",
            "Bellen in sample",
            "Bruine verkleuring"
        ],
        "Oorzaak": [
            "Sample squeeze flow, Gap te klein",
            "Sample te weinig, Gap te groot, Slip",
            "Rand-instabiliteit, Sample slip",
            "Degradatie, Hydrolyse actief",
            "Crosslinking, NCO na-reactie",
            "Vocht in sample, Degradatie CO₂",
            "Oxidatie, Thermische degradatie"
        ],
        "Directe Actie": [
            "Stop → Reset gap → Fresh sample",
            "Stop → Reduceer gap 0.1mm → Retry",
            "Reduceer strain 50% → Retry",
            "Stop → Check droogtijd → Fresh sample",
            "Stop → Reduceer max T (-20°C) → N₂ check",
            "Abort → Re-dry sample 2h → Retry",
            "Abort → Reduceer T (-30°C) → N₂ flow check"
        ]
    }
    
    df_trouble = pd.DataFrame(trouble_data)
    st.table(df_trouble)

# Sidebar
st.sidebar.markdown("---")
st.sidebar.success("""
**🎯 Quick Links:**

Deze pagina helpt met:
- Data format problemen
- Error message oplossingen
- TPU-specifieke tips
- Meetprotocol checklist

**Top Tips:**
1. Droog ALTIJD vacuum
2. N₂ purge VERPLICHT
3. Time-sweep check @ max T
4. Fresh sample per meting
5. LVE check VOORAF
""")

st.sidebar.info("""
**📞 Nog Vragen?**

Zie ook:
- Tab 1: Theorie & Modellen
- Tab 2: Interpretatie Gids
- README.md voor details
""")