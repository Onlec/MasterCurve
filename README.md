# The R.A.T. App - Rheology Analysis Tool

> *"Rheologie is 50% meten en 50% gezond verstand."*

## 📋 Inhoudsopgave

1. [Overzicht](#overzicht)
2. [Theoretische Achtergrond](#theoretische-achtergrond)
3. [Formules & Implementatie](#formules--implementatie)
4. [Tab-per-Tab Uitleg](#tab-per-tab-uitleg)
5. [Installatie & Gebruik](#installatie--gebruik)
6. [Data Format Specificaties](#data-format-specificaties)
7. [Troubleshooting](#troubleshooting)
8. [Referenties](#referenties)

---

## 🎯 Overzicht

**RAT** is een professionele Streamlit-applicatie voor de analyse van frequentie-sweep rheologie data van thermoplastische polyurethanen (TPU). De tool implementeert geavanceerde Time-Temperature Superposition (TTS) principes en biedt diepgaande moleculaire en thermische karakterisatie.

### Kernfunctionaliteiten

- ✅ **Automatische & Handmatige TTS**: Master curve constructie met geoptimaliseerde shift factors
- ✅ **Multi-model Thermische Analyse**: Arrhenius, WLF, en VFT vergelijkingen
- ✅ **Structurele Validatie**: Van Gurp-Palmen, Han Plot, Cole-Cole diagnostiek
- ✅ **Moleculaire Karakterisatie**: Zero-shear viscositeit, plateau modulus, terminal gedrag
- ✅ **Expert Dashboard**: Geconsolideerde metrics met validatie-checks
- ✅ **Professionele Exports**: CSV data met volledige traceerbare parameters

---

## 🧬 Theoretische Achtergrond

### Time-Temperature Superposition (TTS)

Het fundamentele principe achter TTS is dat de rheologische respons van een polymeer bij verschillende temperaturen **equivalent** is, mits gecorrigeerd voor een shift factor `aT`.

**Fysische basis:**
- Bij hogere temperaturen bewegen polymeerketens sneller → materiaal gedraagt zich alsof het bij lagere frequentie gemeten wordt
- Bij lagere temperaturen is de moleculaire beweging trager → equivalent aan hogere frequentie

**Voorwaarde voor geldigheid:**
- Het materiaal moet **thermorheologisch simpel** zijn (homogene smelt, geen fase-overgangen)
- Voor TPU betekent dit: harde segmenten moeten volledig gesmolten zijn

### Thermoplastische Polyurethanen (TPU)

TPU heeft een **gesegmenteerde blokcopolymeer structuur**:

1. **Zachte Segmenten** (polyol + diisocyanaat):
   - Laag Tg (-40 tot -60°C)
   - Verantwoordelijk voor elasticiteit
   - Bepalen de vloei-eigenschappen

2. **Harde Segmenten** (diisocyanaat + chain extender):
   - Hoog Tm (150-220°C)
   - Fungeren als fysische crosslinks
   - Kristalliseren bij afkoeling

**Implicatie voor Rheologie:**
- Onder Tm van harde segmenten: **Thermorheologisch complex** (fase-scheiding)
- Boven Tm: **Thermorheologisch simpel** (homogene smelt)
- De "Softening Point" markeert deze overgang

---

## 📐 Formules & Implementatie

### 1. Basale Rheologische Grootheden

#### 1.1 Complexe Modulus
```python
G* = √(G'² + G"²)
```
- **G'** (Storage Modulus): Elastisch component (opgeslagen energie)
- **G"** (Loss Modulus): Visceus component (gedissipeerde energie)
- **Implementatie**: `g_star = np.sqrt(d['Gp']**2 + d['Gpp']**2)`

#### 1.2 Complexe Viscositeit (Cox-Merz)
```python
η*(ω) = |G*| / ω = √(G'² + G"²) / ω
```
- **Betekenis**: Maat voor de totale stroperigheid bij gegeven frequentie
- **Cox-Merz regel**: η*(ω) ≈ η(γ̇) voor vele polymeren (niet altijd voor TPU!)
- **Implementatie**: `d['eta_s'] = np.sqrt(d['Gp']**2 + d['Gpp']**2) / d['w_s']`

#### 1.3 Fasehoek (Loss Angle)
```python
δ = arctan(G" / G')
```
- **Waarde**: 0° (perfect elastisch) tot 90° (perfect visceus)
- **Terminal zone**: δ > 75° (vloeiend gedrag domineert)
- **Implementatie**: `d['delta'] = np.degrees(np.arctan2(d['Gpp'], d['Gp']))`

#### 1.4 Loss Tangent
```python
tan(δ) = G" / G'
```
- **tan(δ) < 1**: Elastisch gedrag domineert (rubber-achtig)
- **tan(δ) = 1**: Crossover punt (G' = G")
- **tan(δ) > 1**: Visceus gedrag domineert (vloeibaar)
- **Implementatie**: `tan_d = d['Gpp'] / d['Gp']`

---

### 2. Time-Temperature Superposition

#### 2.1 Verschoven Frequentie
```python
ω_shifted = ω · aT
```
- **aT**: Shift factor (dimensieloos)
- **aT > 1**: Verschuiving naar hogere frequentie (lagere T)
- **aT < 1**: Verschuiving naar lagere frequentie (hogere T)
- **Implementatie**: `d['w_s'] = d['omega'] * at` waar `at = 10**log_aT`

#### 2.2 Automatische Optimalisatie
De auto-align functie minimaliseert de afstand tussen curves:

```python
Objectieve functie: Σ(log₁₀(G'_ref(ω_shifted)) - log₁₀(G'_target(ω)))²
```

**Proces:**
1. Interpoleer referentie G'(ω) in log-ruimte
2. Voor elke temperatuur: zoek optimale log(aT) via Nelder-Mead
3. Minimaliseer squared error tussen verschoven en referentie curve

**Implementatie:**
```python
def objective(log_at):
    f = interp1d(np.log10(ref_omega), np.log10(ref_Gp), bounds_error=False)
    v = f(np.log10(target_omega) + log_at)
    return np.sum((v - np.log10(target_Gp))**2)
```

---

### 3. Thermische Modellen

#### 3.1 Arrhenius Vergelijking

**Formule:**
```
log₁₀(aT) = -Ea/(2.303·R) · (1/T - 1/T_ref)

Waarbij:
- Ea = Activatie energie [kJ/mol]
- R = 8.314 J/(mol·K) (gasconstante)
- T = Absolute temperatuur [K]
```

**Lineaire vorm:**
```
log₁₀(aT) = slope · (1/T) + intercept

Ea = -slope · R · ln(10) / 1000
   = -slope · 8.314 · 2.303 / 1000
```

**Implementatie:**
```python
inv_t = 1 / T_kelvin
slope, intercept = np.polyfit(inv_t, log_aT, 1)
ea = abs(slope * 8.314 * np.log(10) / 1000)  # kJ/mol
```

**Interpretatie:**
- **Ea < 50 kJ/mol**: Zwakke temperatuurgevoeligheid (dunne olie-achtig)
- **Ea 50-150 kJ/mol**: Typisch voor polymeer smelten
- **Ea > 150 kJ/mol**: Sterke T-gevoeligheid (let op temperatuur controle!)

**Geldigheid:**
- Werkt goed voor **homogene smelten** boven Tm
- Faalt bij fase-overgangen of in de buurt van Tg

---

#### 3.2 WLF Vergelijking (Williams-Landel-Ferry)

**Formule:**
```
log₁₀(aT) = -C₁(T - T_ref) / (C₂ + T - T_ref)

Waarbij:
- C₁ = Vrije volume parameter (typisch 8-17)
- C₂ = Temperatuurconstante [K] (typisch 40-60K)
- T_ref = Referentie temperatuur
```

**Universele constanten (bij T_ref = Tg + 50K):**
- C₁ᵘ = 17.44
- C₂ᵘ = 51.6 K

**Implementatie:**
```python
def wlf_model(params, T, T_ref):
    C1, C2 = params
    return -C1 * (T - T_ref) / (C2 + (T - T_ref))

# Optimalisatie
def wlf_error(params):
    predicted = wlf_model(params, T_kelvin, T_ref_kelvin)
    return np.sum((log_aT_data - predicted)**2)

result = minimize(wlf_error, x0=[17.4, C2_init], bounds=[(1, 50), (10, 200)])
C1, C2 = result.x
```

**Interpretatie:**
- **C₁ < 0 of C₂ < 0**: Fysisch onmogelijk → thermorheologisch complex!
- **C₁ < 5**: Materiaal gedraagt zich meer Arrhenius-achtig
- **C₁ > 30**: Zeer sterke vrije-volume effecten (atypisch)
- **C₂ ≈ (T_ref - Tg)**: Relatie met glasovergangstemperatuur

**Geschatte Tg:**
```
Tg ≈ T_ref - C₂
```

**Geldigheid:**
- Optimaal tussen **Tg en Tg + 100K**
- Voor TPU: vooral geschikt voor zachte segment fase
- Faalt bij zeer hoge temperaturen (dan Arrhenius beter)

---

#### 3.3 VFT Vergelijking (Vogel-Fulcher-Tammann)

**Formule:**
```
log₁₀(aT) = A + B / (T - T₀)

Waarbij:
- A = Constante
- B = Temperatuur-coëfficiënt
- T₀ = Vogel temperatuur (theoretische "freeze" temp)
```

**Implementatie:**
```python
def vft_model(T, A, B, T0):
    return A + B / (T - T0)

# Fit met strikte bounds
lower_bounds = [-∞, 10, 50]  # T₀ minimaal 50K
upper_bounds = [∞, 5000, min(T_data) - 5]  # T₀ moet onder laagste meting

popt, _ = curve_fit(vft_model, T_kelvin, log_aT, 
                    bounds=(lower_bounds, upper_bounds))
A, B, T0 = popt
```

**Relatie met Tg:**
```
Tg ≈ T₀ + 50K  (empirische regel voor TPU)
```

**Interpretatie T₀ (T∞):**
- **T₀ = -273.15°C (0K)**: Theoretische limiet (bereikt nooit)
- **T₀ typisch tussen -100°C en -50°C** voor TPU zachte segmenten
- **T₀ > 0°C**: Zeer onwaarschijnlijk → fit is waarschijnlijk slecht
- **T₀ te dicht bij meettemperaturen**: Gevaar voor divergentie (explosie)

**Voordeel boven WLF:**
- Kan **beide regimes** beschrijven (rubber EN smelt)
- Meer flexibel bij complexe systemen

---

#### 3.4 Softening Point Bepaling

Het **Softening Point** is de temperatuur waarbij het materiaal overgaat van "hard-segment gedomineerd" naar "homogene smelt".

**Methode:**
```
T_softening = T waarbij |Arrhenius - WLF| minimaal is
```

**Implementatie:**
```python
diff = |log_aT_arrhenius - log_aT_wlf|
softening_idx = np.argmin(diff)
T_softening = T_smooth[softening_idx]
```

**Fysische betekenis:**
- **T < T_softening**: Harde segmenten nog gedeeltelijk kristallijn
  - WLF curve wijkt sterk af
  - TTS is onbetrouwbaar (thermorheologisch complex)
  
- **T > T_softening**: Volledige smelt
  - Arrhenius domineert
  - TTS is geldig

**Waarschuwing voor gebruiker:**
```python
if ref_temp < T_softening:
    ⚠️ Master Curve gebouwd in complex gebied!
    → Kies hogere referentietemperatuur
```

---

### 4. Structurele Parameters

#### 4.1 Plateau Modulus (G_N⁰)

**Definitie:**
Het plateau in G' bij middelhoge frequenties, waar ketens **verstrikt** (entangled) zijn maar nog niet volledig relaxeren.

**Verbeterde Methode:**
```python
# Selecteer elastisch regime: G' > 2·G"
plateau_zone = data[data['Gp'] > 2 * data['Gpp']]

if len(plateau_zone) > 3:
    G_N0 = plateau_zone['Gp'].median()  # Mediaan = robuust
else:
    G_N0 = data['Gp'].max()  # Fallback
```

**Waarom mediaan i.p.v. gemiddelde?**
- **Robuuster** tegen uitschieters bij hoge/lage frequentie
- Vermijdt invloed van terminal zone (G' stijgt niet meer)

**Relatie met Moleculair Gewicht:**
```
G_N⁰ ∝ ρRT/M_e

Waarbij:
- M_e = Entanglement molecuulgewicht
- ρ = Dichtheid
- R = Gasconstante
- T = Temperatuur
```

**Typische waarden:**
- **G_N⁰ < 10⁴ Pa**: Zwak verstrikt (laag Mw of veel zachte segmenten)
- **G_N⁰ = 10⁵-10⁶ Pa**: Goed verstrikt polymeer
- **G_N⁰ > 10⁶ Pa**: Sterk verstrikt of crosslinked

---

#### 4.2 Zero-Shear Viscosity (η₀)

**Definitie:**
De viscositeit bij **oneindige lage schuifsnelheid** (ω → 0), waar het materiaal Newtoniaans gedrag vertoont.

**Cross Model:**
```
η(ω) = η₀ / (1 + (λ·ω)ⁿ)

Waarbij:
- η₀ = Zero-shear viscosity [Pa·s]
- λ = Relaxatietijd [s]
- n = Shear-thinning exponent (0.5-1.0)
```

**Implementatie:**
```python
def cross_model(omega, eta_0, tau, n):
    return eta_0 / (1 + (tau * omega)**n)

# Fit via curve_fit
p0 = [eta_complex.max(), 0.1, 0.8]  # Startwaarden
popt, _ = curve_fit(cross_model, omega_data, eta_data, p0=p0)
eta_0, lambda, n = popt
```

**Molecuulgewicht Relatie:**
```
η₀ ∝ M_w^3.4  (voor lineaire polymeren)

Gevolg:
- 15% toename in η₀ → ~4% toename in M_w
- Zeer gevoelige indicator voor degradatie!
```

**Procesrelevantie voor TPU:**
- **Coating applications**: η₀ bepaalt of film egaal blijft liggen (te laag → druipen)
- **Extrusion**: η₀ voorspelt drukopbouw in de extruder
- **Hydrolyse detectie**: Daling van η₀ = degradatie door vocht

**Typische waarden:**
- **η₀ < 10³ Pa·s**: Laag Mw, eenvoudig te verwerken
- **η₀ = 10⁴-10⁶ Pa·s**: Typisch TPU procesvenster
- **η₀ > 10⁶ Pa·s**: Hoog Mw, risico op verwerkingsproblemen

---

#### 4.3 Terminal Slope

**Definitie:**
De helling van log(G') vs log(ω) in de **terminal zone** (lage frequentie, vloeiend gedrag).

**Theoretische waarde voor lineaire polymeren:**
```
d(log G') / d(log ω) = 2.0  (ideaal Newtoniaans)
d(log G") / d(log ω) = 1.0
```

**Verbeterde Detectie (Professor's Methode):**
```python
# Selectiecriteria:
# 1. Delta > 75° (vloeiend gedrag)
# 2. Laagste 30% van frequentiebereik

cutoff_freq = data['omega'].quantile(0.3)
terminal_zone = data[(data['delta'] > 75) & (data['omega'] <= cutoff_freq)]

if len(terminal_zone) >= 3:
    slope = np.polyfit(np.log10(terminal_zone['omega']), 
                       np.log10(terminal_zone['Gp']), 1)[0]
```

**Waarom delta > 75°?**
```
tan(δ) = G"/G'

Als δ > 75° → tan(δ) > 3.73 → G" >> G'
Dit garandeert dat we in de vloeizone zitten!
```

**Interpretatie:**
- **Slope = 2.0 ± 0.2**: Perfect Newtoniaans gedrag (volledige smelt)
- **Slope < 1.7**: 
  - Onvolledige smelt (harde segmenten nog aanwezig)
  - Mogelijke crosslinking tijdens meting
  - Gevaar voor "sharkskin" defecten bij verwerking
  
- **Slope > 2.3**:
  - Zeer ongebruikelijk
  - Mogelijk degradatie tijdens meting (ketenbreuk)

---

#### 4.4 Crossover Detectie

**Definitie:**
Het punt waar G' = G" (elastisch = visceus).

**Methode (Log-Lineaire Interpolatie):**
```python
def find_all_crossovers(omega, Gp, Gpp):
    crossovers = []
    diff = np.log10(Gp) - np.log10(Gpp)
    
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:  # Tekenwisseling
            # Lineaire interpolatie in log-ruimte
            frac = |diff[i]| / (|diff[i]| + |diff[i+1]|)
            omega_co = 10^(log(ω_i) + frac·(log(ω_{i+1}) - log(ω_i)))
            modulus_co = 10^(log(G'_i) + frac·(log(G'_{i+1}) - log(G'_i)))
            
            crossovers.append({"omega": omega_co, "modulus": modulus_co})
    
    return crossovers
```

**Waarom log-ruimte?**
- Rheologie data is **exponentieel** verdeeld
- Lineaire interpolatie in log-ruimte = exponentiële interpolatie in reële ruimte
- **Veel nauwkeuriger** dan directe lineaire interpolatie!

**Aantal Crossovers & Interpretatie:**

1. **Geen crossover (0x)**:
   - G' > G" over hele bereik → **Sterk elastisch** (gel-achtig)
   - G" > G' over hele bereik → **Puur visceus** (zeer laag Mw)

2. **Één crossover (1x)**:
   - **Thermorheologisch simpel**
   - Klassiek polymeer gedrag
   - TTS is geldig ✅

3. **Meerdere crossovers (2+)**:
   - **Thermorheologisch complex** ⚠️
   - Mogelijke oorzaken voor TPU:
     - Hard-segment kristallisatie tijdens meting
     - Bi-modale molecuulgewichtsverdeling
     - Incomplete menging van soft/hard segmenten
     - Fase-scheiding tijdens afkoelen

**Crossover Frequentie (ω_co):**
```
Karakteristieke relaxatietijd: τ_co ≈ 1/ω_co

Interpretatie:
- ω_co = 1 rad/s → τ = 1 seconde (typisch TPU)
- ω_co = 100 rad/s → τ = 0.01 s (snelle relaxatie, laag Mw)
- ω_co = 0.01 rad/s → τ = 100 s (trage relaxatie, hoog Mw)
```

---

### 5. Validatie Technieken

#### 5.1 Van Gurp-Palmen Plot

**Plot:**
```
Y-as: δ (fasehoek) [°]
X-as: |G*| (complexe modulus) [Pa] (log-schaal)
```

**Principe:**
Als het materiaal thermorheologisch simpel is, moeten **alle temperatuurcurves samenvallen** in deze plot.

**Interpretatie:**

1. **Superpositie (curves vallen samen)**:
   - ✅ Thermorheologisch simpel
   - ✅ TTS is geldig
   - ✅ WLF/Arrhenius betrouwbaar
   - Materiaal = **homogene smelt**

2. **Spreiding (curves wijken af)**:
   - ⚠️ Thermorheologisch complex
   - ⚠️ TTS met voorzichtigheid gebruiken
   - **Voor TPU typisch onder T_softening**
   - Oorzaak: Hard-segment domeinen smelten niet uniform

3. **"Bult" (curve duikt omlaag bij lage |G*|)**:
   - Elastisch netwerk aanwezig
   - Onvolledige smelt
   - Mogelijke crosslinking

**Waarom werkt deze plot?**
```
δ(|G*|) is een "intrinsieke" eigenschap

Als structuur constant blijft bij T-verandering:
→ δ is alleen functie van |G*|, niet van T
→ Curves vallen samen

Als structuur verandert met T:
→ δ hangt af van zowel |G*| als T
→ Curves spreiden
```

---

#### 5.2 Han Plot

**Plot:**
```
Y-as: G' [Pa] (log)
X-as: G" [Pa] (log)
```

**Principe:**
Voor thermorheologisch simpele materialen:
```
log(G') = f(log(G"))

De functie f is onafhankelijk van T!
```

**Interpretatie:**

1. **Curves vallen samen**:
   - ✅ TTS geldig
   - Structuur is temperatuur-onafhankelijk

2. **Verticale verschuiving bij hogere T**:
   - ⚠️ **Thermal crosslinking** (voor TPU)
   - Oorzaak: Na-reactie van NCO groepen tijdens meting
   - **Actie**: Reduceer meettemperatuur of meettijd!

3. **Horizontale verschuiving**:
   - Mogelijk: verandering in viscositeit zonder structuurwijziging
   - Minder alarmerend dan verticale shift

**Voor TPU specifiek:**
```
Opwaartse shift bij hogere T = GEVAAR!
→ Materiaal crosslinkt tijdens de meting
→ Data is niet representatief voor het werkelijke materiaal
```

---

#### 5.3 Cole-Cole Plot

**Plot:**
```
Y-as: η" = G"/ω [Pa·s]
X-as: η' = G'/ω [Pa·s]
```

**Principe:**
Voor een **monodisperse** polymeer (smalle Mw verdeling):
```
Cole-Cole plot = perfecte halve cirkel
```

**Interpretatie:**

1. **Halve cirkel**:
   - Smalle molecuulgewichtsverdeling (MWD)
   - Uniform polymeer

2. **Afgeplatte boog**:
   - **Brede MWD**
   - Typisch voor commerciële TPU
   - Mengsel van verschillende ketenlengte

3. **Asymmetrische vorm**:
   - Bi-modale verdeling
   - Twee populaties van ketenlengtes

**Relatie met procesgedrag:**
```
Brede MWD:
+ Betere verwerkbaarheid (melt strength)
+ Bredere procesvenster
- Minder homogene eigenschappen
- Meer lot-to-lot variatie
```

---

#### 5.4 R² en Adjusted R²

**R² (Determination Coefficient):**
```
R² = 1 - (SS_res / SS_tot)

Waarbij:
SS_res = Σ(y_data - y_fit)²  (residual sum of squares)
SS_tot = Σ(y_data - ȳ)²      (total sum of squares)
```

**Implementatie:**
```python
residuals = log_aT_data - log_aT_arrhenius
ss_res = np.sum(residuals**2)
ss_tot = np.sum((log_aT_data - np.mean(log_aT_data))**2)
r2 = 1 - ss_res / ss_tot
```

**Probleem met R² bij weinig data:**
```
Bij 3-4 punten kan R² = 0.999 zijn, zelfs met slecht model!
→ Overfitting
```

**Adjusted R² (Gecorrigeerd):**
```
R²_adj = 1 - (1 - R²) · (n - 1) / (n - p - 1)

Waarbij:
n = aantal datapunten
p = aantal parameters in model (hier: 1 voor Arrhenius)
```

**Implementatie:**
```python
n_points = len(log_aT_data)
p = 1  # Arrhenius heeft 1 parameter (slope)
r2_adj = 1 - (1 - r2) * (n_points - 1) / max(n_points - p - 1, 1)
```

**Interpretatie:**
- **R²_adj > 0.98**: Uitstekende fit, model zeer betrouwbaar
- **R²_adj 0.90-0.98**: Acceptabel, maar check residuals
- **R²_adj < 0.90**: Slechte fit → model past niet bij data
  - Voor TTS: materiaal is thermorheologisch complex
  - Overweeg andere referentietemperatuur

**Verschil R² vs R²_adj:**
```
Bij n = 4 punten:
R² = 0.995 → Lijkt perfect!
R²_adj = 0.985 → Nog steeds goed, maar realistischer

Bij n = 10 punten:
R² = 0.995
R²_adj = 0.994 → Verschil kleiner bij meer data
```

---

## 📊 Tab-per-Tab Uitleg

### Tab 1: Master Curve

**Doel:** Constructie van de Time-Temperature Superposition master curve.

**Visuele Elementen:**

1. **Hoofdplot (links)**:
   - **G' (volle lijnen, cirkels)**: Elastische modulus
   - **G" (gestippelde lijnen, kruisjes)**: Visceuze modulus
   - **X-as**: ω·aT (verschoven frequentie) [rad/s], log-schaal
   - **Y-as**: Modulus [Pa], log-schaal
   - **Kleuren**: Gradiënt volgens gekozen colormap (elke kleur = temperatuur)

2. **Shift Factor Trend (rechts)**:
   - **X-as**: Temperatuur [°C]
   - **Y-as**: log(aT)
   - **Interpretatie**:
     - Lineair → Arrhenius gedrag (activatie-energie gedomineerd)
     - Gebogen → WLF gedrag (vrije volume gedomineerd)

3. **Smooth Export Sectie**:
   - **Spline smoothing** van η* curve
   - **Slider**: Smoothing sterkte (0.0 = geen, 2.0 = maximaal)
   - **Gebruik**: Voor publicatie-kwaliteit grafieken
   - **Let op**: Te veel smoothing kan echte features verbergen!

**Workflow:**
```
1. Upload data
2. Selecteer temperaturen
3. Kies referentietemperatuur (typisch: hoogste T voor TPU)
4. Klik "🚀 Auto-Align" voor automatische optimalisatie
   OF
   Pas handmatig aan met sliders voor fijnafstemming
5. Controleer overlap van curves
6. Export smoothed data indien gewenst
```

---

### Tab 2: Van Gurp-Palmen Structuuranalyse

**Doel:** Validatie van thermorheologische eenvoud.

**Plot Details:**
- **X-as**: |G*| = √(G'² + G"²) [Pa], log-schaal
- **Y-as**: δ = arctan(G"/G') [°], lineair 0-95°
- **Elke kleur**: Verschillende meettemperatuur

**Diagnostische Criteria:**

| Observatie | Betekenis | Actie |
|------------|-----------|-------|
| Perfecte overlap | Thermorheologisch simpel | ✅ TTS betrouwbaar |
| Kleine spreiding | Licht complex | ⚠️ Gebruik TTS met voorzichtigheid |
| Duidelijke "trappen" | Fase-overgangen | ❌ TTS onbetrouwbaar onder T_softening |
| Bult bij lage \|G*\| | Elastisch netwerk | ⚠️ Onvolledige smelt |

**TPU-specifieke Interpretatie:**
```
Spreiding tussen curves:
→ Harde segmenten smelten niet uniform
→ Elke temperatuur heeft andere morfologie
→ Dit verklaart "negatieve" WLF C1 waarden
→ Kies hogere referentietemperatuur!
```

---

### Tab 3: tan δ Analyse

**Doel:** Visualisatie van relaxatie-spectrum.

**Plot Details:**
- **X-as**: ω [rad/s], log-schaal
- **Y-as**: tan(δ) = G"/G', lineair
- **Rode lijn bij y=1**: Crossover indicator (G' = G")

**Interpretatie Peaks:**
```
Peak positie (ω_peak):
→ Karakteristieke relaxatietijd: τ = 1/ω_peak

Peak hoogte:
→ Sterkte van relaxatie-proces

Voor TPU:
- Lage ω peak: Relaxatie van zachte segmenten
- Hoge ω peak: Lokale bewegingen (β-relaxatie)
- Verschuiving met T: Temperatuurafhankelijkheid mobiliteit
```

**Praktische Relevantie:**
```
tan(δ) bij 1 Hz (typische verwerkingsfrequentie):
- tan(δ) < 0.3: Elastisch, vorm-stabiel (goed voor coatings)
- tan(δ) 0.3-1.0: Visco-elastisch balans
- tan(δ > 1.0: Vloeit gemakkelijk (goed voor injectie)
```

---

### Tab 4: Thermische Karakterisatie (Ea/WLF/VFT)

**Doel:** Bepaling van temperatuurafhankelijkheid via drie complementaire modellen.

**Hoofdplot:**
- **Zwarte punten**: Experimentele shift factors (uit sliders)
- **Rode streepjeslijn**: Arrhenius model (lineair in 1/T)
- **Blauwe lijn**: WLF model (gebogen vorm)
- **Groene stippellijn**: VFT model (indien fit succesvol)
- **Oranje verticale lijn**: Geschat softening point

**Metrics (rechter kolom):**

1. **Ea (Activatie Energie)**:
   - Typisch: 50-150 kJ/mol voor TPU
   - Hoe hoger, hoe gevoeliger voor T-variaties

2. **Softening Point**:
   - Temperatuur waar Arrhenius ≈ WLF
   - **Kritisch**: Ref temp moet **boven** deze waarde!
   - Anders: TTS in thermorheologisch complex gebied

3. **VFT T₀ (Vogel Temperatuur)**:
   - Theoretische "freeze" temperatuur
   - Typisch: -80°C tot -50°C voor TPU
   - Tg ≈ T₀ + 50K

4. **WLF C1 & C2**:
   - C1 tussen 8-17 = normaal
   - C2 ≈ 50K = universele waarde
   - C2 ≈ (T_ref - Tg)

**Validatie Checks:**

```python
if ref_temp < T_softening:
    ⚠️ KRITIEKE WAARSCHUWING
    Master Curve is wiskundig, niet fysisch correct!
    → Harde segmenten nog niet volledig gesmolten
    → TTS approximatie, geen echte superpositie
```

**Model Selectie:**
```
Gebruik Arrhenius als:
- T > T_softening + 30K
- Materiaal is dunne smelt
- R²_adj > 0.98

Gebruik WLF als:
- T_softening - 20K < T < T_softening + 50K
- Elastische eigenschappen belangrijk
- C1 en C2 binnen normale ranges

Gebruik VFT als:
- Breed temperatuurbereik
- Beide regimes aanwezig
- Fit convergeert stabiel
```

---

### Tab 5: TTS Validatie

**Doel:** Visuele controle of TTS aanname geldig is.

#### 5.1 Han Plot (links)

**Principe:**
```
Als log(G') alleen afhankelijk is van log(G"):
→ Alle T-curves vallen samen
→ TTS is geldig
```

**Checklist:**
- ✅ Curves overlappen: TTS geldig
- ⚠️ Curves paralel verschoven: Mogelijk, maar check oorzaak
- ❌ Curves spreiden verticaal: **Thermal crosslinking!**

**TPU Specifiek:**
```
Opwaartse shift bij hogere T:
→ NCO groepen reageren na tijdens meting
→ Materiaal wordt stijver dan verwacht
→ Reduceer meettemperatuur/tijd
```

#### 5.2 Cole-Cole Plot (rechts)

**Principe:**
```
η' vs η" plot toont molecuulgewichtsverdeling:
- Halve cirkel = monodispers
- Afgeplatte boog = polydispers
```

**Interpretatie voor TPU:**
```
Typisch: Afgeplatte boog
→ Commerciële TPU heeft brede MWD
→ Dit is normaal en acceptabel
→ Geeft betere verwerkbaarheid

Als vorm verandert met T:
→ Mogelijk degradatie of na-reactie
```

#### 5.3 R² Kwaliteitscontrole

**Thresholds:**
- **R² > 0.98**: ✅ Uitstekend
- **R² 0.90-0.98**: ⚠️ Matig (controleer vGP plot)
- **R² < 0.90**: ❌ Slecht (TTS niet geldig)

---

### Tab 6: Moleculaire Analyse

**Doel:** Kwantificatie van moleculaire parameters relevant voor verwerking.

#### 6.1 Key Metrics

**η₀ (Zero Shear Viscosity):**
```
Proces-indicator:
- Coating: Te laag → druipen (sagging)
- Extrusion: Te hoog → hoge drukopbouw
- Injection: Sweet spot afhankelijk van geometrie

Hydrolyse detector:
η₀_nieuw / η₀_oud = 1.5
→ ~13% verlies in Mw (door ketenbreuk)
```

**Gₙ⁰ (Plateau Modulus):**
```
Netwerk-dichtheid indicator:
G_N⁰ ∝ kT/M_e

Waarbij M_e = entanglement Mw

Hoge G_N⁰:
+ Goede melt strength
+ Vorm-stabiel
- Moeilijker te verwerken
```

#### 6.2 Cross Model Visualisatie

**Plot:**
- **Zwarte punten**: η*(ω) meetdata
- **Rode lijn**: Cross model fit
- **Rode stippellijn**: Geëxtrapoleerde η₀

**Fit Kwaliteit:**
```
Goede fit:
→ Rode lijn volgt data over gehele bereik
→ η₀ extrapolatie is betrouwbaar

Slechte fit:
→ Rode lijn wijkt af bij lage ω
→ η₀ is onzeker (data niet tot terminal zone)
→ Waarschuwing wordt getoond
```

**τ (Relaxatietijd):**
```
τ = karakteristieke tijd voor keten-ontwarring

Typisch: 0.1 - 10 seconden voor TPU

Proces-implicatie:
τ < verwerkingstijd: Materiaal kan volledig relaxeren
τ > verwerkingstijd: Interne spanningen blijven
```

---

### Tab 7: Expert Dashboard

**Doel:** Geconsolideerd overzicht + automatische validatie.

#### 7.1 KPI Metrics (Bovenaan)

Vier kritieke parameters in één oogopslag:

1. **Ea**: T-gevoeligheid (procescontrole)
2. **η₀**: Verwerkbaarheid + degradatie
3. **Adj. R²**: Betrouwbaarheid TTS
4. **Crossovers**: Complexiteit indicator

#### 7.2 Parameter Tabel

Volledige lijst met **12 parameters** verdeeld over 4 categorieën:

**Thermisch (5):**
- Ea: Activatie energie
- WLF C₁, C₂: Vrije volume parameters
- VFT T∞: Vogel temperatuur
- Geschatte Tg: T∞ + 50K regel

**Viscositeit (2):**
- η₀: Zero-shear viscosity
- τ: Relaxatietijd

**Structuur (3):**
- Terminal Slope: Vloeigedrag
- Gₙ⁰: Plateau modulus
- Crossovers: Aantal kruispunten

**Validatie (2):**
- R²: Raw fit kwaliteit
- R²_adj: Gecorrigeerde fit

#### 7.3 Model Betrouwbaarheid (2 kolommen)

**Linker kolom - Thermische Modellen:**

```python
WLF Validatie:
if C1 < 0 or C2 < 0:
    ❌ Fysisch onmogelijk
elif C1 < 5 or C1 > 30:
    ⚠️ Atypisch (mogelijk complex)
else:
    ✅ Stabiel

Arrhenius:
if R²_adj > 0.98:
    ✅ Uitstekend
elif R²_adj > 0.90:
    ℹ️ Acceptabel
else:
    ⚠️ Zwak

VFT/Tg:
if VFT succesvol:
    Tg_est = T∞ + 50K
    if Tg_est > ref_temp:
        ⚠️ Onmogelijk (check data!)
```

**Rechter kolom - Structuur:**

```python
Terminal Slope:
if slope < 1.5:
    ❌ Vloeiprobleem (onvolledige smelt/crosslinking)
elif slope < 1.8:
    ⚠️ Afwijkend (lichte belemmering)
else:
    ✅ Newtoniaans

Crossovers:
if n_crossovers == 0:
    ⚠️ Geen crossover (extreem elastisch/visceus)
elif n_crossovers == 1:
    ✅ Enkelvoudig (simpel gedrag)
else:
    ❌ Meervoudig (complex, check vGP!)

Hydrolyse Check:
η₀ als referentie voor toekomstige batches
```

#### 7.4 Crossover Tabel

Lijst van crossover punten **per temperatuur**:

| T (°C) | ω_crossover (rad/s) | G=G" (Pa) | Aantal |
|--------|---------------------|-----------|--------|
| ...    | ...                 | ...       | ...    |

**Interpretatie "Aantal" kolom:**
- **1**: Normaal gedrag
- **2+**: Fase-heterogeniteit → Waarschuwing met mogelijke oorzaken

#### 7.5 Data Export (4 knoppen)

1. **Parameters CSV**: Dashboard tabel (12 parameters)
2. **Shift Factors CSV**: T, log(aT), aT voor elke temperatuur
3. **Crossovers CSV**: Crossover data
4. **Master Curve CSV**: Volledige dataset met extra kolommen:
   - omega_shifted_rad_s
   - Gp_Pa, Gpp_Pa
   - Complex_Visc_Pas
   - PhaseAngle_deg
   - tan_delta
   - G_star_Pa
   - Original_T_C

---

## 🚀 Installatie & Gebruik

### Vereisten

```bash
# Python 3.8+
pip install streamlit pandas numpy matplotlib scipy
```

**Dependencies:**
- `streamlit >= 1.28.0`
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `matplotlib >= 3.7.0`
- `scipy >= 1.10.0`

### Starten van de App

```bash
streamlit run RAT.py
```

De app opent automatisch in je browser op `http://localhost:8501`

### Workflow

```
1. UPLOAD
   ├─ Klik "Browse files" in sidebar
   ├─ Selecteer CSV/TXT bestand met frequency sweep data
   └─ Controleer of sample naam correct geladen is

2. CONFIGURATIE
   ├─ Selecteer temperaturen voor analyse
   ├─ Kies referentietemperatuur (advies: hoogste T)
   ├─ Pas colormap aan naar voorkeur
   └─ Vul verwachte Tg in (voor WLF hint)

3. ALIGNMENT
   ├─ Optie A: Klik "🚀 Auto-Align" voor automatisch
   └─ Optie B: Pas sliders handmatig aan

4. ANALYSE
   ├─ Tab 1: Controleer overlap master curve
   ├─ Tab 2: Check Van Gurp-Palmen voor complexiteit
   ├─ Tab 3: Bekijk tan δ relaxaties
   ├─ Tab 4: Analyseer thermische modellen
   ├─ Tab 5: Valideer TTS aannames
   ├─ Tab 6: Extractie moleculaire parameters
   └─ Tab 7: Review dashboard + export

5. EXPORT
   └─ Download CSV's via dashboard buttons
```

---

## 📄 Data Format Specificaties

### Input Bestandsformaat

**Ondersteunde encodings:**
- UTF-16 LE (met BOM: `0xFF 0xFE`)
- UTF-8 met BOM (`0xEF 0xBB 0xBF`)
- Latin-1 (ISO-8859-1)
- UTF-8 (fallback)

### Verwachte Structuur

```
Regel 1: <Header info>
Regel 2: <Metadata>
Regel 3: <Tab-separated: Sample Name in kolom 2>
...
Regel N: Interval data: <Header lijn met kolommen>
         ├─ MOET bevatten: "Point No."
         ├─ MOET bevatten: "Storage Modulus"
         └─ MOET bevatten: "Temperature"
Regel N+3: <Start van data rijen>
```

**Vereiste Kolommen:**
- `Temperature` → wordt `T` [°C]
- `Angular Frequency` → wordt `omega` [rad/s]
- `Storage Modulus` → wordt `Gp` [Pa]
- `Loss Modulus` → wordt `Gpp` [Pa]

**Data Cleaning:**
```python
# Automatisch toegepast:
- Decimaal komma → punt conversie
- Verwijdering van lege rijen
- Filter: Gp > 0 en omega > 0
- NaN verwijdering
```

### Voorbeeld Data Fragment

```
<bestandskop>
Sample ID	TPU_Sample_001
Date	2024-02-08
Interval data:	Point No.	Temperature	Angular Frequency	Storage Modulus	Loss Modulus
		1	150	0.1	1000	500
		2	150	1.0	5000	2000
		3	150	10.0	20000	8000
		...
```

---

## 🔧 Troubleshooting

### Veelvoorkomende Problemen

#### 1. "Geen data gevonden in het bestand"

**Oorzaak:**
- Bestand mist "Interval data:" header
- Verkeerde tab-delimiters
- Encoding probleem

**Oplossing:**
```python
# Check handmatig:
1. Open bestand in teksteditor
2. Controleer of tabs (\t) aanwezig zijn, niet spaties
3. Zoek naar "Interval data:" regel
4. Check of "Storage Modulus" in header staat
```

#### 2. "VFT fit niet succesvol"

**Oorzaak:**
- Te weinig datapunten (n < 4)
- Te smalle temperatuurrange
- T₀ te dicht bij meettemperaturen

**Oplossing:**
```python
# Check:
1. Heb je minimaal 4 verschillende temperaturen?
2. Is T_range > 30°C?
3. Pas Tg_hint aan (dichter bij werkelijke Tg)
4. VFT is optioneel - gebruik WLF als alternatief
```

#### 3. "Terminal Slope = N/A"

**Oorzaak:**
- Geen punten met δ > 75° in lage frequentie bereik
- Meetrange stopt te vroeg

**Oplossing:**
```python
# Check data:
1. Zijn er punten met omega < 1 rad/s?
2. Is G" > G' bij lage frequenties?
3. Overweeg: Meet bij lagere frequenties (indien mogelijk)
```

#### 4. "Negatieve WLF C1 waarde"

**Oorzaak:**
- Thermorheologisch complex materiaal
- Referentietemperatuur in fase-overgangszone
- Shift factors volgen geen WLF gedrag

**Oplossing:**
```python
# Acties:
1. Check Van Gurp-Palmen plot (Tab 2)
2. Kies hogere referentietemperatuur
3. Gebruik Arrhenius i.p.v. WLF
4. Accepteer dat materiaal complex is
```

#### 5. "Master Curve overlapt niet goed"

**Oorzaak:**
- Handmatige shifts onnauwkeurig
- Auto-align convergeert naar lokaal minimum
- Materiaal is inherent thermorheologisch complex

**Oplossing:**
```python
# Workflow:
1. Klik "🔄 Reset" om opnieuw te beginnen
2. Probeer andere referentietemperatuur
3. Start met "🚀 Auto-Align"
4. Fijnafstemming met sliders
5. Als blijft falen: Check vGP plot → mogelijk complex
```

#### 6. "η₀ extrapolatie mislukt"

**Oorzaak:**
- Te weinig punten in terminal zone
- Cross model fit divergeert
- Data is te ruis

**Oplossing:**
```python
# Check:
1. Zijn er punten met omega < 0.1 rad/s?
2. Is er duidelijke Newtoniaanse plateau bij lage ω?
3. Overweeg: Gebruik G'/ω i.p.v. Cross model
4. Verhoog smoothing factor in Tab 1
```

---

## 📚 Referenties

### Wetenschappelijke Literatuur

1. **Time-Temperature Superposition:**
   - Ferry, J.D. (1980). *Viscoelastic Properties of Polymers*, 3rd Ed. Wiley.
   - Dealy, J.M., & Plazek, D.J. (2009). "Time-temperature superposition—a users guide." *Rheology Bulletin*, 78(2), 16-31.

2. **WLF Vergelijking:**
   - Williams, M.L., Landel, R.F., & Ferry, J.D. (1955). "The temperature dependence of relaxation mechanisms in amorphous polymers and other glass-forming liquids." *J. Am. Chem. Soc.*, 77(14), 3701-3707.

3. **Van Gurp-Palmen Plot:**
   - Van Gurp, M., & Palmen, J. (1998). "Time-temperature superposition for polymeric blends." *Rheol. Bull.*, 67(1), 5-8.
   - Trinkle, S., & Friedrich, C. (2001). "Van Gurp-Palmen-plot: a way to characterize polydispersity of linear polymers." *Rheol. Acta*, 40(4), 322-328.

4. **TPU Rheologie:**
   - Kojio, K., et al. (2020). "Effect of hard segment content on the mechanical and thermal properties of polyurethane elastomers." *Polymer*, 206, 122864.
   - Prisacariu, C. (2011). *Polyurethane Elastomers: From Morphology to Mechanical Aspects*. Springer.

5. **Cross Model:**
   - Cross, M.M. (1965). "Rheology of non-Newtonian fluids: A new flow equation for pseudoplastic systems." *J. Colloid Sci.*, 20(5), 417-437.

6. **Cole-Cole & Han Plots:**
   - Han, C.D., & Kim, J. (1993). "On the use of time-temperature superposition in multicomponent/multiphase polymer systems." *Polymer*, 34(12), 2533-2539.

### Online Resources

- **Rheology Basics:** [www.rheologyschool.com](https://www.rheologyschool.com)
- **TA Instruments Application Notes:** [www.tainstruments.com](https://www.tainstruments.com)
- **Anton Paar Webinars:** [www.anton-paar.com/rheology](https://www.anton-paar.com)

---

## 🏆 Best Practices

### Voor Optimale TTS Resultaten

1. **Meetstrategie:**
   ```
   ✅ Gebruik minimaal 5 temperaturen
   ✅ Span minimaal 40°C range
   ✅ Meet bij hoogste T eerst (om thermische geschiedenis te resetten)
   ✅ Wacht 5 min equilibratie bij elke T
   ✅ Check lineariteit (strain sweep eerst!)
   ```

2. **Referentietemperatuur Keuze:**
   ```
   ✅ Kies T > T_softening + 20°C
   ✅ Bij twijfel: Hoogste meettemperatuur
   ❌ NIET: Laagste temperatuur (thermisch complex)
   ❌ NIET: Midden bereik (tenzij goede reden)
   ```

3. **Validatie Checklist:**
   ```
   ✅ Van Gurp-Palmen: Curves overlappen?
   ✅ Han Plot: Geen verticale shift?
   ✅ R²_adj > 0.95?
   ✅ Terminal slope ≈ 2.0?
   ✅ WLF C1 tussen 8-17?
   ```

4. **Voor TPU Specifiek:**
   ```
   ✅ Meet boven Tm van harde segmenten (typ. >180°C)
   ✅ Gebruik N2 purge (voorkom oxidatie)
   ✅ Check reproductie bij één T (stabiliteit)
   ❌ Vermijd lange meettijden bij hoge T (na-reactie!)
   ```

### Interpretatie van Resultaten

**Scenario 1: Perfect TTS**
```
✅ vGP: Superpositie
✅ Han: Overlap
✅ R²_adj > 0.98
✅ 1 crossover
✅ Slope ≈ 2.0

→ Materiaal is homogene smelt
→ Alle parameters betrouwbaar
→ Voorspellingen geldig buiten meetbereik
```

**Scenario 2: Licht Complex**
```
⚠️ vGP: Kleine spreiding
⚠️ R²_adj = 0.92-0.97
✅ 1 crossover
✅ Slope ≈ 1.9

→ Acceptabel voor praktische doeleinden
→ Parameters bruikbaar met voorbehoud
→ NIET extrapoleren ver buiten meetbereik
```

**Scenario 3: Sterk Complex**
```
❌ vGP: Duidelijke trappen
❌ R²_adj < 0.90
❌ Meerdere crossovers
❌ Slope < 1.7

→ TTS is NIET geldig
→ Kies hogere ref temp OF
→ Accepteer dat materiaal niet-thermorheologisch simpel is
→ Gebruik data ALLEEN bij gemeten temperaturen
```

---

## 💡 Tips & Tricks

### Geavanceerde Gebruikerstechnieken

**1. Identificeer Optimale Procestemperatuur:**
```python
# Zoek waar η* = 1000 Pa·s (typisch voor extrusion)
target_visc = 1000
omega_process = 10  # rad/s (typ. shear rate / plaat afstand)

# Via mastercurve:
1. Vind omega_shifted waar η*(omega_shifted) = target_visc
2. Bereken: omega_actual = omega_shifted / aT
3. Los aT op voor gewenste T via WLF/Arrhenius
```

**2. Schat Molecuulgewicht Verandering:**
```python
# Na batch-to-batch vergelijking
delta_eta0 = (eta0_new - eta0_old) / eta0_old

# Voor lineaire polymeren: η₀ ∝ M_w^3.4
delta_Mw = (1 + delta_eta0)^(1/3.4) - 1

# Voorbeeld:
# eta0: 1e5 → 1.2e5 (+20%)
# Mw toename ≈ (1.20)^0.294 - 1 ≈ 5.5%
```

**3. Detecteer Crosslinking:**
```python
# In Han Plot:
if G'(T_high) > G'(T_low) bij zelfde G":
    → Verticale opwaartse shift
    → Thermal crosslinking!
    
# Actie:
- Reduceer max T met 10-20°C
- Verkort meettijd
- Check NCO index van materiaal
```

**4. Optimaliseer Shift Factors Handmatig:**
```python
# Workflow:
1. Start met auto-align
2. Focus op overlap bij LAGE omega (terminal zone)
   → Dit gebied bepaalt η₀ en vloeigedrag
3. Laat hoge omega iets minder perfect overlappen
   → Hoogfrequent gedrag is minder procesrelevant
4. Check δ in vGP plot als final validation
```

**5. Export voor Simulatie Software:**
```python
# Master Curve CSV bevat:
- omega_shifted: voor FEM input
- Complex_Visc: voor Moldflow
- Gp, Gpp: voor Ansys Polyflow
- tan_delta: voor demping berekeningen

# Importeer direct in:
- ANSYS Fluent (UDF via η*(ω))
- COMSOL (interpolation table)
- Moldflow (Cross-WLF fit parameters)
```

---

## ⚙️ Technische Specificaties

### Algoritme Parameters

**Auto-Align Optimalisatie:**
```python
Methode: Nelder-Mead simplex
Tolerance: ftol=1e-6, xtol=1e-6
Max iterations: 1000
Initiële guess: log(aT) = 0.0
Bounds: [-10, 10] (aT tussen 1e-10 en 1e10)
```

**Cross Model Fit:**
```python
Methode: Trust Region Reflective
Max function calls: 5000
Initial guess: [max(η), 0.1, 0.8]
Bounds: Geen (unbounded)
Loss: Least squares in linear space
```

**WLF Optimalisatie:**
```python
Methode: L-BFGS-B
Bounds: C1 ∈ [1, 50], C2 ∈ [10, 200]
Initial: C1=17.4, C2=max(50, T_ref - Tg_hint)
Convergence: gtol=1e-8
```

**VFT Fit:**
```python
Methode: Levenberg-Marquardt (via curve_fit)
Bounds: 
  A ∈ [-∞, ∞]
  B ∈ [10, 5000]
  T0 ∈ [50K, min(T_data) - 5K]
Max iterations: 10000
```

### Performance Benchmarks

**Typische Verwerkingstijden:**
```
Data laden (1000 punten): < 1s
Auto-align (5 temps): 2-5s
Cross model fit: < 0.5s
VFT fit: 0.5-2s (afhankelijk van convergentie)
Plot rendering: 0.2-0.5s per plot
CSV export: < 0.1s
```

**Geheugenvereisten:**
```
App baseline: ~150 MB
Per 1000 datapunten: +5 MB
Alle tabs geladen: ~300 MB totaal
```

---

## ⚠️ BELANGRIJKE DISCLAIMER

**Ontwikkelingsproces:**
Deze tool is ontwikkeld tijdens een weekend "vibe coding" sessie met behulp van AI-assistenten (Claude.ai en Gemini). Het is een **proof-of-concept** en **geen gevalideerd commercieel product**.

**Geen Garanties:**
- ❌ De maker kan **NIET instaan voor de correctheid** van berekeningen
- ❌ Er is **GEEN formele validatie** uitgevoerd tegen industriestandaarden
- ❌ De tool is **NIET bedoeld voor kritische beslissingen** zonder onafhankelijke verificatie
- ❌ Gebruikers zijn **zelf verantwoordelijk** voor validatie van resultaten

**Gebruik op Eigen Risico:**
```
⚠️ CONTROLEER ALTIJD DE RESULTATEN MET:
   - Handberekeningen op testdata
   - Vergelijking met gevalideerde commerciële software
   - Expert review van gevonden parameters
   - Fysische validatie (bijv. viscositeitsmetingen bij specifieke T)
```

**Bekende Beperkingen:**
- Geen error propagation analysis
- Geen uncertainty quantification
- Beperkte input validatie
- Geen unit testing framework
- Geen peer-review van algoritmes

**Aanbevolen Gebruik:**
```
✅ Exploratory data analysis
✅ Snelle eerste beoordeling van TTS geldigheid  
✅ Educatieve doeleinden
✅ Basis voor verdere ontwikkeling
❌ Production quality control
❌ Regulatoire documentatie
❌ Kritische productontwikkeling
```

**Disclaimer Tekst voor Rapporten:**
```
"Analyse uitgevoerd met open-source tool RAT (v1.0, niet-gevalideerd).
Resultaten zijn indicatief en dienen geverifieerd te worden met gevalideerde
methoden voorafgaand aan gebruik in kritische beslissingen."
```

---

## 🔐 Licentie & Gebruik

**Ontwikkeld voor:** TPU Rheologie Analyse  
**Versie:** 1.0.0 (Alpha - Experimenteel)  
**Laatst bijgewerkt:** 2026-02-08  
**Ontwikkeld met:** Claude.ai & Gemini (AI-assisted coding)  

**Gebruik:**
Deze tool is ontwikkeld voor wetenschappelijk en industrieel **experimenteel** gebruik. Voor publicaties die gebruik maken van deze tool, wordt gevraagd te refereren aan de onderliggende wetenschappelijke principes (zie Referenties sectie) en de experimentele aard van de software te vermelden.

**Data Kwaliteit Checklist:**
De resultaten van deze tool zijn zo betrouwbaar als de input data. Controleer altijd:
- ✅ Lineariteit van de meting (LVE regime)
- ✅ Thermische equilibratie (5+ min wachttijd per T)
- ✅ Sample integriteit (geen degradatie, oxidatie, vocht)
- ✅ Instrumentcalibratie (recent, binnen spec)
- ✅ Reproduceerbaarheid (minimaal duplicaat metingen)

---

## 📧 Support & Contact

**Voor vragen over:**
- Theoretische achtergrond → Raadpleeg Ferry (1980) of contacteer polymer rheologie expert
- Technische bugs → Open issue in repository
- Feature requests → Pull requests welkom

**Veelgestelde Vragen:**
Zie `TROUBLESHOOTING.md` (indien beschikbaar) of de Troubleshooting sectie hierboven.

---

**Happy Rheology! 🧪📊**

*"In God we trust, all others must bring data." - W. Edwards Deming*