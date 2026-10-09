# Planned Controlled Learning-Effectiveness Study Protocol

**Status:** PROPOSED FUTURE RESEARCH STUDY (NOT YET CONDUCTED)  
**Document Version:** 1.0  
**Phase:** Phase 9 Preparation for Empirical Educational Trials  
**Project:** A Dynamic Personalized Knowledge Map for Engineering Students using Indic Language NLP  

---

> [!IMPORTANT]
> **Scientific Position & Critical Disclaimer:**  
> This document specifies the experimental design, sample size justification, statistical power analysis, and measurement instruments for a **future controlled clinical learning-effectiveness trial**.  
> **IndicLearn has NOT yet conducted this study.**  
> Neither this document nor the software codebase claims that IndicLearn currently improves student learning, boosts exam scores, reduces cognitive load, or increases knowledge retention. Those claims require executing this controlled trial with real cohorts under formal Institutional Review Board (IRB) ethical oversight.

---

## 1. Research Hypotheses

* **Primary Hypothesis ($H_1$):** Undergraduate engineering students using the multilingual, personalized IndicLearn system will demonstrate significantly higher normalized learning gains from pre-test to post-test than students using static, non-adaptive textbook materials.
* **Secondary Hypothesis ($H_2$):** Students with lower prior English academic fluency will exhibit greater relative achievement gains when accessing explanations in their preferred language modality (Telugu or Code-Mixed) compared to an English-only control baseline.
* **Retention Hypothesis ($H_3$):** Concept mastery estimates updated via closed-loop adaptive practice in IndicLearn will correlate positively with retention scores measured on an unannounced delayed post-test administered 14 days post-intervention.

---

## 2. Experimental Design

A two-arm between-subjects randomized controlled trial (RCT) with pre-test, intervention, immediate post-test, and delayed post-test:

```text
                  [Eligible Engineering Students (N=152 Enrolled)]
                                    │
                         [Baseline Pre-Test (T0)]
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼
           [Control Arm (n=76)]            [Treatment Arm (n=76)]
         Static Textbook Chunks &          IndicLearn Interactive Map,
         Standard Reference Notes          Adaptive Quiz & Indic RAG
                   │                                 │
                   └────────────────┬────────────────┘
                                    │
                       [Immediate Post-Test (T1)]
                                    │
                    (14-Day Retention Gap: ~15% Attrition)
                                    │
                    [Delayed Retention (T2: N≥128 Completing)]
```

---

## 3. Intervention Arms

1. **Control Arm ($C$):**
   - Students receive digital textbook chapters and reference notes for the designated unit (*Signals and Systems: LTI Systems, Convolution, Sampling Theorem*).
   - Non-adaptive: No interactive knowledge map, no adaptive item selection, no multilingual question-answering.
   - Dedicated study duration: 90 minutes.

2. **Treatment Arm ($T$):**
   - Students receive full access to the **IndicLearn** student dashboard.
   - Access to interactive SVG Knowledge Map with dependency visualization.
   - Access to scope-controlled Indic RAG in English, Telugu, Romanized Telugu, and Code-Mixed forms.
   - Access to closed-loop adaptive assessment quizzes with personalized remediation recommendations.
   - Dedicated study duration: 90 minutes.

---

## 4. Outcome Measures & Target Metrics

1. **Normalized Learning Gain ($g$):**
   $$g = \frac{\text{Post-Test Score} - \text{Pre-Test Score}}{\text{Max Possible Score} - \text{Pre-Test Score}}$$
2. **Delayed Retention Loss ($\Delta R$):**
   $$\Delta R = \text{Immediate Post-Test (T1)} - \text{Delayed Post-Test (T2)}$$
3. **Assessment Diagnostic Alignment:**
   Correlation between IndicLearn's latent knowledge estimates ($\hat{S}_k$) and external exam item performance on concept $k$.
4. **Subjective Cognitive Load:**
   NASA-TLX (Task Load Index) administered post-session.

---

## 5. Statistical Power Analysis & Sample Size Justification

* **Study Design & Statistical Test:** Conventional two-arm between-subjects independent two-sample comparison ($H_0: \mu_T = \mu_C$ vs. $H_1: \mu_T \ne \mu_C$) using a two-sided Student's $t$-test with equal allocation ($n_T = n_C = n$).
* **Significance Threshold ($\alpha$):** $\alpha = 0.05$ (two-tailed, critical $t_{0.975} \approx 1.979$).
* **Desired Statistical Power ($1 - \beta$):** $\ge 0.80$ ($\beta = 0.20$).
* **Target Effect Size:** Anticipated medium effect size Cohen's $d = 0.50$ (consistent with personalized intelligent tutoring systems literature, e.g., VanLehn 2011).
* **Mathematical Power Derivation:**
  - Standard Gaussian approximation:
    $$n \approx 2 \times \left( \frac{z_{1 - \alpha/2} + z_{1 - \beta}}{d} \right)^2 = 2 \times \left( \frac{1.95996 + 0.84162}{0.50} \right)^2 = 2 \times (5.60317)^2 \approx 62.79 \implies 64 \text{ per arm}$$
  - Exact non-central Student's $t$-distribution ($\text{df} = 2n - 2$, non-centrality parameter $\delta = d \sqrt{n/2}$):
    - At $n = 60$ ($N = 120$): $\text{df} = 118, \delta = 2.7386 \implies \text{Power} = 0.7753$ (**Underpowered**, fails the conventional $0.80$ threshold).
    - At $n = 63$ ($N = 126$): $\text{df} = 124, \delta = 2.8062 \implies \text{Power} = 0.7952$ ($< 0.80$).
    - At $n = 64$ ($N = 128$): $\text{df} = 126, \delta = 2.8284 \implies \mathbf{\text{Power} = 0.8015}$ ($\ge 0.80$).
  - Thus, a minimum of **$n = 64$ completing participants per group** (total completing sample **$N = 128$**) is mathematically required to guarantee 80% power at $\alpha = 0.05$ (two-sided).
* **Attrition & Retention Gap Adjustment:**
  - Because this study incorporates an unannounced 14-day delayed post-test ($T_2$), longitudinal participant drop-out must be accommodated.
  - Anticipating a conservative **15% attrition rate**:
    $$N_{\text{enroll}} = \frac{N_{\text{complete}}}{1 - 0.15} = \frac{128}{0.85} \approx 150.59 \implies \mathbf{N = 152} \quad (n = 76 \text{ enrolled per arm})$$
  - If attrition reaches **20%**:
    $$N_{\text{enroll}} = \frac{N_{\text{complete}}}{1 - 0.20} = \frac{128}{0.80} = \mathbf{160} \quad (n = 80 \text{ enrolled per arm})$$
  - Therefore, the protocol targets an initial enrollment of **$N = 152$ participants ($n = 76$ per arm)** to guarantee at least **$N = 128$ completing participants ($n = 64$ per arm)** at delayed post-test $T_2$.

---

## 6. Pre-Registered Analysis Plan & Safeguards Against P-Hacking

* **Primary Test:** Independent two-sample Student's $t$-test (or non-parametric Mann-Whitney $U$ test if Shapiro-Wilk rejects normality at $p < 0.05$) comparing normalized learning gain ($g$) between Treatment and Control.
* **Multiple Testing Correction:** All secondary endpoint $p$-values adjusted via the **Holm-Bonferroni step-down procedure** (`src/indiclearn/evaluation/statistics/significance.py`).
* **Subgroup Reporting:** Disaggregated analysis by language preference (English vs Telugu vs Code-Mix) pre-specified before data collection; no post-hoc subset dredging.
