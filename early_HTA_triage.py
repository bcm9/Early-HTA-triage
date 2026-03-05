# early_HTA_triage.py
# Early HTA Triage (Streamlit) working version with keyed tasks + uncertainty
# Run:
#   pip install streamlit pandas numpy matplotlib seaborn
#   streamlit run lihe_hta_triage_app.py

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Early HTA Triage",
    page_icon="🧭",
    layout="wide",
    menu_items={"About": "Quick, structured early HTA triage check."},
)

APP_NAME = "Early Health Technology Assessment Triage"
PRIMARY = "#2B7A78"
ACCENT = "#3AAFA9"
WARNING = "#E29578"
OK = "#84A98C"

sns.set_style("whitegrid")

def pill(text, color):
    st.markdown(
        f"""
        <span style='background:{color};color:white;padding:4px 10px;border-radius:999px;font-size:0.85rem;'>
        {text}
        </span>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Configuration
# -----------------------------
DEVICE_TYPES = ["Hardware", "Software", "AIaMD", "Other"]

STATUS_LEVELS = [
    (0, "Not started"),
    (1, "In progress"),
    (2, "Draft, partial"),
    (3, "Complete & evidenced"),
]
STATUS_LABELS = [f"{n} : {t}" for n, t in STATUS_LEVELS]

# Stable task keys (what logic uses) → question text (what users see)
TASKS = {
    "PICO": "What clinical problem does the technology solve and for whom (population + setting)?",
    "Decision point": "What decision in the care pathway would it influence (e.g. triage, diagnosis, treatment, monitoring)?",
    "Comparator": "What is the current comparator / standard of care?",
    "Clinical pathway mapping": "Where exactly does it fit in the patient care pathway (who does what, when)?",
    "Stakeholder mapping": "Who must adopt/approve it (clinicians, providers, payers, patients)?",
    "Value proposition": "What value does it provide (clinical, operational, economic) and for whom?",
    "Performance data": "Do you have early data showing it works as intended (signal/pilot/usability/performance)?",
    "RCT/RWE evidence": "Do you have clinical evidence (trial, service evaluation, real-world outcomes) aligned to the claim?",
    "Early HE modelling": "Have you estimated potential economic value (simple scenario/threshold, cost offsets, budget impact)?",
    "Uncertainty clarity": "What are the biggest uncertainties (clinical, operational, economic), and which one matters most?",
}

# Nice short labels for plots (display only)
TASK_DISPLAY = {
    "PICO": "Problem",
    "Decision point": "Decision",
    "Comparator": "Comparator",
    "Clinical pathway mapping": "Pathway",
    "Stakeholder mapping": "Stakeholders",
    "Value proposition": "Value",
    "Performance data": "Early data",
    "RCT/RWE evidence": "Evidence",
    "Early HE modelling": "Economics",
    "Uncertainty clarity": "Uncertainty",
}

# Fixed plotting order
TASK_ORDER = list(TASKS.keys())

# Optional lightweight weights (keyed to TASK_ORDER)
DEFAULT_WEIGHTS = {
    "PICO": 1.3,
    "Decision point": 1.3,
    "Comparator": 1.0,
    "Clinical pathway mapping": 1.4,
    "Stakeholder mapping": 0.9,
    "Value proposition": 1.3,
    "Performance data": 1.1,
    "RCT/RWE evidence": 1.1,
    "Early HE modelling": 1.2,
    "Uncertainty clarity": 1.2,
}

def label_for_pct(p):
    if p < 40:
        return "Early"
    if p < 70:
        return "Progressing"
    return "Strong"

def color_for_pct(p):
    if p < 40:
        return WARNING
    if p < 70:
        return ACCENT
    return OK

# Recommendations keyed to task keys (so you never get the generic "Advance..." unless you add new keys)
RECOMMENDATIONS = {
    "PICO": "Lock a decision problem: population, setting, comparator, and the specific claim you want to enable.",
    "Decision point": "Name the exact decision point you change (triage/diagnosis/treatment/monitoring) and who makes that decision.",
    "Comparator": "Clarify what happens today in real practice (not ideal practice) and what you're replacing/augmenting.",
    "Clinical pathway mapping": "Map the real pathway: steps, decision points, who acts, bottlenecks, and where implementation could fail.",
    "Stakeholder mapping": "Identify decision-makers and implementers; capture incentives, blockers, and what evidence each stakeholder needs.",
    "Value proposition": "Turn the pitch into measurable value: health gain, cost offsets, and/or system capacity (time, throughput, waiting lists).",
    "Performance data": "Assemble early evidence aligned to intended use (performance/usability/reliability/safety) with clear endpoints.",
    "RCT/RWE evidence": "Define a minimum viable evidence path (pilot/service eval → RWE → comparative study/RCT only if needed).",
    "Early HE modelling": "Build a simple proto-model: costs, drivers, effect size thresholds that would matter, and plausible scenarios.",
    "Uncertainty clarity": "List uncertainties, pick the single biggest one, and define the cheapest evidence that would reduce it.",
}

def top_next_steps(task_scores, weights, n=4):
    """Pick the highest-priority items: low score + high weight."""
    rows = []
    for key, score in task_scores.items():
        w = float(weights.get(key, 1.0))
        priority = (3 - score) * w  # higher = more urgent
        rows.append((key, score, w, priority))
    rows.sort(key=lambda x: x[3], reverse=True)
    picks = rows[:n]

    steps = []
    for key, score, w, _ in picks:
        msg = RECOMMENDATIONS.get(key)
        if msg:
            steps.append(msg)
        else:
            steps.append(f"Advance '{key}' from {STATUS_LEVELS[score][1].lower()} toward evidence-backed completion.")
    return steps

# -----------------------------
# Charts
# -----------------------------
def make_radar(task_scores):
    keys = TASK_ORDER
    labels = [TASK_DISPLAY[k] for k in keys]
    values = [task_scores.get(k, 0) for k in keys]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + values[:1]
    angles_plot = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(5.6, 5.6), subplot_kw=dict(polar=True))
    color = sns.color_palette("crest", 1)[0]

    ax.plot(angles_plot, values_plot, color=color, linewidth=2.2)
    ax.fill(angles_plot, values_plot, color=color, alpha=0.25)

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0", "1", "2", "3"], fontsize=9, color="gray")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.tick_params(axis="x", pad=10)

    ax.spines["polar"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_title("Early HTA Triage Radar", fontsize=13, weight="semibold", pad=18)

    st.pyplot(fig, use_container_width=False)

def make_bar(task_scores):
    keys = TASK_ORDER
    df = pd.DataFrame(
        {
            "Workstream": [TASK_DISPLAY[k] for k in keys],
            "Score": [task_scores.get(k, 0) for k in keys],
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    palette = sns.color_palette("crest", len(df))
    bars = ax.bar(df["Workstream"], df["Score"], color=palette, edgecolor="none")

    for b, v in zip(bars, df["Score"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 3.4)
    ax.set_ylabel("Rating")
    ax.set_title("Task Status by Workstream", fontsize=13, weight="semibold")
    plt.xticks(rotation=20, ha="right", fontsize=9)

    sns.despine(left=True, bottom=True)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)

    st.pyplot(fig, use_container_width=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.write("Enter info (optional)")

with st.sidebar.expander("Venture info", expanded=True):
    venture_name = st.text_input("Venture", value="Untitled venture")
    team_name = st.text_input("Team / programme", value="(optional)")
    contact = st.text_input("Contact (email or name)", value="(optional)")

with st.sidebar.expander("Device type", expanded=True):
    device_type = st.selectbox("Device type", DEVICE_TYPES, index=1)

with st.sidebar.expander("Scoring key", expanded=False):
    st.markdown(
        """
**0** Not started  
**1** In progress  
**2** Draft / partial  
**3** Complete & evidenced
"""
    )

with st.sidebar.expander("Optional weights", expanded=False):
    st.caption("Use to emphasise certain workstreams for triage. Defaults are sensible for early HTA.")
    weights = {}
    for key in TASK_ORDER:
        weights[key] = st.slider(
            TASK_DISPLAY[key],
            min_value=0.5,
            max_value=2.0,
            value=float(DEFAULT_WEIGHTS.get(key, 1.0)),
            step=0.1,
            key=f"w-{key}",
        )

st.sidebar.caption("Tip: quick honesty — this is triage, not an audit.")

# -----------------------------
# Main
# -----------------------------
st.title(APP_NAME)
st.caption("A 5-minute structured check of problem definition, pathway fit, value logic, evidence, and uncertainty.")

st.markdown(
    """
This is an **early HTA triage** check for med-tech, diagnostics, service innovations, and digital health ventures. This helps teams to:

- Identify evidence, pathway, and implementation gaps early  
- Clarify the value proposition and decision context for health systems  
- Prioritise the next piece of evidence needed to reduce uncertainty

Rate the 10 workstreams to identify evidence and implementation gaps and suggest next steps. Results can be exported as CSV.
"""
)
st.markdown("---")
# st.info("Use this to prioritise what to do next, align stakeholders, and shape the evidence + value plan early.")
left, right = st.columns([1.6, 1.0])

with left:
    st.subheader("Rate workstreams")
    task_scores = {}
    for i, key in enumerate(TASK_ORDER, start=1):
        question = TASKS[key]
        choice = st.radio(
            f"{i}. {question}",
            STATUS_LABELS,
            index=0,
            horizontal=True,
            key=f"task-{key}",
        )
        task_scores[key] = int(choice.split(" ")[0])

with right:
    st.subheader("Your overall score")
    st.write(f"**Venture:** {venture_name}")
    st.write(f"**Device type:** {device_type}")

    # Weighted readiness (0–100)
    w_sum = sum(weights.values()) if weights else len(TASK_ORDER)
    weighted_score = (
        sum(task_scores[k] * weights.get(k, 1.0) for k in TASK_ORDER) / (3 * w_sum)
        if w_sum
        else 0
    )
    overall_pct = round(100 * weighted_score, 1)

    st.metric("Weighted readiness", f"{overall_pct}%")
    pill(label_for_pct(overall_pct), color_for_pct(overall_pct))
    # st.caption("Combines all workstreams with optional weights")

st.markdown("---")

c1, c2, c3 = st.columns([1.2, 1.2, 1.1])
with c1:
    st.subheader("Radar")
    make_radar(task_scores)
with c2:
    st.subheader("Bar")
    make_bar(task_scores)
with c3:
    st.subheader("Next steps")
    steps = top_next_steps(task_scores, weights, n=4)
    for s in steps:
        st.write(f"• {s}")

st.markdown("---")
st.subheader("Data review & export")

rows = []
ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
for key in TASK_ORDER:
    rows.append(
        {
            "Timestamp_UTC": ts,
            "Venture": venture_name,
            "Team": team_name,
            "Contact": contact,
            "Device type": device_type,
            "Workstream": key,
            "Workstream_label": TASK_DISPLAY[key],
            "Question": TASKS[key],
            "Score": task_scores.get(key, 0),
            "Weight": weights.get(key, 1.0),
        }
    )

df = pd.DataFrame(rows)

df_no_label = df.drop(columns=["Workstream_label","Weight"])

st.dataframe(df_no_label, use_container_width=True, hide_index=True)

st.download_button(
    "Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"lihe_hta_triage_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
)

st.markdown("---")
st.subheader("Further reading and resources")

with st.expander("Frameworks and guidance"):
    st.markdown("""
**Evidence standards**

- [NICE Evidence Standards Framework for Digital Health Technologies](https://www.nice.org.uk/what-nice-does/digital-health/evidence-standards-framework-esf-for-digital-health-technologies)  
- [NICE Technology Appraisal guidance](https://www.nice.org.uk/what-nice-does/our-guidance/about-technology-appraisal-guidance)  
- [NICE Real-World Evidence Framework](https://www.nice.org.uk/corporate/ecd9)  
- [ISPOR Health Economic Evaluation Guidelines](https://www.ispor.org/heor-resources/good-practices)

These help determine what **level of clinical and economic evidence** is expected before adoption.
""")

with st.expander("Economic data and costing"):
    st.markdown("""
**NHS cost datasets**

- [NHS National Cost Collection](https://www.england.nhs.uk/costing-in-the-nhs/national-cost-collection/)  
- [NHS Reference Costs dataset](https://www.gov.uk/government/collections/nhs-reference-costs)

These datasets provide **average NHS costs for healthcare services**, commonly used in health-economic models.
- Published cost studies in peer-reviewed literature
- [OECD health system data](https://www.oecd.org/en/data/datasets/oecd-health-statistics.html)
- [Cost-effectiveness app](https://cost-effectiveness-analysis.streamlit.app/)
""")

with st.expander("Further support for innovators"):
    st.markdown("""
- NHS Innovation Service  
- NIHR research funding and evidence programmes  
- DigitalHealth.London guidance for innovators  
- NHS England AI Lab (for AI-based technologies)
""")

st.info("""
Evidence should grow with the product lifecycle.

Early stage: feasibility data, pathway mapping, value hypothesis  
Mid stage: clinical validation and comparative evidence  
Later stage: real-world outcomes and economic value
""")


st.markdown("---")
st.markdown("""  
**Created by Ben Caswell-Midwinter | © 2026 | [LinkedIn](https://www.linkedin.com/in/ben-caswell-midwinter-7a0701107/)**
""")
st.caption("Indicative triage only: not a formal HTA, regulatory, or clinical safety assessment.")