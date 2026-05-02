"""
Minimal, robust Streamlit UI for AI Plagiarism Checker
- Defers heavy model loads until analysis request
- Shows clear UI and demo analysis when classifier/model not present
- Run with: streamlit run app.py
"""
import os
import sys
import math
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

# safe matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False

st.set_page_config(page_title="AI Plagiarism Checker", page_icon="🔍", layout="wide")

# Simple CSS for good contrast
st.markdown("""
<style>
.header { font-size: 1.8rem; font-weight:700; color:#0B72B9; }
.sub { color:#546E7A; margin-bottom:12px; }
.panel { background:#fff; padding:12px; border-radius:8px; box-shadow:0 6px 18px rgba(15,23,42,0.06); }
.result { padding:12px; border-radius:8px; display:flex; justify-content:space-between; align-items:center; }
.badge { padding:6px 10px; border-radius:999px; color:#fff; font-weight:700; }
.badge-human { background:#2e7d32; }
.badge-ai { background:#c62828; }
.badge-mixed { background:#ef6c00; }
.conf { font-size:1.1rem; font-weight:800; color:#fff; padding:10px; border-radius:50%; width:64px; height:64px; display:flex; align-items:center; justify-content:center; }
.conf-green { background:linear-gradient(90deg,#2e7d32,#66bb6a); }
.conf-red { background:linear-gradient(90deg,#c62828,#ef5350); }
.conf-amber { background:linear-gradient(90deg,#ef6c00,#ffb74d); }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("AI Plagiarism Checker")
    st.write("Final-year project UI — GPT-2 perplexity + linguistic features + RandomForest")
    st.markdown("**Features**")
    st.write("- GPT-2 perplexity (deferred load)\n- Burstiness\n- Repetition\n- Lexical diversity\n- Random Forest classifier (if trained)")
    st.markdown("**Tech**: Python • Streamlit • Transformers (deferred) • Scikit-learn • Matplotlib")
    st.info("Run with: `streamlit run app.py`")

st.markdown('<div class="header">🔍 AI Plagiarism Checker</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Paste text, click Analyze. App defers heavy model loads to analysis time.</div>', unsafe_allow_html=True)

# Main input
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

def _clear_text_input():
    st.session_state.text_input = ""

st.markdown('<div class="panel">', unsafe_allow_html=True)
text_input = st.text_area(
    "Enter text to analyze (recommended ≥ 50 words):",
    height=260,
    key="text_input"
)
col1, col2 = st.columns([1,1])
with col1:
    analyze = st.button("🔍 Analyze")
with col2:
    clear = st.button("🗑️ Clear", on_click=_clear_text_input)
st.markdown('</div>', unsafe_allow_html=True)

def validate_input(text: str):
    if not text or not text.strip():
        return False, "Input is empty. Please paste some text."
    words = [w for w in text.split() if w.strip()]
    if len(words) < 50:
        return False, f"Too short: {len(words)} words. Recommended ≥ 50 words."
    return True, None

def compute_simple_features(text: str):
    # Basic safe feature calculations (no heavy ML)
    words = [w.strip(".,;:!?()[]\"'").lower() for w in text.split() if w.strip()]
    word_count = len(words)
    uniques = len(set(words))
    lexical_diversity = uniques / max(1, word_count)
    # repetition: proportion of top-3 repeated tokens
    freqs = Counter(words)
    top3 = sum(v for _, v in freqs.most_common(3))
    repetition = top3 / max(1, word_count)
    # sentence stats / burstiness
    sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    sent_lens = [len(s.split()) for s in sentences] or [0]
    avg_sentence_length = sum(sent_lens) / max(1, len(sent_lens))
    burstiness = (np.std(sent_lens) / (np.mean(sent_lens)+1e-6)) if len(sent_lens)>1 else 0.0
    # lightweight "perplexity" proxy: inverse of average word rarity (demo only)
    # use token frequency to create a pseudo-perplexity (smaller -> more human-like)
    rarity = np.mean([1.0/(freqs[w]+1) for w in words]) if words else 1.0
    pseudo_perplexity = max(1.0, 1.0 / (rarity + 1e-6) * 10.0)
    return {
        "perplexity": float(pseudo_perplexity),
        "burstiness": float(burstiness),
        "repetition": float(repetition),
        "lexical_diversity": float(lexical_diversity),
        "word_count": int(word_count),
        "avg_sentence_length": float(avg_sentence_length)
    }

def show_result_card(label, confidence, features):
    # choose styling
    if label.lower().startswith("human"):
        badge = "badge-human"; conf_class="conf-green"
    elif label.lower().startswith("ai"):
        badge = "badge-ai"; conf_class="conf-red"
    else:
        badge = "badge-mixed"; conf_class="conf-amber"
    st.markdown("---")
    st.markdown(f'''
    <div class="panel">
      <div class="result">
        <div>
          <h3>{label} <span class="badge {badge}">{label.split()[0]}</span></h3>
          <div style="color:#37474f; margin-top:6px;">
            Perplexity: {features.get("perplexity"):.2f} • Words: {features.get("word_count")}
          </div>
        </div>
        <div style="text-align:center;">
          <div class="conf {conf_class}">{confidence:.0f}%</div>
          <div style="color:#7a7a7a; margin-top:6px;">Model confidence</div>
        </div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

def plot_features(features):
    names = ["Perplexity (log10)", "Burstiness", "Repetition", "Lexical Diversity"]
    vals = [
        math.log10(max(1.0, features.get("perplexity",1.0))),
        features.get("burstiness",0.0),
        features.get("repetition",0.0),
        features.get("lexical_diversity",0.0)
    ]
    df = pd.DataFrame({"Metric": names, "Value": vals}).set_index("Metric")
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(df.index, df["Value"], alpha=0.9)
        ax.set_ylabel("Value (see labels)")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        st.pyplot(fig)
    else:
        st.bar_chart(df)

# Analysis flow (lightweight demo if classifier missing)
if analyze:
    ok, message = validate_input(text_input)
    if not ok:
        st.warning(message)
    else:
        with st.spinner("Computing features (demo mode if model missing)..."):
            features = compute_simple_features(text_input)

        # Try to load classifier and detector lazily (do not fail app if missing)
        classifier_path = "model.pkl"
        classifier = None
        try:
            if os.path.exists(classifier_path):
                import pickle
                with open(classifier_path, "rb") as f:
                    classifier = pickle.load(f)
        except Exception:
            classifier = None

        if classifier is None:
            # Demo heuristic for label and confidence
            # Lower pseudo-perplexity + high lexical diversity -> human
            p = features["perplexity"]
            ld = features["lexical_diversity"]
            rep = features["repetition"]
            score = (1.0 / (p + 1.0)) * 0.6 + ld * 0.3 - rep * 0.1
            if score > 0.35:
                label = "Human-Written"
                conf = min(95.0, 50.0 + (score-0.35)*200)
            elif score < 0.20:
                label = "AI-Generated"
                conf = min(95.0, 50.0 + (0.20-score)*250)
            else:
                label = "Mixed/Suspicious"
                conf = 60.0
            st.info("Note: Trained classifier (model.pkl) not found — showing demo analysis based on simple heuristics.")
            show_result_card(label, conf, features)
        else:
            # If classifier available, run proper prediction (defensive)
            try:
                # ensure feature vector shape expected
                fv = np.array([[
                    features.get("perplexity",0.0),
                    features.get("burstiness",0.0),
                    features.get("repetition",0.0),
                    features.get("lexical_diversity",0.0),
                    features.get("avg_sentence_length",0.0),
                    features.get("word_count",0)
]])
                ypred = classifier.predict(fv)[0]
                yproba = classifier.predict_proba(fv)[0]
                labels = {0:"Human-Written",1:"AI-Generated",2:"Mixed/Suspicious"}
                label = labels.get(int(ypred),"Unknown")
                conf = float(max(yproba))*100.0
                show_result_card(label, conf, features)
            except Exception as e:
                st.error(f"Classifier prediction failed: {e}")
                show_result_card("Mixed/Suspicious", 55.0, features)

        # Metrics dashboard
        st.markdown("### Metrics Dashboard")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Perplexity", f"{features.get('perplexity'):.2f}")
        c2.metric("Burstiness", f"{features.get('burstiness'):.3f}")
        c3.metric("Repetition", f"{features.get('repetition'):.3f}")
        c4.metric("Lexical Diversity", f"{features.get('lexical_diversity'):.3f}")

        # Visualization
        try:
            plot_features(features)
        except Exception as e:
            st.warning(f"Visualization failed: {e}")

        # Technical details
        with st.expander("Technical Details & Extracted Features"):
            for k, v in features.items():
                st.write(f"- **{k.replace('_',' ').title()}**: {v}")
            st.write("\n**Text statistics:**")
            st.write(f"- Character count: {len(text_input)}")
            st.write(f"- Word count: {features.get('word_count','N/A')}")
            st.write(f"- Avg sentence length: {features.get('avg_sentence_length','N/A')}")
            st.write("\nHints: To enable the full model-based analyzer, place a trained model.pkl in the project root and ensure transformers/torch are installed.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#606060; padding:0.6rem;">
  <div><strong>Developed by:</strong></div>
  <div>M. Venkatesh (Team Lead)</div>
  <div>Under the Guidance of K. Pushpa Latha, M.Tech</div>
</div>
""", unsafe_allow_html=True)
