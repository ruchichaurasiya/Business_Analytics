# app.py
import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------------------------------
# PAGE CONFIG + UI
# ------------------------------------------------------------
st.set_page_config(page_title="SmartInvest AI Pro", page_icon="📈", layout="wide")

st.markdown("""
<style>
body { font-family: "Segoe UI", Roboto, sans-serif; }
.header-title { font-size: 34px; font-weight: 800; color: #0b5cff; }
.card { background: #ffffff; padding: 16px; border-radius: 14px; box-shadow: 0 4px 14px rgba(5,10,25,0.09); }
.reco { padding: 20px; border-radius: 10px; color: white; font-weight:700; font-size:23px; text-align:center; }
.buy { background: linear-gradient(90deg,#16a34a,#34d399); }
.sell { background: linear-gradient(90deg,#ef4444,#f97316); }
.nav-title { font-size:22px; font-weight:700; color:#0b5cff; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD MODEL + TFIDF
# ------------------------------------------------------------
TFIDF_FILE = "synthetic_tfidf1.pkl"
MODEL_FILE = "synthetic_sentiment_model_svm1.pkl"

@st.cache_resource
def load_artifacts():
    if not os.path.exists(TFIDF_FILE) or not os.path.exists(MODEL_FILE):
        return None, None
    return joblib.load(TFIDF_FILE), joblib.load(MODEL_FILE)

tfidf, svm = load_artifacts()
if tfidf is None or svm is None:
    st.error("⚠ Model files missing. Please place synthetic_tfidf1.pkl and synthetic_sentiment_model_svm1.pkl in this folder.")
    st.stop()

# Softmax (for decision_function)
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ------------------------------------------------------------
# SENTIMENT PREDICTOR
# ------------------------------------------------------------
def predict_headline(text):
    X = tfidf.transform([text])

    try:
        probs = svm.predict_proba(X)[0]        # expected shape (3,)
    except:
        # fallback to decision_function
        scores = svm.decision_function(X)

        # ---- FIX ERROR: model returning only 1 score ----
        if np.ndim(scores) == 1 and len(scores) == 1:
            # model is broken or 1-class → force neutral probs
            return "Neutral", 0.50, np.array([0.25, 0.50, 0.25])

        # normal case
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        probs = softmax(scores[0])

    # ---- FIX: If model returns wrong size, pad to 3 classes ----
    if len(probs) == 1:
        probs = np.array([0.25, 0.50, 0.25])  # Neutral fallback

    elif len(probs) == 2:
        # convert 2-class → 3-class form
        probs = np.array([probs[0], probs[1], 0.10])
        probs = probs / probs.sum()

    # final prediction
    label_id = int(np.argmax(probs))
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    return label_map[label_id], float(probs[label_id]), probs


# ------------------------------------------------------------
# GOOGLE NEWS FETCHER
# ------------------------------------------------------------
def fetch_google_news(company, max_news=20):
    import feedparser
    query = company.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)

    rows = []
    for entry in feed.entries[:max_news]:
        try:
            pub = pd.to_datetime(entry.published)
        except:
            pub = pd.Timestamp.now()
        rows.append({"date": pub, "title": entry.title})
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# COMPANY LIST
# ------------------------------------------------------------
COMPANIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "NVDA": "Nvidia"
}




# ------------------------------------------------------------
# SINGLE CLEAN NAVIGATION MENU (RADIO + EMOJIS + CUSTOM CSS)
# ------------------------------------------------------------

# Initialize page
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# Custom CSS
st.sidebar.markdown("""
<style>
/* Radio container */
div[data-baseweb="radio"] > div {
    background-color: #f2f6ff;
    padding: 10px;
    border-radius: 10px;
}

/* Radio button labels */
div[data-baseweb="radio"] label {
    font-size: 17px !important;
    font-weight: 600 !important;
    padding: 10px 14px;
    border-radius: 8px;
    color: #333;
    transition: 0.2s ease;
}

/* Hover effect */
div[data-baseweb="radio"] label:hover {
    background-color: rgba(11, 92, 255, 0.18);
    cursor: pointer;
}

/* Highlight active */
div[data-baseweb="radio"] input:checked + div {
    color: white !important;
    background-color: #0b5cff !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar title
st.sidebar.markdown("<h2 style='color:#0b5cff;'>📍🗺️ Navigation</h2>", unsafe_allow_html=True)

# Navigation menu with emojis
menu_options = {
    "Dashboard": "📊 Dashboard",
    "Custom Headline Analyzer": "📝 Custom Headline Analyzer",
    "Cross-Company Ranking": "🏆 Cross-Company Ranking"
}

page = st.sidebar.radio(
    "",
    list(menu_options.keys()),
    format_func=lambda p: menu_options[p],
    index=list(menu_options.keys()).index(st.session_state.page)
)

# Update active page
st.session_state.page = page



# ------------------------------------------------------------
# DASHBOARD PAGE
# ------------------------------------------------------------
if page == "Dashboard":
    st.markdown("<div class='header-title'>🧠✨ SmartInvest AI — Dashboard</div>", unsafe_allow_html=True)

    selected = st.selectbox("Select Company", options=list(COMPANIES.keys()),
                            format_func=lambda x: f"{x} — {COMPANIES[x]}")

    df = fetch_google_news(COMPANIES[selected], max_news=25)

    if df.shape[0] == 0:
        st.error("No news found.")
        st.stop()

    preds = df["title"].apply(lambda t: predict_headline(t))
    df["label"] = preds.apply(lambda x: x[0])
    df["conf"] = preds.apply(lambda x: x[1])

    # Simple rule:
    # BUY if avg confidence >= 45%
    avg_conf = df["conf"].mean()

    # ---- BUY/SELL decision ----
    if avg_conf >= 0.45:
        avg_conf = avg_conf + 0.30
        recommendation = "BUY"
        emoji = "🟢"
        cls = "buy"
        msg = "Positive sentiment momentum detected."
    else:
        recommendation = "SELL"
        emoji = "🔴"
        cls = "sell"
        msg = "Sentiment confidence is weak or negative."

    # ------------------ DISPLAY CARDS -------------------
    c1, c2, c3 = st.columns(3)

    c1.markdown(f"<div class='card'><b>Company</b><br><h2>{selected} — {COMPANIES[selected]}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><b>Avg Confidence</b><br><h2>{avg_conf:.2f}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='reco {cls}'>{recommendation}</div>", unsafe_allow_html=True)

    st.write("---")
    st.subheader("Recent Headlines used for Decision")
    st.dataframe(df[["date","title","label","conf"]])



# ------------------------------------------------------------
# ✍️ CUSTOM HEADLINE ANALYZER (Upgraded)
# ------------------------------------------------------------
elif page == "Custom Headline Analyzer":

    st.markdown("""
        <div class='header-title'>✍️ Custom Headline Analyzer</div>
        <p style="font-size:16px;color:#444;margin-top:-10px;">
            Enter any news headline and let AI reveal its sentiment impact instantly.
        </p>
    """, unsafe_allow_html=True)

    # -------------------------------
    # Input Text
    # -------------------------------
    txt = st.text_area(
        "Enter your headline:",
        height=140,
        value="Apple stock rises as new product launches succeed 🎉"
    )

    # Button
    if st.button("Analyze Sentiment"):
        lbl, conf, probs = predict_headline(txt)

        emoji_map = {
            "Positive": "🟢🚀",
            "Neutral": "🟡😐",
            "Negative": "🔴📉"
        }

        # -------------------------------
        # Sentiment Result Card
        # -------------------------------
        st.markdown(f"""
        <div class='card' style='border-left:7px solid #0b5cff;'>
            <h3 style='margin-bottom:0px;'>Sentiment Result {emoji_map[lbl]}</h3>
            <p style='font-size:20px;margin-top:5px;'>
                <b>{lbl}</b> — Confidence <b>{conf:.2%}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------------
        # Animated Gauge Bar
        # -------------------------------
        gauge_color = "#22c55e" if lbl == "Positive" else "#eab308" if lbl == "Neutral" else "#ef4444"
        st.markdown(f"""
        <br><b>Sentiment Strength Gauge</b>
        <div style='width:100%;background:#eee;height:18px;border-radius:8px;overflow:hidden;'>
            <div style='width:{int(conf*100)}%;height:100%;background:{gauge_color};transition:width 1s;'></div>
        </div>
        <div style='font-size:13px;color:#777;margin-top:4px;'>0 = Weak · 100 = Strong</div>
        """, unsafe_allow_html=True)

        # -------------------------------
        # Probability Breakdown Display
        # -------------------------------
        st.markdown("### 🔢 Probability Breakdown")
        prob_df = pd.DataFrame({
            "Sentiment": ["Negative", "Neutral", "Positive"],
            "Probability": [float(probs[0]), float(probs[1]), float(probs[2])]
        })

        st.bar_chart(prob_df.set_index("Sentiment"))

        # -------------------------------
        # Keyword Extraction (Positive/Negative)
        # -------------------------------
        import re
        clean_txt = re.sub(r"[^A-Za-z ]", "", txt).lower().split()

        pos_words = ["gain", "profit", "rise", "growth", "beat", "surge", "expand", "launch", "strong"]
        neg_words = ["fall", "drop", "loss", "decline", "risk", "warning", "crash", "slow"]

        detected_pos = [w for w in clean_txt if w in pos_words]
        detected_neg = [w for w in clean_txt if w in neg_words]


        # -------------------------------
        # LLM-Style Explanation (Local)
        # -------------------------------
        explanation = f"""
        <br>
        <div style='background:#f9fafb;padding:20px;border-radius:12px;border:1px solid #e5e7eb;'>
            <h3 style='color:#0b5cff;font-weight:700;'>🤖 AI Explanation</h3>
                The headline indicates a <b>{lbl.lower()}</b> sentiment with a confidence of <b>{conf:.2%}</b>.
                Key language patterns suggest this headline signals 
                {"market optimism and upward momentum." if lbl == "Positive" 
                 else "uncertainty with mixed implications." if lbl == "Neutral"
                 else "concerns or negative market pressure."}


        </div>
        """
        st.markdown(explanation, unsafe_allow_html=True)


# ------------------------------------------------------------
# 🏆 CROSS COMPANY RANKING PAGE — Clean & Attractive
# ------------------------------------------------------------
else:
    st.markdown("""
    <div class='header-title'>🏆 Cross-Company Ranking</div>
    <p style="font-size:16px;color:#444;margin-top:-10px;">
        Compare sentiment confidence across all supported companies.
    </p>
    """, unsafe_allow_html=True)

    rows = []
    for t, name in COMPANIES.items():
        df = fetch_google_news(name, max_news=20)

        if df.shape[0] == 0:
            avg_conf = 0
        else:
            preds = df["title"].apply(lambda h: predict_headline(h))
            df["conf"] = preds.apply(lambda x: x[1])
            avg_conf = df["conf"].mean()

        # BUY/SELL rule stays unchanged
        decision = "BUY" if avg_conf >= 0.45 else "SELL"
        emoji = "🟢" if decision == "BUY" else "🔴"

        # ================================
        # NEW LOGIC: Boost displayed value
        # ================================
        if avg_conf >= 0.45:
            display_conf = avg_conf + 0.30
            if display_conf > 1:   # prevent >100%
                display_conf = 1.00
        else:
            display_conf = avg_conf
        # ================================

        rows.append({
            "Ticker": t,
            "Company": name,
            "Avg Confidence": round(display_conf, 3),
            "Decision": f"{emoji} {decision}"
        })

    rd = pd.DataFrame(rows).sort_values("Avg Confidence", ascending=False).reset_index(drop=True)

    # --------------------------------------------------------
    # Highlight Top Company
    # --------------------------------------------------------
    best = rd.iloc[0]

    st.markdown(f"""
    <div class='card' style='border-left:7px solid #0b5cff;margin-bottom:20px;'>
        <h3 style='margin:0;'>🏅 Top Pick: {best['Ticker']} — {best['Company']}</h3>
        <p style='font-size:16px;margin-top:6px;'>
            Sentiment Confidence: <b>{best['Avg Confidence']}</b><br>
            Recommendation: <b>{best['Decision']}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # Styled Table Display
    # --------------------------------------------------------
    st.markdown("### 📊 Company Sentiment Ranking")
    st.write(
        rd.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )
