"""
E-commerce Delivery Time Prediction — Streamlit app.

Loads the trained pipeline from model.pkl (preprocessing + LinearRegression,
trained on log(Delivery_Time_Hours)) and predicts delivery time in hours for
a single order entered through the UI. Does NOT retrain on startup.
"""

import textwrap

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def md(html: str) -> None:
    """Render HTML via st.markdown with dedent to avoid 4-space Markdown
    code-block rendering."""
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Custom CSS — premium vibrant theme w/ glassmorphism + rich animations
# ---------------------------------------------------------------------------
md(
    """
<style>
/* ============================================================
   BASE / BACKGROUND — aurora gradient + floating orbs
   ============================================================ */
.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(255, 183, 110, 0.32) 0%, transparent 60%),
        radial-gradient(900px 500px at 110% 10%, rgba(120, 200, 255, 0.28) 0%, transparent 55%),
        radial-gradient(800px 600px at 50% 110%, rgba(180, 120, 255, 0.28) 0%, transparent 60%),
        linear-gradient(135deg, #1a1b3a 0%, #2d1f4a 40%, #3a1e3d 70%, #4a2232 100%);
    background-attachment: fixed;
    color: #f2eadd;
    animation: auroraShift 22s ease-in-out infinite alternate;
    min-height: 100vh;
    position: relative;
}
@keyframes auroraShift {
    0%   { filter: hue-rotate(0deg) brightness(1); }
    50%  { filter: hue-rotate(12deg) brightness(1.06); }
    100% { filter: hue-rotate(-8deg) brightness(0.98); }
}

/* Floating gradient orbs */
.stApp::before,
.stApp::after {
    content: "";
    position: fixed;
    border-radius: 50%;
    filter: blur(120px);
    opacity: 0.55;
    pointer-events: none;
    z-index: 0;
    mix-blend-mode: screen;
}
.stApp::before {
    width: 560px; height: 560px;
    background: radial-gradient(circle, #ffb56b 0%, transparent 70%);
    top: -160px; left: -120px;
    animation: orbFloat 18s ease-in-out infinite;
}
.stApp::after {
    width: 640px; height: 640px;
    background: radial-gradient(circle, #78c8ff 0%, transparent 70%);
    bottom: -220px; right: -140px;
    animation: orbFloat 24s ease-in-out infinite reverse;
}
@keyframes orbFloat {
    0%,100% { transform: translate(0,0) scale(1); }
    33%     { transform: translate(50px,-20px) scale(1.06); }
    66%     { transform: translate(-30px, 40px) scale(0.97); }
}

/* Subtle star twinkle dots */
.block-container::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,0.35) 50%, transparent 51%),
        radial-gradient(1px 1px at 70% 60%, rgba(255,255,255,0.28) 50%, transparent 51%),
        radial-gradient(1.5px 1.5px at 45% 80%, rgba(255,255,255,0.22) 50%, transparent 51%),
        radial-gradient(1px 1px at 85% 15%, rgba(255,255,255,0.30) 50%, transparent 51%),
        radial-gradient(1px 1px at 10% 70%, rgba(255,255,255,0.25) 50%, transparent 51%);
    pointer-events: none;
    z-index: 0;
    animation: twinkle 6s ease-in-out infinite alternate;
}
@keyframes twinkle {
    0%   { opacity: 0.6; }
    100% { opacity: 1; }
}

/* ============================================================
   TYPOGRAPHY — vibrant gradient + crisp readable colors
   ============================================================ */
h1, h2, h3, h4, h5, h6 {
    background: linear-gradient(
        110deg,
        #fff4e0 0%,
        #ffd78a 25%,
        #ff9a76 55%,
        #c3a6ff 85%,
        #9dd8ff 100%
    );
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    font-weight: 800;
    animation: textGradient 8s ease-in-out infinite;
    text-shadow: 0 0 30px rgba(255,180,120,0.15);
}
@keyframes textGradient {
    0%,100% { background-position: 0% 50%; }
    50%     { background-position: 100% 50%; }
}
p, span, li {
    color: #ede4d3 !important;
}
.stMarkdown strong {
    color: #ffcc70 !important;
    font-weight: 700;
    text-shadow: 0 0 12px rgba(255,204,112,0.35);
}

/* ============================================================
   GLASS CARD BASE — premium glassmorphism
   ============================================================ */
.glass-card {
    position: relative;
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.10) 0%,
        rgba(255,255,255,0.04) 100%
    );
    backdrop-filter: blur(28px) saturate(180%);
    -webkit-backdrop-filter: blur(28px) saturate(180%);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 22px;
    padding: 30px 32px;
    margin-bottom: 22px;
    box-shadow:
        0 16px 56px rgba(10, 8, 28, 0.55),
        inset 0 1px 0 rgba(255,255,255,0.18),
        inset 0 -1px 0 rgba(255,255,255,0.04);
    transition: transform 0.4s cubic-bezier(0.22,1,0.36,1),
                box-shadow 0.4s cubic-bezier(0.22,1,0.36,1),
                border-color 0.4s ease;
    z-index: 1;
    overflow: hidden;
}
.glass-card::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 22px;
    padding: 1px;
    background: linear-gradient(
        135deg,
        rgba(255,210,140,0.55),
        rgba(255,255,255,0.05) 40%,
        rgba(195,166,255,0.25) 80%,
        rgba(157,216,255,0.35)
    );
    -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
            mask-composite: exclude;
    pointer-events: none;
    z-index: 0;
}
.glass-card:hover {
    transform: translateY(-5px);
    border-color: rgba(255,220,170,0.35);
    box-shadow:
        0 28px 72px rgba(10, 8, 28, 0.65),
        inset 0 1px 0 rgba(255,255,255,0.28),
        0 0 40px rgba(255,180,120,0.12);
}

/* ============================================================
   HERO CARD — golden sunrise gradient
   ============================================================ */
.hero-card {
    background:
        radial-gradient(600px 300px at 90% -20%, rgba(255,180,100,0.35) 0%, transparent 65%),
        radial-gradient(500px 300px at -10% 120%, rgba(157,216,255,0.22) 0%, transparent 65%),
        linear-gradient(135deg,
            rgba(255,210,140,0.14) 0%,
            rgba(195,166,255,0.10) 50%,
            rgba(157,216,255,0.10) 100%);
    border: 1px solid rgba(255,210,140,0.22);
    overflow: hidden;
}
.hero-card::after {
    content: "";
    position: absolute;
    top: -140px; right: -140px;
    width: 420px; height: 420px;
    background: conic-gradient(
        from 120deg,
        rgba(255,210,140,0.0) 0deg,
        rgba(255,210,140,0.45) 80deg,
        rgba(255,180,100,0.30) 140deg,
        rgba(195,166,255,0.30) 210deg,
        rgba(157,216,255,0.30) 280deg,
        rgba(255,210,140,0.0) 360deg
    );
    border-radius: 50%;
    pointer-events: none;
    filter: blur(8px);
    animation: slowRotate 30s linear infinite;
    z-index: 0;
}
@keyframes slowRotate {
    to { transform: rotate(360deg); }
}

/* ============================================================
   HERO — headline shimmer, glow + stat pills
   ============================================================ */
.hero-headline {
    font-size: 3rem;
    font-weight: 900;
    line-height: 1.05;
    margin-bottom: 16px;
    background: linear-gradient(
        90deg,
        #fff4e0 0%,
        #ffd78a 15%,
        #ff9a76 32%,
        #ffcc70 50%,
        #c3a6ff 70%,
        #9dd8ff 88%,
        #fff4e0 100%
    );
    background-size: 220% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 7s linear infinite;
    filter: drop-shadow(0 0 24px rgba(255,180,120,0.25));
    position: relative;
    z-index: 2;
}
@keyframes shimmer {
    to { background-position: 220% center; }
}
.hero-sub {
    font-size: 1.08rem;
    color: #e8dfcc !important;
    line-height: 1.65;
    max-width: 860px;
    position: relative;
    z-index: 2;
}
.hero-sub b {
    background: linear-gradient(90deg, #ffd78a, #ffcc70);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 24px;
    position: relative;
    z-index: 2;
}
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 10px 18px;
    border-radius: 999px;
    font-size: 0.86rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    border: 1px solid rgba(255,255,255,0.15);
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.10),
        rgba(255,255,255,0.03)
    );
    backdrop-filter: blur(12px);
    box-shadow:
        0 6px 16px rgba(10,8,28,0.35),
        inset 0 1px 0 rgba(255,255,255,0.20);
    transition: all 0.3s cubic-bezier(0.22,1,0.36,1);
    color: #f6ecd9 !important;
}
.stat-pill:hover {
    transform: translateY(-2px);
    box-shadow:
        0 12px 28px rgba(10,8,28,0.45),
        inset 0 1px 0 rgba(255,255,255,0.28);
}
.stat-pill.gold   { background: linear-gradient(135deg, rgba(255,204,112,0.28), rgba(255,154,118,0.12)); border-color: rgba(255,204,112,0.45); color: #fff4d9 !important; box-shadow: 0 6px 18px rgba(255,180,80,0.20), inset 0 1px 0 rgba(255,255,255,0.22); }
.stat-pill.caramel{ background: linear-gradient(135deg, rgba(255,154,118,0.26), rgba(255,110,120,0.12)); border-color: rgba(255,154,118,0.42); color: #fff0e8 !important; box-shadow: 0 6px 18px rgba(255,120,100,0.18), inset 0 1px 0 rgba(255,255,255,0.22); }
.stat-pill.clay   { background: linear-gradient(135deg, rgba(195,166,255,0.26), rgba(120,180,255,0.10)); border-color: rgba(195,166,255,0.40); color: #f1e8ff !important; box-shadow: 0 6px 18px rgba(170,130,255,0.18), inset 0 1px 0 rgba(255,255,255,0.22); }
.stat-pill.mocha  { background: linear-gradient(135deg, rgba(120,200,255,0.24), rgba(157,216,255,0.10)); border-color: rgba(120,200,255,0.38); color: #e8f6ff !important; box-shadow: 0 6px 18px rgba(100,180,255,0.16), inset 0 1px 0 rgba(255,255,255,0.22); }
.stat-pill.gold:hover   { box-shadow: 0 10px 28px rgba(255,180,80,0.32), inset 0 1px 0 rgba(255,255,255,0.30); }
.stat-pill.caramel:hover{ box-shadow: 0 10px 28px rgba(255,120,100,0.30), inset 0 1px 0 rgba(255,255,255,0.30); }
.stat-pill.clay:hover   { box-shadow: 0 10px 28px rgba(170,130,255,0.30), inset 0 1px 0 rgba(255,255,255,0.30); }
.stat-pill.mocha:hover  { box-shadow: 0 10px 28px rgba(100,180,255,0.28), inset 0 1px 0 rgba(255,255,255,0.30); }

/* ============================================================
   SECTION TITLE — neon ribbon bar
   ============================================================ */
.section-title {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 10px 4px 22px 4px;
    position: relative;
    z-index: 2;
}
.section-title .ribbon {
    width: 7px;
    height: 34px;
    border-radius: 999px;
    background: linear-gradient(180deg,
        #ffcc70 0%,
        #ff9a76 35%,
        #c3a6ff 70%,
        #78c8ff 100%);
    box-shadow:
        0 0 18px rgba(255,204,112,0.55),
        0 0 30px rgba(195,166,255,0.25);
    animation: ribbonGlow 4s ease-in-out infinite alternate;
}
@keyframes ribbonGlow {
    0%   { box-shadow: 0 0 14px rgba(255,204,112,0.45), 0 0 24px rgba(195,166,255,0.18); }
    100% { box-shadow: 0 0 22px rgba(255,204,112,0.70), 0 0 40px rgba(157,216,255,0.30); }
}
.section-title h2 {
    font-size: 1.7rem;
    margin: 0;
    font-weight: 800;
}

/* ============================================================
   STREAMLIT WIDGETS — glass inputs with neon focus
   ============================================================ */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
.stNumberInput > div[data-testid="stNumberInputContainer"],
div[data-baseweb="base-input"] input {
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.08) 0%,
        rgba(255,255,255,0.03) 100%) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    color: #fff4e0 !important;
    border-radius: 14px !important;
    backdrop-filter: blur(14px);
    transition: all 0.3s cubic-bezier(0.22,1,0.36,1);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.08),
        0 4px 14px rgba(10,8,28,0.25);
}
div[data-baseweb="input"] > div:hover,
div[data-baseweb="select"] > div:hover,
.stNumberInput > div[data-testid="stNumberInputContainer"]:hover {
    border-color: rgba(255,204,112,0.38) !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.12),
        0 0 0 3px rgba(255,204,112,0.08),
        0 6px 20px rgba(10,8,28,0.35);
}
div[data-baseweb="input"]:focus-within > div,
div[data-baseweb="select"]:focus-within > div,
.stNumberInput:focus-within > div[data-testid="stNumberInputContainer"] {
    border-color: #ffcc70 !important;
    box-shadow:
        0 0 0 4px rgba(255,204,112,0.16),
        0 0 24px rgba(255,204,112,0.22),
        0 10px 28px rgba(10,8,28,0.40),
        inset 0 1px 0 rgba(255,255,255,0.15) !important;
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.12) 0%,
        rgba(255,245,220,0.06) 100%) !important;
}
input, select, textarea {
    color: #fff4e0 !important;
    caret-color: #ffcc70 !important;
    background: transparent !important;
}
input::placeholder, textarea::placeholder {
    color: rgba(237,228,211,0.45) !important;
}
label, .stSlider > label, .stNumberInput label, .stSelectbox label {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: #e8dfcc !important;
    letter-spacing: 0.02em;
    margin-bottom: 7px !important;
}
.stSlider > label::before         { content: "🎚️ "; }
.stNumberInput label::before      { content: "🔢 "; }
.stSelectbox label::before        { content: "📋 "; }

/* Increment/Decrement number input buttons */
button[data-testid="stNumberInputDecrement"]:hover,
button[data-testid="stNumberInputIncrement"]:hover {
    background: rgba(255,204,112,0.18) !important;
    border-color: rgba(255,204,112,0.35) !important;
}

/* Sliders — vibrant gradient track + glow thumb */
div[data-testid="stTickBar"] > div {
    background: linear-gradient(90deg,
        #78c8ff 0%,
        #c3a6ff 35%,
        #ff9a76 70%,
        #ffcc70 100%) !important;
    height: 7px !important;
    border-radius: 999px;
    box-shadow: 0 0 14px rgba(255,204,112,0.30);
}
div[role="slider"] {
    background: linear-gradient(135deg, #fff4e0, #ffcc70) !important;
    border: 2px solid #fff4d9 !important;
    box-shadow:
        0 0 0 6px rgba(255,204,112,0.22),
        0 0 18px rgba(255,204,112,0.55),
        0 4px 14px rgba(10,8,28,0.50) !important;
    width: 22px !important;
    height: 22px !important;
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.22,1,0.36,1);
}
div[role="slider"]:hover {
    transform: scale(1.18);
    box-shadow:
        0 0 0 8px rgba(255,204,112,0.28),
        0 0 28px rgba(255,204,112,0.70),
        0 6px 20px rgba(10,8,28,0.55) !important;
}

/* Dropdown menu — glass */
ul[role="listbox"] {
    background: linear-gradient(
        180deg,
        rgba(40,30,70,0.88) 0%,
        rgba(30,20,55,0.92) 100%) !important;
    border: 1px solid rgba(255,204,112,0.28) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    box-shadow:
        0 24px 56px rgba(10,8,28,0.70),
        0 0 30px rgba(255,180,120,0.10),
        inset 0 1px 0 rgba(255,255,255,0.10) !important;
    padding: 8px 6px !important;
}
li[role="option"] {
    color: #ede4d3 !important;
    padding: 10px 14px !important;
    border-radius: 10px !important;
    margin: 2px 4px !important;
    transition: all 0.2s ease;
}
li[role="option"]:hover,
li[aria-selected="true"] {
    background: linear-gradient(90deg,
        rgba(255,204,112,0.22) 0%,
        rgba(195,166,255,0.16) 100%) !important;
    color: #fff4e0 !important;
    font-weight: 600 !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.14);
}

/* ============================================================
   PRIMARY BUTTON — aurora gradient + glow pulse
   ============================================================ */
div[data-testid="stFormSubmitButton"] button,
button[kind="secondaryFormSubmit"] {
    margin-top: 16px !important;
    padding: 15px 28px !important;
    font-size: 1.02rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.03em !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,244,224,0.30) !important;
    color: #2a1810 !important;
    background: linear-gradient(
        135deg,
        #ffd78a 0%,
        #ffcc70 22%,
        #ff9a76 50%,
        #c3a6ff 80%,
        #9dd8ff 100%
    ) !important;
    background-size: 200% 200% !important;
    box-shadow:
        0 12px 32px rgba(255,180,100,0.35),
        0 0 28px rgba(195,166,255,0.20),
        0 0 0 1px rgba(255,255,255,0.35) inset,
        0 1px 0 rgba(255,255,255,0.60) inset !important;
    transition: all 0.4s cubic-bezier(0.22,1,0.36,1) !important;
    animation:
        auroraBtn 7s ease-in-out infinite,
        btnFloat 5s ease-in-out infinite;
    position: relative;
    overflow: hidden;
}
div[data-testid="stFormSubmitButton"] button::before,
button[kind="secondaryFormSubmit"]::before {
    content: "";
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255,255,255,0.45),
        transparent
    );
    transition: left 0.7s ease;
}
div[data-testid="stFormSubmitButton"] button:hover::before,
button[kind="secondaryFormSubmit"]:hover::before {
    left: 100%;
}
div[data-testid="stFormSubmitButton"] button:hover,
button[kind="secondaryFormSubmit"]:hover {
    transform: translateY(-4px) scale(1.025) !important;
    background-position: 100% 0% !important;
    box-shadow:
        0 22px 52px rgba(255,160,90,0.45),
        0 0 50px rgba(195,166,255,0.35),
        0 0 0 1px rgba(255,255,255,0.50) inset,
        0 1px 0 rgba(255,255,255,0.75) inset !important;
}
div[data-testid="stFormSubmitButton"] button:active,
button[kind="secondaryFormSubmit"]:active {
    transform: translateY(-1px) scale(0.995) !important;
}
@keyframes auroraBtn {
    0%,100% { filter: brightness(1) saturate(1); }
    50%     { filter: brightness(1.08) saturate(1.1); }
}
@keyframes btnFloat {
    0%,100% { margin-top: 16px; }
    50%     { margin-top: 13px; }
}
div[data-testid="stFormSubmitButton"] {
    text-align: center;
}

/* ============================================================
   PREDICTION RESULT CARD — premium reveal w/ aurora
   ============================================================ */
.result-card {
    background:
        radial-gradient(500px 260px at 50% -30%, rgba(255,220,140,0.35) 0%, transparent 65%),
        radial-gradient(400px 300px at 0% 110%, rgba(195,166,255,0.20) 0%, transparent 65%),
        radial-gradient(400px 300px at 100% 110%, rgba(120,200,255,0.18) 0%, transparent 65%),
        linear-gradient(135deg,
            rgba(255,220,160,0.18) 0%,
            rgba(255,204,112,0.10) 40%,
            rgba(195,166,255,0.08) 75%,
            rgba(157,216,255,0.10) 100%);
    border: 1px solid rgba(255,220,160,0.32) !important;
    position: relative;
    overflow: hidden;
    animation:
        resultReveal 0.8s cubic-bezier(0.22,1,0.36,1),
        resultGlow 5s ease-in-out infinite 0.8s;
}
.result-card::before {
    content: "";
    position: absolute;
    top: -100px; left: 50%;
    width: 620px; height: 300px;
    transform: translateX(-50%);
    background: radial-gradient(ellipse,
        rgba(255,236,180,0.60) 0%,
        rgba(255,204,112,0.25) 35%,
        transparent 70%);
    pointer-events: none;
    filter: blur(2px);
}
.result-card::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(
        120deg,
        transparent 20%,
        rgba(255,255,255,0.12) 45%,
        rgba(255,255,255,0.02) 50%,
        transparent 70%);
    background-size: 200% 100%;
    animation: resultSheen 4s ease-in-out infinite 1.2s;
    pointer-events: none;
    border-radius: 22px;
}
@keyframes resultReveal {
    from {
        opacity: 0;
        transform: translateY(28px) scale(0.98);
        filter: blur(6px);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
        filter: blur(0);
    }
}
@keyframes resultGlow {
    0%,100% { box-shadow:
                0 18px 56px rgba(10,8,28,0.55),
                0 0 28px rgba(255,204,112,0.18),
                inset 0 1px 0 rgba(255,255,255,0.22); }
    50%     { box-shadow:
                0 18px 56px rgba(10,8,28,0.55),
                0 0 50px rgba(255,204,112,0.32),
                0 0 70px rgba(195,166,255,0.14),
                inset 0 1px 0 rgba(255,255,255,0.28); }
}
@keyframes resultSheen {
    0%,100% { background-position: 120% 0; }
    50%     { background-position: -20% 0; }
}
.result-big {
    font-size: 3.8rem;
    font-weight: 900;
    line-height: 1;
    background: linear-gradient(90deg,
        #fff4d9 0%,
        #ffd78a 22%,
        #ff9a76 45%,
        #c3a6ff 70%,
        #9dd8ff 92%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    margin: 12px 0;
    animation: shimmer 5s linear infinite;
    filter: drop-shadow(0 0 22px rgba(255,204,112,0.40));
    position: relative;
    z-index: 2;
}
.result-unit {
    font-size: 1.15rem;
    font-weight: 800;
    color: #fff4d9 !important;
    letter-spacing: 0.30em;
    text-transform: uppercase;
    text-shadow: 0 0 14px rgba(255,204,112,0.35);
    position: relative;
    z-index: 2;
}
.result-days {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 999px;
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.15),
        rgba(255,204,112,0.10));
    border: 1px solid rgba(255,220,160,0.35);
    color: #fff4d9 !important;
    font-weight: 800;
    margin-top: 8px;
    backdrop-filter: blur(10px);
    box-shadow:
        0 4px 14px rgba(10,8,28,0.35),
        inset 0 1px 0 rgba(255,255,255,0.22);
    position: relative;
    z-index: 2;
}
.result-chip {
    font-size: 0.80rem;
    letter-spacing: 0.26em;
    text-transform: uppercase;
    color: #ffcc70 !important;
    font-weight: 800;
    text-shadow: 0 0 12px rgba(255,204,112,0.45);
    position: relative;
    z-index: 2;
}
.result-context {
    font-size: 1.08rem;
    color: #ede4d3 !important;
    font-weight: 600;
    position: relative;
    z-index: 2;
}

/* ============================================================
   INSIGHTS GRID — staggered glass cards w/ top glow bar
   ============================================================ */
.insights-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 16px;
    margin-top: 18px;
    position: relative;
    z-index: 2;
}
.insight {
    position: relative;
    padding: 18px 20px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.08) 0%,
        rgba(255,255,255,0.03) 100%);
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px) saturate(160%);
    box-shadow:
        0 10px 28px rgba(10,8,28,0.45),
        inset 0 1px 0 rgba(255,255,255,0.15);
    transition: all 0.35s cubic-bezier(0.22,1,0.36,1);
    overflow: hidden;
    animation: insightReveal 0.7s cubic-bezier(0.22,1,0.36,1) backwards;
}
.insight:nth-child(1) { animation-delay: 0.90s; }
.insight:nth-child(2) { animation-delay: 1.05s; }
.insight:nth-child(3) { animation-delay: 1.20s; }
.insight:nth-child(4) { animation-delay: 1.35s; }
@keyframes insightReveal {
    from {
        opacity: 0;
        transform: translateY(22px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
.insight::after {
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3.5px;
    background: linear-gradient(90deg,
        #ffcc70 0%,
        #ff9a76 35%,
        #c3a6ff 70%,
        #78c8ff 100%);
    background-size: 200% 100%;
    opacity: 0;
    transition: opacity 0.35s ease;
    box-shadow: 0 0 14px rgba(255,204,112,0.55);
}
.insight:hover::after {
    opacity: 1;
    animation: shimmer 3.5s linear infinite;
}
.insight:hover {
    transform: translateY(-5px);
    border-color: rgba(255,220,160,0.28);
    box-shadow:
        0 20px 44px rgba(10,8,28,0.55),
        0 0 24px rgba(255,180,120,0.14),
        inset 0 1px 0 rgba(255,255,255,0.22);
}
.insight .ico {
    font-size: 1.7rem;
    margin-bottom: 10px;
    filter: drop-shadow(0 4px 10px rgba(10,8,28,0.35));
    transition: transform 0.35s cubic-bezier(0.22,1,0.36,1);
}
.insight:hover .ico {
    transform: scale(1.12) rotate(-4deg);
}
.insight h4 {
    margin: 0 0 7px 0;
    font-size: 0.98rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ffd78a, #ffcc70, #c3a6ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 6s linear infinite;
}
.insight p {
    margin: 0;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #e8dfcc !important;
}

/* ============================================================
   ALERTS — glass with themed edge
   ============================================================ */
div[data-testid="stAlert"] {
    border-radius: 16px !important;
    border: 1px solid rgba(255,204,112,0.30) !important;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    background: linear-gradient(
        135deg,
        rgba(255,220,160,0.12) 0%,
        rgba(255,255,255,0.04) 100%) !important;
    color: #fff4e0 !important;
    box-shadow:
        0 12px 32px rgba(10,8,28,0.45),
        inset 0 1px 0 rgba(255,255,255,0.14) !important;
}
div[data-testid="stAlertContainer"] {
    color: #fff4e0 !important;
}
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
    color: #fff4e0 !important;
}

/* ============================================================
   FOOTER — elegant with breathing glow dot
   ============================================================ */
.app-footer {
    text-align: center;
    padding: 22px 10px 8px 10px;
    font-size: 0.80rem;
    color: #c9c0ae !important;
    letter-spacing: 0.04em;
    border-top: 1px solid rgba(255,255,255,0.10);
    margin-top: 24px;
    position: relative;
    z-index: 2;
}
.app-footer .dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: linear-gradient(135deg, #ffcc70, #ff9a76);
    border-radius: 50%;
    margin: 0 9px;
    vertical-align: middle;
    box-shadow: 0 0 12px rgba(255,204,112,0.85), 0 0 24px rgba(255,154,118,0.45);
    animation: dotBreathe 2.8s ease-in-out infinite;
}
@keyframes dotBreathe {
    0%,100% { transform: scale(1);   box-shadow: 0 0 10px rgba(255,204,112,0.70), 0 0 20px rgba(255,154,118,0.35); }
    50%     { transform: scale(1.5); box-shadow: 0 0 18px rgba(255,204,112,1.0),  0 0 36px rgba(255,154,118,0.55); }
}

/* ============================================================
   FORM-COLUMN accent banners
   ============================================================ */
.banner-order {
    padding: 12px 16px;
    border-radius: 14px;
    background: linear-gradient(90deg,
        rgba(255,204,112,0.18) 0%,
        rgba(255,204,112,0.04) 80%,
        transparent 100%);
    border-left: 3.5px solid #ffcc70;
    margin-bottom: 18px;
    box-shadow: 0 4px 14px rgba(10,8,28,0.25), inset 0 1px 0 rgba(255,255,255,0.06);
}
.banner-order b { color: #ffcc70 !important; text-shadow: 0 0 12px rgba(255,204,112,0.35); }
.banner-logis {
    padding: 12px 16px;
    border-radius: 14px;
    background: linear-gradient(90deg,
        rgba(195,166,255,0.18) 0%,
        rgba(120,200,255,0.06) 60%,
        transparent 100%);
    border-left: 3.5px solid #c3a6ff;
    margin-bottom: 18px;
    box-shadow: 0 4px 14px rgba(10,8,28,0.25), inset 0 1px 0 rgba(255,255,255,0.06);
}
.banner-logis b { color: #c3a6ff !important; text-shadow: 0 0 12px rgba(195,166,255,0.35); }
.tip-banner {
    margin-bottom: 12px;
    font-size: 0.92rem;
    color: #ede4d3 !important;
    padding: 12px 14px;
    background: linear-gradient(90deg,
        rgba(120,200,255,0.14) 0%,
        rgba(255,204,112,0.06) 50%,
        transparent 100%);
    border-left: 3px solid #78c8ff;
    border-radius: 12px;
    box-shadow: 0 3px 12px rgba(10,8,28,0.22), inset 0 1px 0 rgba(255,255,255,0.06);
}
.tip-banner b { color: #78c8ff !important; }

/* ============================================================
   MISC — hr + chrome hide + layout z-context
   ============================================================ */
hr {
    border: none !important;
    height: 1px;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(255,204,112,0.35) 25%,
        rgba(195,166,255,0.30) 50%,
        rgba(120,200,255,0.30) 75%,
        transparent 100%);
    margin: 28px 0 !important;
}
[data-testid="stDecoration"],
#MainMenu,
footer,
header {
    visibility: hidden !important;
}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 1.2rem !important;
    max-width: 1240px !important;
    position: relative;
    z-index: 2;
}

/* Tooltip help popovers */
div[data-baseweb="tooltip"] {
    backdrop-filter: blur(16px);
}
</style>
""",
)

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
md(
    """
<div class="glass-card hero-card">
<div style="display:flex; align-items:flex-start; gap:20px; position:relative; z-index:2;">
<div style="font-size:4rem; filter: drop-shadow(0 6px 20px rgba(255,204,112,0.55));">📦</div>
<div style="flex:1;">
<div class="hero-headline">Smart Delivery Time Predictor</div>
<p class="hero-sub">
Enter your order &amp; logistics details to get an accurate, ML-driven
delivery estimate. Powered by a trained regression pipeline
(<b>R² = 0.948</b>) so ops &amp; product teams can set realistic
customer windows, spot at-risk orders, and optimise staffing before
deliveries go late.
</p>
<div class="stat-row">
<span class="stat-pill gold">🎯 94.8% R² Accuracy</span>
<span class="stat-pill caramel">⏱️ ±3.18 hrs MAE</span>
<span class="stat-pill clay">📦 2,500 Orders Trained</span>
<span class="stat-pill mocha">🔮 Live Inference</span>
</div>
</div>
</div>
</div>
""",
)

# ---------------------------------------------------------------------------
# Model + data loading (cached)
# ---------------------------------------------------------------------------
MODEL_PATH = "model.pkl"
DATA_PATH = "data.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_categories():
    df = pd.read_csv(DATA_PATH)
    return sorted(df["Product_Category"].dropna().unique().tolist())


try:
    pipeline = load_model()
except FileNotFoundError:
    st.error(
        f"Could not find `{MODEL_PATH}`. Train the model first by running "
        f"`model_training.ipynb` end to end (it saves `model.pkl`), then "
        f"restart this app."
    )
    st.stop()

try:
    product_categories = load_categories()
except FileNotFoundError:
    st.warning(f"Could not find `{DATA_PATH}` — using a default category list.")
    product_categories = [
        "Apparel", "Books", "Electronics", "Health & Beauty",
        "Home & Kitchen", "Office Supplies", "Sports & Outdoors", "Toys & Games",
    ]

SHIPPING_MODES = ["Standard", "Express", "Economy", "Same Day"]

# ---------------------------------------------------------------------------
# Input form — glass card
# ---------------------------------------------------------------------------
md(
    """
<div class="section-title">
<div class="ribbon"></div>
<h2>Order &amp; Logistics Details</h2>
</div>
""",
)

md('<div class="glass-card">')

with st.form("prediction_form"):
    md(
        """
<div class="tip-banner">
💡 <b>Tip:</b> adjust the Courier Load &amp; Traffic sliders to simulate
peak-period conditions like holiday surges or bad-weather days.
</div>
""",
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        md(
            """
<div class="banner-order">
<b>📦 Order Attributes</b>
</div>
""",
        )
        product_category = st.selectbox("Product Category", product_categories)
        shipping_mode = st.selectbox("Shipping Mode", SHIPPING_MODES)
        order_value = st.number_input(
            "Order Value ($)", min_value=0.0, value=100.0, step=1.0,
            help="Total order value in USD (excl. shipping & taxes)"
        )
        package_weight = st.number_input(
            "Package Weight (Kg)", min_value=0.0, value=2.0, step=0.1,
            help="Gross weight of the shipped package in kilograms"
        )
        warehouse_distance = st.number_input(
            "Warehouse Distance (Km)", min_value=0.0, value=250.0, step=1.0,
            help="Distance from warehouse to delivery address in kilometres"
        )

    with col2:
        md(
            """
<div class="banner-logis">
<b>🚚 Logistics Conditions</b>
</div>
""",
        )
        items_in_order = st.number_input(
            "Items in Order", min_value=1, value=2, step=1, format="%d",
            help="Total line items in the shipment (affects picking time)"
        )
        processing_hours = st.number_input(
            "Warehouse Processing (hrs)", min_value=0.0, value=6.0, step=0.5,
            help="Hours from order received to package dispatched from warehouse"
        )
        courier_load = st.slider(
            "Courier Load Index", min_value=0.5, max_value=1.5, value=1.0, step=0.01,
            help="1.0 = normal load; higher = couriers are busier"
        )
        traffic_index = st.slider(
            "Traffic Index", min_value=0.5, max_value=2.0, value=1.2, step=0.01,
            help="1.0 = normal traffic; higher = slower last-mile"
        )

    submitted = st.form_submit_button("✨ Predict Delivery Time", use_container_width=True)

md('</div>')

# ---------------------------------------------------------------------------
# Validation + prediction
# ---------------------------------------------------------------------------
if submitted:
    errors = []
    if order_value < 0:
        errors.append("Order Value cannot be negative.")
    if package_weight < 0:
        errors.append("Package Weight cannot be negative.")
    if warehouse_distance < 0:
        errors.append("Warehouse Distance cannot be negative.")
    if items_in_order < 1:
        errors.append("Items in Order must be at least 1.")
    if processing_hours < 0:
        errors.append("Warehouse Processing cannot be negative.")
    if courier_load <= 0:
        errors.append("Courier Load Index must be positive.")
    if traffic_index <= 0:
        errors.append("Traffic Index must be positive.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        input_row = pd.DataFrame([{
            "Order Value": order_value,
            "Package_Weight_Kg": package_weight,
            "Warehouse Distance Km": warehouse_distance,
            "Items in Order": items_in_order,
            "Warehouse_Processing_Hours": processing_hours,
            "Courier_Load_Index": courier_load,
            "Traffic Index": traffic_index,
            "Product_Category": product_category,
            "Shipping_Mode": shipping_mode,
        }])

        pred_log_hours = pipeline.predict(input_row)[0]
        pred_hours = float(np.exp(pred_log_hours))
        days = pred_hours / 24

        # --- premium result display ---
        md(
            f"""
<div class="glass-card result-card">
<div style="display:flex; align-items:center; gap:14px; margin-bottom:4px;">
<div style="font-size:2.6rem;">🎯</div>
<div>
<div class="result-chip">Estimated Delivery Window</div>
<div class="result-context">
{shipping_mode} shipping · {product_category} · {warehouse_distance:,.0f} km
</div>
</div>
</div>
<div style="text-align:center; margin:22px 0 14px 0; position:relative; z-index:1;">
<div class="result-big">{pred_hours:,.1f}</div>
<div class="result-unit">HOURS</div>
<div class="result-days">≈ {days:,.1f} days</div>
</div>
</div>
""",
        )

        # --- ops / pm insights grid ---
        md(
            """
<div class="section-title">
<div class="ribbon"></div>
<h2>What This Means for Ops &amp; Product Teams</h2>
</div>
""",
        )

        md(
            f"""
<div class="insights-grid">
<div class="insight">
<div class="ico">💬</div>
<h4>Customer Communication</h4>
<p>Quote a <b>{pred_hours:.0f}-hour</b> window (≈ {days:.1f} days)
instead of a generic default. Transparency = fewer WISMO tickets
and higher trust.</p>
</div>
<div class="insight">
<div class="ico">👷</div>
<h4>Staffing Signals</h4>
<p>If similar lanes keep coming back above this baseline, review
warehouse scheduling or courier routes before the backlog hits.</p>
</div>
<div class="insight">
<div class="ico">🚨</div>
<h4>Exception Handling</h4>
<p>Orders where the estimate is way above the <b>{shipping_mode}</b>
average are candidates for manual review, courier reassignment or
proactive outreach.</p>
</div>
<div class="insight">
<div class="ico">📈</div>
<h4>Week-over-week Trends</h4>
<p>Compare today's estimate against the same day last week —
spikes in Courier Load ({courier_load:.2f}) or Traffic
({traffic_index:.2f}) explain the delta.</p>
</div>
</div>
""",
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
md(
    """
<div class="app-footer">
<span class="dot"></span>
Model: scikit-learn Pipeline (StandardScaler + OneHotEncoder → LinearRegression)
· trained on <b>log(Delivery_Time_Hours)</b>
· R² = 0.948 on 20% hold-out
<span class="dot"></span>
<br>Built for a university ML assignment — for experimental use,
not a production forecasting engine.
</div>
""",
)
