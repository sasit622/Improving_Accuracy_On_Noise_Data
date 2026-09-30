import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_squared_error, r2_score, silhouette_score
)
from sklearn.impute import SimpleImputer

# ─── TensorFlow / Keras (ANN + RNN) ────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    # Suppress TF logs
    tf.get_logger().setLevel("ERROR")
    import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
except ImportError:
    TF_AVAILABLE = False

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data-Centric Neural Optimizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
h1,h2,h3 { font-family:'Syne',sans-serif; font-weight:800; }

.hero-title {
    font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800;
    background:linear-gradient(135deg,#00f5a0,#00d9f5,#a78bfa);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; line-height:1.2; margin-bottom:0.4rem;
}
.hero-sub {
    font-family:'Space Mono',monospace; font-size:0.8rem; color:#6b7280;
    letter-spacing:0.15em; text-transform:uppercase; margin-bottom:2rem;
}
.metric-card {
    background:linear-gradient(135deg,#13131f 0%,#1a1a2e 100%);
    border:1px solid #2a2a4a; border-radius:16px; padding:1.5rem;
    text-align:center; position:relative; overflow:hidden;
}
.metric-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#00f5a0,#00d9f5);
}
.metric-label {
    font-family:'Space Mono',monospace; font-size:0.7rem; color:#6b7280;
    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;
}
.metric-value { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; color:#00f5a0; }
.metric-value.accent { color:#a78bfa; }
.metric-value.warn   { color:#f59e0b; }
.metric-value.bad    { color:#ef4444; }

.section-header {
    font-family:'Space Mono',monospace; font-size:0.72rem; color:#00d9f5;
    letter-spacing:0.2em; text-transform:uppercase; border-left:3px solid #00d9f5;
    padding-left:0.75rem; margin:2rem 0 1rem;
}
.feature-row {
    background:#13131f; border:1px solid #1e1e38; border-radius:10px;
    padding:0.7rem 1rem; margin-bottom:0.4rem;
    font-family:'Space Mono',monospace; font-size:0.8rem;
}
.badge {
    display:inline-block; padding:0.2rem 0.6rem; border-radius:6px;
    font-family:'Space Mono',monospace; font-size:0.68rem; font-weight:700; letter-spacing:0.05em;
}
.badge-num  { background:#0d3d2e; color:#00f5a0; border:1px solid #00f5a044; }
.badge-cat  { background:#2d1b69; color:#a78bfa; border:1px solid #a78bfa44; }
.badge-bool { background:#3d2000; color:#f59e0b; border:1px solid #f59e0b44; }

.compare-box   { border-radius:16px; padding:1.5rem; text-align:center; }
.compare-before { background:linear-gradient(135deg,#1f1020,#2d1b2e); border:1px solid #ef444444; }
.compare-after  { background:linear-gradient(135deg,#0d1f15,#0a1f2a); border:1px solid #00f5a044; }
.compare-title  { font-family:'Space Mono',monospace; font-size:0.72rem; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:1rem; }
.compare-stat   { font-size:2.4rem; font-weight:800; font-family:'Syne',sans-serif; }

.info-pill {
    display:inline-block; background:#13131f; border:1px solid #2a2a4a;
    border-radius:20px; padding:0.3rem 0.9rem;
    font-family:'Space Mono',monospace; font-size:0.75rem; color:#9ca3af; margin:0.2rem;
}
.model-section {
    background:linear-gradient(135deg,#0e1420,#131a2e);
    border:1px solid #1e2a4a; border-radius:16px; padding:1.25rem; margin:1rem 0 0.5rem;
}
.model-tag {
    display:inline-block; background:#0a1628; border:1px solid #00d9f544;
    color:#00d9f5; border-radius:8px; padding:0.25rem 0.75rem;
    font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:0.08em; margin:0.2rem;
}
.outlier-config-box {
    background:linear-gradient(135deg,#0e1a14,#0a1520);
    border:1px solid #00f5a033; border-radius:16px; padding:1.5rem; margin:1rem 0;
}
.outlier-config-title {
    font-family:'Space Mono',monospace; font-size:0.72rem; color:#00f5a0;
    letter-spacing:0.2em; text-transform:uppercase; margin-bottom:1rem;
    display:flex; align-items:center; gap:0.5rem;
}
.feature-chip {
    display:inline-block; background:#0d3d2e; border:1px solid #00f5a033;
    color:#00f5a0; border-radius:20px; padding:0.25rem 0.8rem;
    font-family:'Space Mono',monospace; font-size:0.7rem; margin:0.2rem;
}
.warning-box {
    background:linear-gradient(135deg,#1f1500,#2d1f00);
    border:1px solid #f59e0b44; border-radius:12px; padding:1rem 1.25rem; margin:0.75rem 0;
    font-family:'Space Mono',monospace; font-size:0.75rem; color:#f59e0b;
}
.neural-badge {
    background:linear-gradient(135deg,#1a0a2e,#0a1a2e);
    border:1px solid #a78bfa66; border-radius:8px; padding:0.25rem 0.75rem;
    font-family:'Space Mono',monospace; font-size:0.7rem; color:#a78bfa; margin:0.2rem;
    display:inline-block;
}
.stButton > button {
    background:linear-gradient(135deg,#00f5a0,#00d9f5) !important;
    color:#0a0a0f !important; font-family:'Syne',sans-serif !important;
    font-weight:700 !important; border:none !important; border-radius:10px !important;
    padding:0.6rem 2rem !important; letter-spacing:0.05em !important;
    font-size:0.95rem !important; transition:opacity 0.2s !important;
}
.stButton > button:hover { opacity:0.85 !important; }
div[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid #2a2a4a; }
.stProgress > div > div > div { background:linear-gradient(90deg,#00f5a0,#00d9f5) !important; }
hr { border-color:#1e1e38 !important; }
.sidebar-label { font-family:'Space Mono',monospace; font-size:0.7rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.1em; }
</style>
""", unsafe_allow_html=True)


# ─── ANN / RNN Builders ─────────────────────────────────────────────────────────

def build_ann(input_dim, output_dim, task):
    """Fully-connected feedforward ANN."""
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    if task == "classification":
        act = "sigmoid" if output_dim == 1 else "softmax"
        out = layers.Dense(output_dim if output_dim > 1 else 1, activation=act)(x)
        loss = "binary_crossentropy" if output_dim == 1 else "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    else:  # regression
        out = layers.Dense(1)(x)
        loss = "mse"
        metrics = ["mae"]
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss=loss, metrics=metrics)
    return model


def build_rnn(input_dim, output_dim, task):
    """LSTM-based RNN (input reshaped as sequence of length 1 per feature)."""
    inp = keras.Input(shape=(input_dim, 1))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    if task == "classification":
        act = "sigmoid" if output_dim == 1 else "softmax"
        out = layers.Dense(output_dim if output_dim > 1 else 1, activation=act)(x)
        loss = "binary_crossentropy" if output_dim == 1 else "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    else:
        out = layers.Dense(1)(x)
        loss = "mse"
        metrics = ["mae"]
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss=loss, metrics=metrics)
    return model


def run_keras_classification(X, y, arch="ann", epochs=20, batch_size=32):
    if X is None or y is None or len(X) < 20:
        return None
    try:
        # Normalize labels to [0, n_classes-1] for Keras sparse_categorical_crossentropy
        le = LabelEncoder()
        y = le.fit_transform(y)
        classes = le.classes_
        n_classes = len(classes)
        
        strat = y if n_classes > 1 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)

        # Scale
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)

        output_dim = 1 if n_classes == 2 else n_classes
        input_dim = Xtr_s.shape[1]

        if arch == "ann":
            model = build_ann(input_dim, output_dim, "classification")
            Xtr_in, Xte_in = Xtr_s, Xte_s
        else:  # rnn
            model = build_rnn(input_dim, output_dim, "classification")
            Xtr_in = Xtr_s.reshape(Xtr_s.shape[0], Xtr_s.shape[1], 1)
            Xte_in = Xte_s.reshape(Xte_s.shape[0], Xte_s.shape[1], 1)

        model.fit(Xtr_in, ytr, epochs=epochs, batch_size=batch_size, verbose=0,
                  validation_split=0.1)

        raw_preds = model.predict(Xte_in, verbose=0)
        if output_dim == 1:
            preds = (raw_preds.flatten() >= 0.5).astype(int)
        else:
            preds = np.argmax(raw_preds, axis=1)

        avg = "binary" if n_classes == 2 else "weighted"
        return {
            "accuracy": accuracy_score(yte, preds),
            "f1": f1_score(yte, preds, average=avg, zero_division=0),
            "report": classification_report(yte, preds, target_names=[str(c) for c in classes], zero_division=0),
            "predictions": le.inverse_transform(preds[:20]),
            "actuals": le.inverse_transform(yte[:20]),
        }
    except Exception as e:
        return {"error": str(e)}


def run_keras_regression(X, y, arch="ann", epochs=30, batch_size=32):
    if X is None or y is None or len(X) < 20:
        return None
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)

        input_dim = Xtr_s.shape[1]

        if arch == "ann":
            model = build_ann(input_dim, 1, "regression")
            Xtr_in, Xte_in = Xtr_s, Xte_s
        else:
            model = build_rnn(input_dim, 1, "regression")
            Xtr_in = Xtr_s.reshape(Xtr_s.shape[0], Xtr_s.shape[1], 1)
            Xte_in = Xte_s.reshape(Xte_s.shape[0], Xte_s.shape[1], 1)

        model.fit(Xtr_in, ytr, epochs=epochs, batch_size=batch_size, verbose=0,
                  validation_split=0.1)
        preds = model.predict(Xte_in, verbose=0).flatten()
        mse = mean_squared_error(yte, preds)
        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2": r2_score(yte, preds),
            "predictions": preds[:20],
            "actuals": yte[:20],
        }
    except Exception as e:
        return {"error": str(e)}


# ─── ML Model Registry ──────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "Random Forest (Classification)":  {"task":"classification","cls":RandomForestClassifier,    "params":{"n_estimators":100,"random_state":42,"n_jobs":-1},"emoji":"🌲","backend":"sklearn"},
    "Logistic Regression":             {"task":"classification","cls":LogisticRegression,         "params":{"max_iter":1000,"random_state":42},"emoji":"📈","backend":"sklearn"},
    "Decision Tree":                   {"task":"classification","cls":DecisionTreeClassifier,     "params":{"random_state":42},"emoji":"🌳","backend":"sklearn"},
    "Support Vector Machine (SVM)":   {"task":"classification","cls":SVC,                        "params":{"kernel":"rbf","random_state":42,"probability":True},"emoji":"⚡","backend":"sklearn"},
    "K-Nearest Neighbors (KNN)":      {"task":"classification","cls":KNeighborsClassifier,       "params":{"n_neighbors":5},"emoji":"🔵","backend":"sklearn"},
    "Gradient Boosting":               {"task":"classification","cls":GradientBoostingClassifier, "params":{"n_estimators":100,"random_state":42},"emoji":"🚀","backend":"sklearn"},
    "AdaBoost":                        {"task":"classification","cls":AdaBoostClassifier,         "params":{"n_estimators":100,"random_state":42},"emoji":"🔥","backend":"sklearn"},
    "Naive Bayes":                     {"task":"classification","cls":GaussianNB,                 "params":{},"emoji":"🧮","backend":"sklearn"},
    # ── NEW: Neural Networks ──────────────────────────────────────────────────
    "ANN (Neural Network)":            {"task":"classification","cls":None,"params":{"arch":"ann","epochs":20,"batch_size":32},"emoji":"🧠","backend":"keras"},
    "RNN / LSTM":                      {"task":"classification","cls":None,"params":{"arch":"rnn","epochs":20,"batch_size":32},"emoji":"🔁","backend":"keras"},
    # ── Regression ───────────────────────────────────────────────────────────
    "Linear Regression":               {"task":"regression",    "cls":LinearRegression,           "params":{},"emoji":"📉","backend":"sklearn"},
    "Ridge Regression":                {"task":"regression",    "cls":Ridge,                      "params":{"alpha":1.0},"emoji":"🏔️","backend":"sklearn"},
    "Lasso Regression":                {"task":"regression",    "cls":Lasso,                      "params":{"alpha":1.0,"max_iter":5000},"emoji":"🎯","backend":"sklearn"},
    "ANN Regression":                  {"task":"regression",    "cls":None,"params":{"arch":"ann","epochs":30,"batch_size":32},"emoji":"🧠","backend":"keras"},
    "RNN / LSTM Regression":           {"task":"regression",    "cls":None,"params":{"arch":"rnn","epochs":30,"batch_size":32},"emoji":"🔁","backend":"keras"},
    # ── Clustering ───────────────────────────────────────────────────────────
    "K-Means Clustering":              {"task":"clustering",    "cls":KMeans,                     "params":{"n_clusters":3,"random_state":42,"n_init":10},"emoji":"🔮","backend":"sklearn"},
}

# ─── Helpers ────────────────────────────────────────────────────────────────────

def clean_string_column(series):
    def cv(v):
        if pd.isna(v): return v
        v = re.sub(r"[^a-zA-Z0-9\s_\-]", "", str(v)).strip()
        return v if v else np.nan
    return series.apply(cv)

def detect_column_types(df):
    types = {}
    for col in df.columns:
        if df[col].dtype == bool or (df[col].nunique() == 2 and set(df[col].dropna().unique()).issubset({0,1,True,False,"0","1"})):
            types[col] = "boolean"
        elif pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        else:
            types[col] = "categorical"
    return types

def preprocess_raw(df):
    df = df.copy()
    col_types = detect_column_types(df)
    for col, ctype in col_types.items():
        if ctype == "categorical":
            df[col] = clean_string_column(df[col])
        elif ctype == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, col_types

def remove_outliers_kmeans(df, col_types, outlier_feature_cols, n_clusters=3):
    valid_cols = [c for c in outlier_feature_cols if c in df.columns and col_types.get(c) == "numeric"]
    if not valid_cols:
        valid_cols = [c for c, t in col_types.items() if t == "numeric"]
    if not valid_cols:
        return df, pd.Series([False] * len(df), index=df.index)
    sub = df[valid_cols].copy()
    imp = SimpleImputer(strategy="median")
    sub_imp = pd.DataFrame(imp.fit_transform(sub), columns=valid_cols, index=df.index)
    scaled = StandardScaler().fit_transform(sub_imp)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    sizes = pd.Series(labels).value_counts(normalize=True)
    small = sizes[sizes < 0.05].index.tolist()
    mask = pd.Series([l in small for l in labels], index=df.index)
    return df[~mask].copy(), mask

def encode_for_model(df, col_types, target_col):
    df = df.copy()
    for col, ctype in col_types.items():
        if ctype == "categorical":
            le = LabelEncoder()
            df[col] = df[col].fillna("__missing__")
            df[col] = le.fit_transform(df[col].astype(str))
        elif ctype == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    if target_col not in df.columns or len(df) < 10:
        return None, None
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if y.dtype == object or str(y.dtype) == "category":
        y = LabelEncoder().fit_transform(y.astype(str))
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    return X, np.array(y)

def run_classification(X, y, model_cls, params):
    if X is None or y is None or len(X) < 10: return None
    try:
        le = LabelEncoder()
        y = le.fit_transform(y)
        classes = le.classes_
        n_classes = len(classes)
        
        strat = y if n_classes > 1 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)
        clf = model_cls(**params)
        clf.fit(Xtr, ytr)
        preds = clf.predict(Xte)
        avg = "binary" if n_classes == 2 else "weighted"
        return {
            "accuracy": accuracy_score(yte, preds),
            "f1": f1_score(yte, preds, average=avg, zero_division=0),
            "report": classification_report(yte, preds, target_names=[str(c) for c in classes], zero_division=0),
            "predictions": le.inverse_transform(preds[:20]), 
            "actuals": le.inverse_transform(yte[:20])
        }
    except Exception as e:
        return {"error": str(e)}

def run_regression(X, y, model_cls, params):
    if X is None or y is None or len(X) < 10: return None
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
        reg = model_cls(**params)
        reg.fit(Xtr, ytr)
        preds = reg.predict(Xte)
        mse = mean_squared_error(yte, preds)
        return {"mse": mse, "rmse": np.sqrt(mse), "r2": r2_score(yte, preds),
                "predictions": preds[:20], "actuals": yte[:20]}
    except Exception as e:
        return {"error": str(e)}

def run_clustering(X, params):
    if X is None or len(X) < 10: return None
    try:
        Xs = StandardScaler().fit_transform(X)
        km = KMeans(**params)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels) if len(np.unique(labels)) > 1 else 0.0
        counts = pd.Series(labels).value_counts().sort_index()
        return {"inertia": km.inertia_, "silhouette": sil,
                "n_clusters": params.get("n_clusters", 3),
                "cluster_counts": counts.to_dict(), "labels": labels[:50]}
    except Exception as e:
        return {"error": str(e)}


def run_model_on_data(model_name, X, y, ann_epochs=20, rnn_epochs=20, batch_size=32):
    info = MODEL_REGISTRY[model_name]
    backend = info.get("backend", "sklearn")

    # ── Keras / TF models ────────────────────────────────────────────────────
    if backend == "keras":
        if not TF_AVAILABLE:
            return {"error": "TensorFlow is not installed. Run: pip install tensorflow"}
        arch = info["params"]["arch"]
        ep = ann_epochs if arch == "ann" else rnn_epochs
        if info["task"] == "classification":
            return run_keras_classification(X, y, arch=arch, epochs=ep, batch_size=batch_size)
        elif info["task"] == "regression":
            return run_keras_regression(X, y, arch=arch, epochs=ep, batch_size=batch_size)

    # ── Sklearn models ───────────────────────────────────────────────────────
    if info["task"] == "classification": return run_classification(X, y, info["cls"], info["params"])
    if info["task"] == "regression":     return run_regression(X, y, info["cls"], info["params"])
    if info["task"] == "clustering":     return run_clustering(X, info["params"])


def render_result_card(result, label, is_before=True):
    if result is None:
        st.warning("Could not run model — not enough data or invalid configuration.")
        return
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return

    tc = "#ef4444" if is_before else "#00f5a0"
    ic = "⚠" if is_before else "✅"
    cls_box = "compare-before" if is_before else "compare-after"

    if "accuracy" in result:
        st.markdown(f"""
        <div class="compare-box {cls_box}">
            <div class="compare-title" style="color:{tc};">{ic} {label}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                <div><div class="metric-label">Accuracy</div>
                     <div class="compare-stat" style="color:{tc};">{result['accuracy']*100:.2f}%</div></div>
                <div><div class="metric-label">F1 Score</div>
                     <div class="compare-stat" style="color:{tc};">{result['f1']:.4f}</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
        with st.expander(f"📋 Classification Report — {label}"):
            st.code(result["report"], language="")
        with st.expander(f"🔮 Sample Predictions — {label} (first 20 rows)"):
            pdf = pd.DataFrame({"Actual": result["actuals"], "Predicted": result["predictions"]})
            pdf["✓ Match"] = pdf["Actual"] == pdf["Predicted"]
            st.dataframe(pdf, use_container_width=True)

    elif "r2" in result:
        st.markdown(f"""
        <div class="compare-box {cls_box}">
            <div class="compare-title" style="color:{tc};">{ic} {label}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;">
                <div><div class="metric-label">R² Score</div>
                     <div class="compare-stat" style="color:{tc};">{result['r2']:.4f}</div></div>
                <div><div class="metric-label">RMSE</div>
                     <div class="compare-stat" style="color:{tc};">{result['rmse']:.4f}</div></div>
                <div><div class="metric-label">MSE</div>
                     <div class="compare-stat" style="color:{tc};">{result['mse']:.4f}</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
        with st.expander(f"🔮 Sample Predictions — {label} (first 20 rows)"):
            pdf = pd.DataFrame({"Actual": result["actuals"], "Predicted": result["predictions"]})
            pdf["Abs Error"] = (pdf["Actual"] - pdf["Predicted"]).abs()
            st.dataframe(pdf, use_container_width=True)

    elif "silhouette" in result:
        counts_str = " | ".join([f"C{k}: {v}" for k,v in result["cluster_counts"].items()])
        st.markdown(f"""
        <div class="compare-box {cls_box}">
            <div class="compare-title" style="color:{tc};">{ic} {label}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;">
                <div><div class="metric-label">Silhouette</div>
                     <div class="compare-stat" style="color:{tc};">{result['silhouette']:.4f}</div></div>
                <div><div class="metric-label">Inertia</div>
                     <div class="compare-stat" style="color:{tc};">{result['inertia']:.1f}</div></div>
                <div><div class="metric-label">Clusters</div>
                     <div class="compare-stat" style="color:{tc};">{result['n_clusters']}</div></div>
            </div>
            <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#6b7280;margin-top:0.75rem;">{counts_str}</div>
        </div>""", unsafe_allow_html=True)
        with st.expander(f"🔮 Cluster Labels — {label} (first 50 rows)"):
            st.dataframe(pd.DataFrame({"Row": range(len(result["labels"])), "Cluster": result["labels"]}), use_container_width=True)


# ─── Main App ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 Data-Centric Neural Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">An Empirical Study · Noisy Dataset Refinement · Multi-Model Benchmarking</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-label">⚙ Configuration</div>', unsafe_allow_html=True)
    st.markdown("---")
    n_clusters_outlier = st.slider("KMeans Outlier Clusters", 2, 10, 3,
                                   help="Number of clusters used during outlier detection")
    st.markdown("---")

    # ── Neural Network Hyperparameters ───────────────────────────────────────
    st.markdown('<div class="sidebar-label">🧠 Neural Network Hyperparams</div>', unsafe_allow_html=True)
    ann_epochs = st.slider("ANN Epochs", 5, 100, 20,
                           help="Training epochs for ANN (classification & regression)")
    rnn_epochs = st.slider("RNN/LSTM Epochs", 5, 100, 20,
                           help="Training epochs for RNN/LSTM (classification & regression)")
    nn_batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128, 256], value=32)

    # TF availability indicator
    if TF_AVAILABLE:
        st.markdown('<div style="font-family:Space Mono,monospace;font-size:0.68rem;color:#00f5a0;margin-top:0.5rem;">✅ TensorFlow detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-family:Space Mono,monospace;font-size:0.68rem;color:#ef4444;margin-top:0.5rem;">⚠ TensorFlow not found — ANN/RNN unavailable</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-label">📌 Pipeline Steps</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.72rem;color:#6b7280;line-height:2.2;'>
    1. Upload CSV / Excel<br>
    2. Inspect features &amp; types<br>
    3. Select target column<br>
    4. Choose ML models<br>
    5. Select outlier detection features<br>
    6. KMeans outlier removal<br>
    7. Compare predictions &amp; metrics
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sidebar-label">🤖 Model Catalogue</div>', unsafe_allow_html=True)
    for name, info in MODEL_REGISTRY.items():
        c = {"classification":"#00f5a0","regression":"#a78bfa","clustering":"#f59e0b"}[info["task"]]
        backend_label = " <span style='color:#a78bfa;'>[TF]</span>" if info.get("backend") == "keras" else ""
        st.markdown(
            f"<div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#9ca3af;margin:0.2rem 0;'>"
            f"{info['emoji']} <span style='color:{c};'>[{info['task'][:3].upper()}]</span> {name}{backend_label}</div>",
            unsafe_allow_html=True
        )

# Upload
st.markdown('<div class="section-header">01 · Upload Dataset</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop your CSV or Excel file", type=["csv","xlsx","xls"])

if not TF_AVAILABLE:
    st.markdown("""
    <div class="warning-box">
        ⚠ <strong>TensorFlow not installed.</strong> ANN and RNN models will return an error when selected.
        Install with: <code>pip install tensorflow</code>
    </div>""", unsafe_allow_html=True)

if uploaded:
    try:
        raw_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}"); st.stop()

    # Overview
    st.markdown('<div class="section-header">02 · Dataset Overview</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Dataset Name</div><div class="metric-value accent" style="font-size:1rem;">{uploaded.name}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Rows</div><div class="metric-value">{len(raw_df):,}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Features</div><div class="metric-value accent">{len(raw_df.columns)}</div></div>', unsafe_allow_html=True)
    with c4:
        missing = raw_df.isnull().sum().sum()
        mv_cls = "warn" if missing > 0 else ""
        st.markdown(f'<div class="metric-card"><div class="metric-label">Missing Values</div><div class="metric-value {mv_cls}">{missing:,}</div></div>', unsafe_allow_html=True)

    # Feature Details
    st.markdown('<div class="section-header">03 · Feature Details</div>', unsafe_allow_html=True)
    _, col_types = preprocess_raw(raw_df)
    feat_data = []
    for col in raw_df.columns:
        ctype = col_types.get(col,"categorical")
        feat_data.append({"Feature":col,"Type":ctype.capitalize(),
                          "Unique Values":raw_df[col].nunique(),
                          "Missing %":f"{raw_df[col].isnull().mean()*100:.1f}%",
                          "Sample":str(raw_df[col].dropna().iloc[0])[:25] if not raw_df[col].dropna().empty else "—"})
    feat_df = pd.DataFrame(feat_data)
    cols_d = st.columns(min(3,len(raw_df.columns)))
    for i, row in feat_df.iterrows():
        bc = "badge-num" if row["Type"]=="Numeric" else ("badge-bool" if row["Type"]=="Boolean" else "badge-cat")
        with cols_d[i%3]:
            st.markdown(f"""
            <div class="feature-row">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
                    <span style="color:#e8e8f0;font-weight:700;">{row['Feature']}</span>
                    <span class="badge {bc}">{row['Type']}</span>
                </div>
                <span class="info-pill">🔢 {row['Unique Values']} unique</span>
                <span class="info-pill">❌ {row['Missing %']} null</span>
                <span class="info-pill">e.g. {row['Sample']}</span>
            </div>""", unsafe_allow_html=True)
    with st.expander("📋 Feature Summary Table"): st.dataframe(feat_df, use_container_width=True)
    with st.expander("📊 Preview Raw Data (first 100 rows)"): st.dataframe(raw_df.head(100), use_container_width=True)

    # Model Configuration
    st.markdown('<div class="section-header">04 · Model Configuration</div>', unsafe_allow_html=True)
    cfg1, cfg2 = st.columns([1,2])
    with cfg1:
        target_col = st.selectbox("🎯 Target / Label Column", raw_df.columns.tolist(),
                                  help="Column to predict. Ignored for clustering.")
    with cfg2:
        selected_models = st.multiselect(
            "🤖 Select ML Models",
            list(MODEL_REGISTRY.keys()),
            default=["Random Forest (Classification)", "Logistic Regression", "ANN (Neural Network)", "RNN / LSTM", "K-Means Clustering"],
            help="Pick any combination. Each model runs on both raw and refined data."
        )

    if selected_models:
        badge_html = ""
        for m in selected_models:
            task = MODEL_REGISTRY[m]["task"]
            color = {"classification":"#00f5a0","regression":"#a78bfa","clustering":"#f59e0b"}[task]
            backend = MODEL_REGISTRY[m].get("backend","sklearn")
            b_tag = ' <span style="color:#a78bfa;font-size:0.6rem;">[TF]</span>' if backend == "keras" else ""
            badge_html += f'<span class="model-tag">{MODEL_REGISTRY[m]["emoji"]} {m}{b_tag} <span style="color:{color};">· {task}</span></span>'
        st.markdown(f'<div style="margin:0.5rem 0;">{badge_html}</div>', unsafe_allow_html=True)

    # Extra: k for clustering
    if any(MODEL_REGISTRY.get(m,{}).get("task") == "clustering" for m in selected_models):
        km_k = st.slider("K-Means Prediction Clusters", 2, 10, 3, key="km_pred_k")
        MODEL_REGISTRY["K-Means Clustering"]["params"]["n_clusters"] = km_k

    # Outlier Detection Feature Selection
    st.markdown('<div class="section-header">05 · KMeans Outlier Detection — Feature Selection</div>', unsafe_allow_html=True)
    numeric_cols_available = [col for col, t in col_types.items() if t == "numeric"]
    non_numeric_cols = [col for col, t in col_types.items() if t != "numeric"]

    st.markdown("""
    <div class="outlier-config-box">
        <div class="outlier-config-title">🔬 Choose Features for Cluster-Based Outlier Detection</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#6b7280;line-height:1.8;margin-bottom:1rem;">
            KMeans clustering will be run <strong style="color:#e8e8f0;">only on the features you select below</strong>
            to identify and remove outlier rows. Selecting domain-relevant numeric features
            gives more meaningful outlier detection than using all columns blindly.
            <br><br>
            ⚠ Only <strong style="color:#00f5a0;">numeric</strong> features can be used for KMeans distance calculations.
            Categorical and boolean columns are excluded automatically.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not numeric_cols_available:
        st.markdown('<div class="warning-box">⚠ No numeric columns found in the dataset. KMeans outlier removal will be skipped.</div>', unsafe_allow_html=True)
        outlier_feature_cols = []
    else:
        if non_numeric_cols:
            non_num_html = "".join([f'<span style="display:inline-block;background:#1a1020;border:1px solid #a78bfa33;color:#6b7280;border-radius:20px;padding:0.25rem 0.8rem;font-family:Space Mono,monospace;font-size:0.68rem;margin:0.2rem;text-decoration:line-through;">{c}</span>' for c in non_numeric_cols])
            st.markdown(f"""
            <div style="margin-bottom:0.75rem;">
                <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#6b7280;margin-bottom:0.4rem;text-transform:uppercase;letter-spacing:0.1em;">
                    ✗ Non-numeric (excluded from selection)
                </div>
                {non_num_html}
            </div>
            """, unsafe_allow_html=True)

        col_sel1, col_sel2 = st.columns([3, 1])
        with col_sel1:
            outlier_feature_cols = st.multiselect(
                "✅ Select numeric features for outlier clustering",
                options=numeric_cols_available,
                default=numeric_cols_available,
                help="KMeans will cluster rows based ONLY on these features to detect outliers."
            )
        with col_sel2:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            if st.button("🔄 Reset to All"):
                outlier_feature_cols = numeric_cols_available

        if outlier_feature_cols:
            chips_html = "".join([f'<span class="feature-chip">✓ {c}</span>' for c in outlier_feature_cols])
            st.markdown(f"""
            <div style="margin:0.75rem 0 0.5rem;">
                <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#00f5a0;margin-bottom:0.4rem;text-transform:uppercase;letter-spacing:0.1em;">
                    ✓ Selected for outlier detection ({len(outlier_feature_cols)} feature{'s' if len(outlier_feature_cols)!=1 else ''})
                </div>
                {chips_html}
            </div>
            """, unsafe_allow_html=True)
            with st.expander("📊 Stats Preview of Selected Outlier Features"):
                stats_df = raw_df[outlier_feature_cols].describe().T
                stats_df.index.name = "Feature"
                st.dataframe(stats_df.style.format("{:.3f}"), use_container_width=True)
        else:
            st.markdown('<div class="warning-box">⚠ No features selected. Outlier removal will be skipped.</div>', unsafe_allow_html=True)

    # Run
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    if st.button("🚀 Run Full Analysis"):
        if not selected_models:
            st.warning("Please select at least one model."); st.stop()

        with st.spinner("Running pipeline…"):
            prog = st.progress(0, text="Cleaning data…")
            processed_raw, col_types = preprocess_raw(raw_df)
            prog.progress(15, text="Encoding features…")
            X_raw, y_raw = encode_for_model(processed_raw, col_types, target_col)

            prog.progress(25, text="Dropping missing values…")
            rows_before_cleaning = len(processed_raw)
            df_no_na = processed_raw.dropna()
            n_missing_removed = rows_before_cleaning - len(df_no_na)

            prog.progress(40, text="KMeans outlier removal on selected features…")
            effective_outlier_cols = outlier_feature_cols if outlier_feature_cols else []
            if effective_outlier_cols:
                clean_df, outlier_mask = remove_outliers_kmeans(
                    df_no_na, col_types,
                    outlier_feature_cols=effective_outlier_cols,
                    n_clusters=n_clusters_outlier
                )
            else:
                clean_df = df_no_na.copy()
                outlier_mask = pd.Series([False] * len(df_no_na), index=df_no_na.index)

            n_outliers = int(outlier_mask.sum())
            X_clean, y_clean = encode_for_model(clean_df, col_types, target_col)

            raw_results, clean_results = {}, {}
            total = len(selected_models)
            for i, mn in enumerate(selected_models):
                prog.progress(50 + int(40*i/total), text=f"Training {mn}…")
                raw_results[mn]   = run_model_on_data(mn, X_raw,   y_raw,
                                                      ann_epochs=ann_epochs,
                                                      rnn_epochs=rnn_epochs,
                                                      batch_size=nn_batch_size)
                clean_results[mn] = run_model_on_data(mn, X_clean, y_clean,
                                                      ann_epochs=ann_epochs,
                                                      rnn_epochs=rnn_epochs,
                                                      batch_size=nn_batch_size)
            prog.progress(100, text="Done!"); prog.empty()

        # Refinement Stats
        st.markdown('<div class="section-header">06 · Refinement Analysis</div>', unsafe_allow_html=True)
        if effective_outlier_cols:
            used_chips = "".join([f'<span class="feature-chip">🔬 {c}</span>' for c in effective_outlier_cols])
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0a1520,#0d1f15);border:1px solid #00d9f533;
                        border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem;">
                <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#00d9f5;
                            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">
                    🔬 Outlier Detection Ran On These Features
                </div>
                {used_chips}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠ Outlier removal was skipped (no features selected).</div>', unsafe_allow_html=True)

        oc1,oc2,oc3,oc4 = st.columns(4)
        with oc1: st.markdown(f'<div class="metric-card"><div class="metric-label">Missing Rows Removed</div><div class="metric-value warn">{n_missing_removed:,}</div></div>', unsafe_allow_html=True)
        with oc2: st.markdown(f'<div class="metric-card"><div class="metric-label">Outliers Detected</div><div class="metric-value warn">{n_outliers:,}</div></div>', unsafe_allow_html=True)
        with oc3: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Rows After</div><div class="metric-value accent">{len(clean_df):,}</div></div>', unsafe_allow_html=True)
        with oc4:
            total_removed = n_missing_removed + n_outliers
            ref_pct = total_removed / max(len(processed_raw),1) * 100
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Refinement Impact</div><div class="metric-value warn">{ref_pct:.1f}%</div></div>', unsafe_allow_html=True)

        # Per-Model Results
        st.markdown('<div class="section-header">07 · Model-by-Model Comparison</div>', unsafe_allow_html=True)
        for mn in selected_models:
            task  = MODEL_REGISTRY[mn]["task"]
            emoji = MODEL_REGISTRY[mn]["emoji"]
            backend = MODEL_REGISTRY[mn].get("backend","sklearn")
            tc    = {"classification":"#00f5a0","regression":"#a78bfa","clustering":"#f59e0b"}[task]
            backend_badge = '<span class="neural-badge">TensorFlow · Keras</span>' if backend == "keras" else ""

            st.markdown(f"""
            <div class="model-section">
                <div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">
                    <span style="font-size:1.6rem;">{emoji}</span>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#e8e8f0;">{mn}</div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:{tc};text-transform:uppercase;letter-spacing:0.1em;">{task} model &nbsp; {backend_badge}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            col_b, col_a = st.columns(2)
            with col_b: render_result_card(raw_results[mn],   "Raw Dataset (Before Refinement)", is_before=True)
            with col_a: render_result_card(clean_results[mn], "Refined Dataset (After Refinement)", is_before=False)

            r, c = raw_results.get(mn) or {}, clean_results.get(mn) or {}
            if "accuracy" in r and "accuracy" in c:
                da = (c["accuracy"]-r["accuracy"])*100
                df1 = c["f1"]-r["f1"]
                d1,d2 = st.columns(2)
                with d1:
                    sc = "accent" if da >= 0 else "bad"
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Accuracy Δ (Refinement Impact)</div><div class="metric-value {sc}">{da:+.2f}%</div></div>', unsafe_allow_html=True)
                with d2:
                    sc = "accent" if df1 >= 0 else "bad"
                    st.markdown(f'<div class="metric-card"><div class="metric-label">F1 Score Δ</div><div class="metric-value {sc}">{df1:+.4f}</div></div>', unsafe_allow_html=True)
            elif "r2" in r and "r2" in c:
                dr2 = c["r2"]-r["r2"]; drmse = r["rmse"]-c["rmse"]
                d1,d2 = st.columns(2)
                with d1: st.markdown(f'<div class="metric-card"><div class="metric-label">R² Δ</div><div class="metric-value {"accent" if dr2>=0 else "bad"}">{dr2:+.4f}</div></div>', unsafe_allow_html=True)
                with d2: st.markdown(f'<div class="metric-card"><div class="metric-label">RMSE Δ (↓ better)</div><div class="metric-value {"accent" if drmse>=0 else "bad"}">{drmse:+.4f}</div></div>', unsafe_allow_html=True)
            elif "silhouette" in r and "silhouette" in c:
                ds = c["silhouette"]-r["silhouette"]
                st.markdown(f'<div class="metric-card" style="max-width:300px;"><div class="metric-label">Silhouette Δ</div><div class="metric-value {"accent" if ds>=0 else "bad"}">{ds:+.4f}</div></div>', unsafe_allow_html=True)

            st.markdown("---")

        # Summary Leaderboard
        st.markdown('<div class="section-header">08 · Summary Leaderboard</div>', unsafe_allow_html=True)
        rows = []
        for mn in selected_models:
            r = raw_results.get(mn) or {}; c = clean_results.get(mn) or {}
            row = {"Model": f'{MODEL_REGISTRY[mn]["emoji"]} {mn}', "Task": MODEL_REGISTRY[mn]["task"].capitalize()}
            if "accuracy" in r:
                row.update({"Raw Acc": f'{r["accuracy"]*100:.2f}%', "Clean Acc": f'{c.get("accuracy",0)*100:.2f}%',
                            "Raw F1": f'{r["f1"]:.4f}', "Clean F1": f'{c.get("f1",0):.4f}',
                            "Acc Δ": f'{(c.get("accuracy",r["accuracy"])-r["accuracy"])*100:+.2f}%'})
            elif "r2" in r:
                row.update({"Raw R²": f'{r["r2"]:.4f}', "Clean R²": f'{c.get("r2",0):.4f}',
                            "Raw RMSE": f'{r["rmse"]:.4f}', "Clean RMSE": f'{c.get("rmse",0):.4f}',
                            "R² Δ": f'{c.get("r2",r["r2"])-r["r2"]:+.4f}'})
            elif "silhouette" in r:
                row.update({"Raw Sil": f'{r["silhouette"]:.4f}', "Clean Sil": f'{c.get("silhouette",0):.4f}',
                            "Raw Inertia": f'{r["inertia"]:.1f}', "Clean Inertia": f'{c.get("inertia",0):.1f}'})
            rows.append(row)
        if rows: st.dataframe(pd.DataFrame(rows).fillna("—"), use_container_width=True)

        # Download
        st.markdown('<div class="section-header">09 · Download Refined Dataset</div>', unsafe_allow_html=True)
        with st.expander("👁 Preview Refined Dataset"): st.dataframe(clean_df.head(100), use_container_width=True)
        buf = io.BytesIO(); clean_df.to_csv(buf, index=False)
        st.download_button("⬇ Download Refined CSV", buf.getvalue(), "refined_dataset.csv", "text/csv")

else:
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;background:linear-gradient(135deg,#0d0d1a,#13131f);
         border:1px dashed #2a2a4a;border-radius:20px;margin-top:2rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">📂</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:#e8e8f0;margin-bottom:0.5rem;">
            Upload a Dataset to Begin
        </div>
        <div style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#6b7280;line-height:2;">
            CSV · Excel (.xlsx / .xls) · String cleaning · KMeans outlier removal · Multi-model comparison
        </div>
        <div style="margin-top:2rem;">
            <span class="model-tag">🌲 Random Forest</span><span class="model-tag">📈 Logistic Regression</span>
            <span class="model-tag">📉 Linear Regression</span><span class="model-tag">⚡ SVM</span>
            <span class="model-tag">🔵 KNN</span><span class="model-tag">🚀 Gradient Boosting</span>
            <span class="model-tag">🔮 K-Means</span><span class="model-tag">🌳 Decision Tree</span>
            <span class="model-tag">🧮 Naive Bayes</span><span class="model-tag">🔥 AdaBoost</span>
            <span class="model-tag">🏔️ Ridge</span><span class="model-tag">🎯 Lasso</span>
            <span class="neural-badge">🧠 ANN (Neural Network)</span>
            <span class="neural-badge">🔁 RNN / LSTM</span>
            <span class="neural-badge">🧠 ANN Regression</span>
            <span class="neural-badge">🔁 RNN / LSTM Regression</span>
        </div>
    </div>""", unsafe_allow_html=True)