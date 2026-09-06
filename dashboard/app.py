"""
WebLock - Fraud Analyst Dashboard
-------------------------------------
Streamlit app that lets a fraud analyst:
  1. See a feed of transactions with fraud scores
  2. Filter/sort by risk tier
  3. Click into any transaction to see WHY it was flagged (SHAP)
  4. Visualize the shared-entity network around a suspicious account

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import joblib
import os
import gc
import matplotlib.pyplot as plt
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

st.set_page_config(page_title="WebLock Fraud Dashboard", layout="wide")

# ---------------------------------------------------------------
# Load artifacts (cached so the app stays fast)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "graph_model.pkl"))
    explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "graph_features.pkl"))
    return model, explainer, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "transactions_with_graph_features.csv"))
    dummies = pd.get_dummies(df["merchant_category"], prefix="cat")
    df = pd.concat([df, dummies], axis=1)
    for col in feature_cols:
        if col.startswith("cat_") and col not in df.columns:
            df[col] = 0

    # Downcast numeric dtypes to roughly halve memory usage -- no meaningful
    # precision loss for this use case, but matters a lot on memory-limited
    # free-tier hosting.
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    int_cols = df.select_dtypes(include="int64").columns
    df[int_cols] = df[int_cols].astype("int32")

    # Cap the LIVE public demo to a representative sample rather than holding
    # all 590K rows in memory at once. The full-scale results are already
    # documented from the local pipeline run (see README) -- this only
    # limits what the hosted demo keeps resident in RAM, which is standard
    # practice for public demos of large-scale systems on free infrastructure.
    MAX_DEMO_ROWS = 150_000
    if len(df) > MAX_DEMO_ROWS:
        df = df.sample(MAX_DEMO_ROWS, random_state=42).reset_index(drop=True)

    return df

model, explainer, feature_cols = load_artifacts()

@st.cache_resource
def load_baseline_comparison():
    from sklearn.metrics import average_precision_score, roc_auc_score
    baseline_model = joblib.load(os.path.join(MODEL_DIR, "baseline_model.pkl"))
    tabular_features = joblib.load(os.path.join(MODEL_DIR, "tabular_features.pkl"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))["is_fraud"]

    baseline_proba = baseline_model.predict_proba(X_test[tabular_features])[:, 1]
    graph_proba = model.predict_proba(X_test[feature_cols])[:, 1]

    return {
        "baseline_pr_auc": average_precision_score(y_test, baseline_proba),
        "graph_pr_auc": average_precision_score(y_test, graph_proba),
        "baseline_roc_auc": roc_auc_score(y_test, baseline_proba),
        "graph_roc_auc": roc_auc_score(y_test, graph_proba),
    }
    
df = load_data()

# ---------------------------------------------------------------
# Score all transactions (in a real system this would already be
# stored; here we score on the fly for the demo)
# ---------------------------------------------------------------
@st.cache_data
def score_transactions(_df):
    X = _df[feature_cols]
    proba = model.predict_proba(X)[:, 1]
    _df = _df.copy()
    _df["fraud_probability"] = proba
    _df["risk_tier"] = pd.cut(
        proba, bins=[-0.01, 0.3, 0.7, 1.01], labels=["LOW", "MEDIUM", "HIGH"]
    )
    return _df

scored_df = score_transactions(df)

# ---------------------------------------------------------------
# Header + summary metrics
# ---------------------------------------------------------------
st.title("🔒 WebLock - Fraud Analyst Dashboard")
st.caption("Graph-based real-time fraud detection with explainable risk scoring")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total transactions", f"{len(scored_df):,}")
col2.metric("Flagged HIGH risk", f"{(scored_df['risk_tier']=='HIGH').sum():,}")
col3.metric("Flagged MEDIUM risk", f"{(scored_df['risk_tier']=='MEDIUM').sum():,}")
col4.metric("Actual fraud rate", f"{scored_df['is_fraud'].mean():.1%}")


with st.expander("📊 Why graph features matter — baseline vs. graph-enhanced model", expanded=True):
    comp = load_baseline_comparison()
    c1, c2 = st.columns(2)
    c1.metric("Baseline PR-AUC (tabular only)", f"{comp['baseline_pr_auc']:.3f}")
    c2.metric("Graph-enhanced PR-AUC", f"{comp['graph_pr_auc']:.3f}",
              delta=f"{comp['graph_pr_auc'] - comp['baseline_pr_auc']:+.3f}")
    fig = go.Figure(data=[
        go.Bar(x=["Baseline", "Graph-enhanced"],
               y=[comp["baseline_pr_auc"], comp["graph_pr_auc"]],
               marker_color=["#d9534f", "#5cb85c"],
               text=[f"{comp['baseline_pr_auc']:.3f}", f"{comp['graph_pr_auc']:.3f}"],
               textposition="outside")
    ])
    fig.update_layout(yaxis_title="PR-AUC", showlegend=False, height=350, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def compute_global_importance(_sample_df):
    X_sample = _sample_df[feature_cols]
    sample_shap = explainer.shap_values(X_sample)
    sample_sv = sample_shap[1] if isinstance(sample_shap, list) else sample_shap
    return pd.Series(np.abs(sample_sv).mean(axis=0), index=feature_cols).sort_values(ascending=False)

with st.expander("🔍 Global feature importance (model-wide, sampled)"):
    sample = scored_df.sample(min(2000, len(scored_df)), random_state=42)
    importance = compute_global_importance(sample)
    st.bar_chart(importance.head(10))
    st.caption("Computed on a random 2,000-row sample for speed — mean absolute SHAP value across the sample.")
    
st.divider()

# ---------------------------------------------------------------
# Layout: transaction feed (left) + explanation panel (right)
# ---------------------------------------------------------------
left, right = st.columns([1.3, 1])

with left:
    st.subheader("Transaction Feed")
    risk_filter = st.multiselect(
        "Filter by risk tier", options=["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"]
    )
    search_query = st.text_input("Search by transaction ID or account ID (optional):", "")
    filtered = scored_df[scored_df["risk_tier"].isin(risk_filter)]
    if search_query:
        filtered = filtered[
            filtered["transaction_id"].str.contains(search_query, case=False, na=False) |
            filtered["account_id"].str.contains(search_query, case=False, na=False)
        ]
    filtered = filtered.sort_values("fraud_probability", ascending=False)

    display_cols = [
        "transaction_id", "account_id", "amount", "merchant_category",
        "component_size", "degree", "fraud_probability", "risk_tier"
    ]
    st.dataframe(
        filtered[display_cols].head(200),
        use_container_width=True,
        height=450,
        column_config={
            "fraud_probability": st.column_config.ProgressColumn(
                "fraud_probability",
                help="Model's predicted fraud probability",
                min_value=0,
                max_value=1,
                format="%.3f",
            ),
            "amount": st.column_config.NumberColumn("amount", format="$%.2f"),
        }
    )
    csv_data = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download flagged transactions (CSV)",
        data=csv_data,
        file_name="weblock_flagged_transactions.csv",
        mime="text/csv"
    )

    top_50 = filtered.head(50)
    label_map = {
        row["transaction_id"]: f"{row['transaction_id']} — ${row['amount']:.2f} — {row['risk_tier']} ({row['fraud_probability']:.0%})"
        for _, row in top_50.iterrows()
    }
    selected_label = st.selectbox("Select a transaction to inspect:", options=list(label_map.values()))
    selected_txn_id = [k for k, v in label_map.items() if v == selected_label][0]

with right:
    st.subheader("Why was this flagged?")
    if selected_txn_id:
        txn_row = scored_df[scored_df["transaction_id"] == selected_txn_id].iloc[0]
        idx_in_df = scored_df.index[scored_df["transaction_id"] == selected_txn_id][0]

        st.metric("Fraud probability", f"{txn_row['fraud_probability']:.1%}")
        st.metric("Risk tier", txn_row["risk_tier"])

        X_row = scored_df.loc[[idx_in_df], feature_cols]
        shap_values = explainer.shap_values(X_row)
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        contributions = pd.Series(sv, index=feature_cols).sort_values(key=abs, ascending=False)
        st.write("**Top contributing factors:**")
        for feat, val in contributions.head(5).items():
            direction = "🔺 increases risk" if val > 0 else "🔻 decreases risk"
            st.write(f"- `{feat}` = {X_row.iloc[0][feat]:.2f}  —  {direction} ({val:+.3f})")

        st.divider()
        
        st.write("**Account network (shared devices/IPs/cards):**")

        # Cap how many accounts we pull into the visualization. Some accounts
        # sit in huge, mostly-generic clusters (e.g. a common card value shared
        # by thousands of accounts) -- same reasoning as the entity-group cap
        # in graph_features.py. Trying to lay out a graph with thousands of
        # nodes is both unreadable and slow, so we skip any entity value shared
        # by more accounts than this and note it to the user instead.
        MAX_VIZ_NODES = 40

        acct_id = txn_row["account_id"]
        acct_entities = df[df.account_id == acct_id]

        related_frames = []
        skipped_entities = set()
        for entity_col in ["device_id", "ip_id", "card_id"]:
            for val in acct_entities[entity_col].dropna().unique():
                group = df[df[entity_col] == val]
                group_size = group["account_id"].nunique()
                if 1 < group_size <= MAX_VIZ_NODES:
                    related_frames.append(group)
                elif group_size > MAX_VIZ_NODES:
                    skipped_entities.add(entity_col)

        if related_frames:
            related = pd.concat(related_frames).drop_duplicates(
                subset=["account_id", "device_id", "ip_id", "card_id"]
            )
            # extra safety cap on total accounts shown, even if several
            # small-ish groups combine to something larger than intended
            keep_accounts = related["account_id"].unique()[:MAX_VIZ_NODES]
            related = related[related["account_id"].isin(keep_accounts)]
        else:
            related = pd.DataFrame(columns=df.columns)

        G = nx.Graph()
        for _, r in related.iterrows():
            for entity_col in ["device_id", "ip_id", "card_id"]:
                G.add_edge(r["account_id"], r[entity_col])
        G.add_node(acct_id)  # always show the selected account, even if isolated

        if skipped_entities:
            st.caption(
                f"⚠️ This account shares {', '.join(sorted(skipped_entities))} with a very large "
                f"generic cluster (1000s of accounts) — excluded from the visualization below for "
                f"clarity, same cap used during model training."
            )

        if G.number_of_nodes() > 1:
            fig, ax = plt.subplots(figsize=(5, 4))
            pos = nx.spring_layout(G, seed=42)
            node_colors = ["red" if n == acct_id else ("lightblue" if n.startswith("acct") else "lightgray") for n in G.nodes()]
            nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=300, with_labels=False, edge_color="gray")
            ax.set_title(f"Network around {acct_id[:14]}...")
            st.pyplot(fig)
            plt.close(fig)
            plt.close("all")  # extra safety net against any other stray figures
            gc.collect()      # nudge Python to reclaim memory immediately
            st.caption("🔴 = selected account | 🔵 = other accounts | ⚪ = shared device/IP/card")
        else:
            st.info("No small-cluster connections to visualize — either isolated, or only linked via large generic clusters excluded above.")

st.divider()
st.caption("WebLock v1.0 - Graph-based fraud detection demo. Built with LightGBM, NetworkX, SHAP, FastAPI, Streamlit.")
