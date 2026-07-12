"""Screen-recording demo UI: type a query, watch the router decide, in real time.

Not part of the hackathon submission (the container only needs agent.py). This
is purely to show what's happening under the hood: the local fine-tuned router
making a free decision, the prompt-based baseline paying for the same decision,
and the real Fireworks answer call, side by side.

Run with: streamlit run demo_app.py
"""
import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def load_setting(name: str) -> str:
    """Read local env vars or root-level Streamlit Community Cloud secrets."""
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        value = str(st.secrets.get(name, ""))
    except FileNotFoundError:
        value = ""
    if value:
        os.environ[name] = value
    return value


for setting_name in ("FIREWORKS_API_KEY", "MODEL", "ALLOWED_MODELS"):
    load_setting(setting_name)

from src.fireworks_client import chat
from router.infer_router import predict_with_backend
from src.baseline_router import classify as baseline_classify

allowed_models = [
    model.strip()
    for model in os.environ.get("ALLOWED_MODELS", "").split(",")
    if model.strip()
]
MODEL = (
    next((model for model in allowed_models if "kimi" in model.lower()), allowed_models[-1])
    if allowed_models
    else os.environ.get("MODEL", "accounts/fireworks/models/kimi-k2p6")
)


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


st.set_page_config(page_title="Query Router Demo", page_icon="🔀", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🔀 Token-Efficient Query Router")
st.caption(
    "Zero-token local router vs. a prompt-based baseline — built for AMD Developer "
    "Hackathon Act II's Track 1 (Hybrid Token-Efficient Routing Agent)."
)

missing_settings = [
    name
    for name, value in {
        "FIREWORKS_API_KEY": load_setting("FIREWORKS_API_KEY"),
    }.items()
    if not value
]
if missing_settings:
    st.error(
        "Missing configuration: "
        + ", ".join(missing_settings)
        + ". Add these values to Streamlit Secrets (or a local .env file)."
    )
    st.stop()


prompt = st.text_area("Enter a query", height=100, placeholder="e.g. What is 12 + 7?")
run = st.button("Run through router", type="primary", disabled=not prompt.strip())

if run:
    with st.status("Routing query...", expanded=True) as status:
        st.write("**Step 1 — Local router** (local decision, zero tokens)")
        t0 = time.time()
        finetuned_label, router_backend = predict_with_backend(prompt)
        finetuned_latency_ms = (time.time() - t0) * 1000
        if router_backend == "deterministic local fallback":
            st.info(
                "The fine-tuned checkpoint is not bundled with this deployment, so the "
                "project's deterministic zero-token router is active."
            )
        st.write(
            f"Backend: `{router_backend}`  \n"
            f"Decision: `{finetuned_label}` → uses `{short_name(MODEL)}` "
            f"({finetuned_latency_ms:.0f}ms, 0 tokens)"
        )

        st.write("**Step 2 — Prompt-based baseline** (real Fireworks call, for comparison only)")
        baseline_result = baseline_classify(prompt)
        st.write(
            f"Decision: `{baseline_result['label']}` → still uses `{short_name(MODEL)}` "
            f"({baseline_result['tokens']} tokens spent just deciding)"
        )

        st.write(f"**Step 3 — Answering via `{short_name(MODEL)}`**")
        answer = chat(
            MODEL,
            prompt,
            max_tokens=700,
            extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"},
        )
        answer_tokens = answer["total_tokens"]
        baseline_query_tokens = baseline_result["tokens"] + answer_tokens

        status.update(label="Done", state="complete")

    st.subheader("Answer")
    st.write(answer["text"])

    st.session_state.history.append({
        "query": prompt[:60] + ("..." if len(prompt) > 60 else ""),
        "finetuned_tokens": answer_tokens,
        "baseline_tokens": baseline_query_tokens,
    })


if st.session_state.history:
    st.subheader("Tokens per query, this session")
    df = pd.DataFrame(st.session_state.history)
    chart_df = df.set_index("query")[["finetuned_tokens", "baseline_tokens"]]
    chart_df.columns = ["Local router", "Prompt-based baseline"]
    st.bar_chart(chart_df)

    st.subheader("Query log")
    st.dataframe(df, use_container_width=True, hide_index=True)

with st.sidebar:
    st.subheader("Fireworks model")
    st.markdown(f"**Model:** `{short_name(MODEL)}`")
    st.divider()
    st.subheader("Session totals")
    n = len(st.session_state.history)
    finetuned_total = sum(h["finetuned_tokens"] for h in st.session_state.history)
    baseline_total = sum(h["baseline_tokens"] for h in st.session_state.history)
    saved = baseline_total - finetuned_total
    saved_pct = (saved / baseline_total * 100) if baseline_total else 0
    st.metric("Queries run", n)
    st.metric("Local router — total tokens", finetuned_total)
    st.metric("Prompt-based baseline — total tokens", baseline_total)
    st.metric("Tokens saved by local routing", saved, delta=f"{saved_pct:.0f}% fewer tokens" if n else None)
    if st.button("Reset session"):
        st.session_state.history = []
        st.rerun()
