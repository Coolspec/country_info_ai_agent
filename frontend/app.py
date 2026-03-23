import os
import streamlit as st
import requests
import json

BACKEND_URL = st.secrets.get("BACKEND_URL", "http://127.0.0.1:8000")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Country Info AI Agent",
    page_icon="🌍",
    layout="centered",
)

# ── Custom CSS — ChatGPT-style dark theme ────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Root variables ── */
    :root {
        --bg-primary: #212121;
        --bg-secondary: #171717;
        --bg-sidebar: #171717;
        --bg-input: #2f2f2f;
        --bg-user-bubble: #303030;
        --text-primary: #ececec;
        --text-secondary: #b4b4b4;
        --accent: #10a37f;
        --accent-hover: #1a7f64;
        --border-color: #424242;
    }

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Hide default Streamlit elements ── */
    #MainMenu, footer {visibility: hidden;}
    header {background-color: transparent !important;}
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
        max-width: 48rem !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-sidebar);
        border-right: 1px solid var(--border-color);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown label {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-secondary);
        font-size: 0.85rem;
    }

    /* ── Branding ── */
    .brand-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .brand-header h1 {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.35rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    .brand-header p {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin: 0.25rem 0 0 0;
    }

    /* ── Status badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }
    .status-online {
        background: rgba(16, 163, 127, 0.15);
        color: #10a37f;
        border: 1px solid rgba(16, 163, 127, 0.3);
    }
    .status-offline {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* ── Tools chip ── */
    .tool-chip {
        display: inline-block;
        background: rgba(16, 163, 127, 0.12);
        color: #10a37f;
        border: 1px solid rgba(16, 163, 127, 0.25);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-family: 'Inter', monospace;
        margin: 2px 4px 2px 0;
    }

    /* ── Tool detail card ── */
    .tool-detail-card {
        background: rgba(16, 163, 127, 0.06);
        border: 1px solid rgba(16, 163, 127, 0.15);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    .tool-detail-card .tool-name {
        font-family: 'Inter', monospace;
        font-weight: 600;
        font-size: 0.82rem;
        color: #10a37f;
        margin-bottom: 4px;
    }
    .tool-detail-card .tool-args {
        font-family: 'Inter', monospace;
        font-size: 0.75rem;
        color: var(--text-secondary);
        background: rgba(0,0,0,0.2);
        padding: 6px 10px;
        border-radius: 4px;
        margin-top: 4px;
    }

    /* ── Welcome card ── */
    .welcome-card {
        text-align: center;
        padding: 3rem 1rem;
    }
    .welcome-card .emoji {
        font-size: 3rem;
        margin-bottom: 0.75rem;
    }
    .welcome-card h2 {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 0.5rem 0;
    }
    .welcome-card p {
        font-size: 0.9rem;
        color: var(--text-secondary);
        max-width: 28rem;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* ── Suggestion chips ── */
    .suggestion-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
        margin-top: 1.25rem;
    }
    .suggestion-chip {
        background: var(--bg-input);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-family: 'Inter', sans-serif;
        cursor: default;
        transition: border-color 0.2s;
    }
    .suggestion-chip:hover {
        border-color: var(--accent);
        color: var(--text-primary);
    }

    /* ── Relevance indicator ── */
    .relevance-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 0.72rem;
        color: var(--text-secondary);
        margin-top: 6px;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = BACKEND_URL


# ── Helper: call backend ────────────────────────────────────────────────────
def call_chat_api(query: str) -> dict | None:
    """POST to /api/chat and return the parsed JSON, or None on failure."""
    try:
        resp = requests.post(
            f"{st.session_state.api_base_url}/api/chat",
            json={"query": query},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("⚠️ Could not connect to the backend. Is the server running?")
    except requests.Timeout:
        st.error("⏳ The request timed out. The agent might be overloaded.")
    except requests.HTTPError as exc:
        st.error(f"❌ Server error: {exc.response.status_code}")
    except Exception as exc:
        st.error(f"❌ Unexpected error: {exc}")
    return None


def check_health() -> bool:
    """GET /health — returns True if the backend is reachable."""
    try:
        resp = requests.get(
            f"{st.session_state.api_base_url}/health",
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <h1>🌍 Country Info AI</h1>
        <p>Powered by LangGraph</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # API config
    st.session_state.api_base_url = st.text_input(
        "API Base URL",
        value=st.session_state.api_base_url,
        label_visibility="collapsed",
        placeholder="http://localhost:8000",
    )

    # Health check
    if st.button("🔌 Check Connection", use_container_width=True):
        if check_health():
            st.markdown(
                '<span class="status-badge status-online">● Connected</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge status-offline">● Disconnected</span>',
                unsafe_allow_html=True,
            )

    st.divider()

    # New chat
    if st.button("✨ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Example queries
    st.markdown("**Try asking about:**")
    st.markdown(
        """
        - Capital cities
        - Population & area
        - Languages & currencies
        - Comparing countries
        """
    )


# ── Welcome Screen (when no messages yet) ────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="emoji">🌍</div>
        <h2>Country Info AI Agent</h2>
        <p>Ask me anything about countries — capitals, populations, languages, currencies, borders, and more. I'll fetch real-time data to answer your questions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # We use Streamlit columns and buttons to make clickable suggestion chips
    # We hide the default button styling via CSS to make them look like our chips
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] button {
        background: var(--bg-input);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-family: 'Inter', sans-serif;
        cursor: pointer;
        transition: border-color 0.2s;
        width: 100%;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        border-color: var(--accent);
        color: var(--text-primary);
        background: var(--bg-input);
    }
    div[data-testid="stHorizontalBlock"] button p {
        font-size: 0.8rem;
    }
    /* Center the horizontal block */
    div[data-testid="stHorizontalBlock"] {
        justify-content: center;
        margin-top: 1rem;
        gap: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    suggestions = [
        "What is the capital of Japan?",
        "Compare India and China",
        "Languages spoken in Switzerland",
        "Bordering countries of Germany"
    ]
    
    # Create columns to simulate the row of chips
    cols = st.columns(len(suggestions))
    for i, col in enumerate(cols):
        with col:
            if st.button(suggestions[i], key=f"sugg_{i}"):
                st.session_state.selected_suggestion = suggestions[i]
                st.rerun()

# ── Render Chat History ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🌍"):
        st.markdown(msg["content"])

        # Show metadata for assistant messages
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            tools_used = meta.get("tools_used", [])
            is_relevant = meta.get("is_relevant", True)
            guardrail_rationale = meta.get("guardrail_rationale", "")

            if tools_used:
                with st.expander(f"🔧 Tools used ({len(tools_used)})", expanded=False):
                    for tool in tools_used:
                        tool_name = tool.get("name", "unknown")
                        tool_args = tool.get("args", {})
                        st.markdown(
                            f"""<div class="tool-detail-card">
                                <div class="tool-name">🛠️ {tool_name}</div>
                                <div class="tool-args">{json.dumps(tool_args, indent=2)}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            if not is_relevant:
                rationale_text = f": {guardrail_rationale}" if guardrail_rationale else ""
                st.markdown(
                    f'<span class="relevance-tag">⚠️ Query flagged as off-topic by guardrail{rationale_text}</span>',
                    unsafe_allow_html=True,
                )


# ── Chat Input ───────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask about any country...")

# If suggestion was clicked, override prompt
if "selected_suggestion" in st.session_state:
    prompt = st.session_state.selected_suggestion
    del st.session_state.selected_suggestion

if prompt:
    # ── Append & display user message ──
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # ── Call API & display response ──
    with st.chat_message("assistant", avatar="🌍"):
        with st.spinner("Thinking..."):
            data = call_chat_api(prompt)

        if data:
            answer = data.get("answer", "I couldn't generate a response.")
            st.markdown(answer)

            tools_used = data.get("tools_used", [])
            is_relevant = data.get("is_relevant", True)
            guardrail_rationale = data.get("guardrail_rationale", "")

            if tools_used:
                with st.expander(f"🔧 Tools used ({len(tools_used)})", expanded=False):
                    for tool in tools_used:
                        tool_name = tool.get("name", "unknown")
                        tool_args = tool.get("args", {})
                        st.markdown(
                            f"""<div class="tool-detail-card">
                                <div class="tool-name">🛠️ {tool_name}</div>
                                <div class="tool-args">{json.dumps(tool_args, indent=2)}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            if not is_relevant:
                rationale_text = f": {guardrail_rationale}" if guardrail_rationale else ""
                st.markdown(
                    f'<span class="relevance-tag">⚠️ Query flagged as off-topic by guardrail{rationale_text}</span>',
                    unsafe_allow_html=True,
                )

            # Persist
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "meta": {
                    "tools_used": tools_used,
                    "is_relevant": is_relevant,
                    "guardrail_rationale": guardrail_rationale,
                },
            })
        else:
            fallback = "Sorry, I couldn't reach the backend. Please check the connection."
            st.markdown(fallback)
            st.session_state.messages.append({"role": "assistant", "content": fallback})
