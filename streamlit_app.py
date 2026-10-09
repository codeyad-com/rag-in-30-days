"""
RAG Agent Streamlit UI - Optimized for Speed
Persian RTL, role separation, document upload, chat
"""

import streamlit as st
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from settings import RAGConfig
from agent import RAGAgent
from index_builder import IndexBuilder, PDFProcessor
from vector_db import QdrantDB
from db_manager import db

# ============================================================================
# Page config (must be first)
# ============================================================================
st.set_page_config(
    page_title="دستیار هوشمند اسناد",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Minimal RTL CSS (no heavy animations)
# ============================================================================
st.markdown("""
<style>
    /* Force RTL */
    .stApp, .stMarkdown, .stChatMessage, .stButton, .stTextInput, .stSelectbox {
        direction: rtl;
        text-align: right;
    }
            
    .stMetric.st-emotion-cache-10klw3m.eozy2n80 {
    background: #104385;
    }
            
    /* Sidebar dark */
    [data-testid="stSidebar"] {
        background: #1e1e2f;
        color: white;
    }
    /* User chat bubble */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    /* Assistant bubble */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background: #f0f2f6;
        color: #1e1e2f;
        border-radius: 20px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    /* Source citation */
    .source-citation {
        background: #e8f5e9;
        border-right: 4px solid #4caf50;
        padding: 0.3rem;
        margin: 0.3rem 0;
        border-radius: 6px;
        font-size: 0.8rem;
    }
    /* Metrics */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 12px;
        padding: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Text constants
# ============================================================================
TEXTS = {
    "app_title": "دستیار هوشمند اسناد",
    "app_subtitle": "پاسخ به سوالات شما بر اساس محتوای اسناد",
    "user_label": "شما",
    "assistant_label": "دستیار",
    "sources_label": "منابع",
    "thinking": "در حال پردازش...",
    "clear_chat": "پاک کردن گفتگو",
    "query_placeholder": "سوال خود را بنویسید...",
    "user_role": "کاربر عادی",
    "admin_role": "مدیر سیستم",
    "login": "ورود",
    "logout": "خروج",
    "login_success": "ورود موفق!",
    "login_failed": "نام کاربری یا رمز عبور اشتباه است",
    "retrieval_settings": "تنظیمات بازیابی",
    "top_k": "تعداد نتایج",
    "min_score": "حداقل امتیاز",
    "llm_settings": "تنظیمات مدل زبانی",
    "temperature": "دمای خلاقیت",
    "max_tokens": "حداکثر توکن",
    "save_config": "ذخیره تنظیمات",
    "stats": "آمار",
    "debug_panel": "پنل دیباگ",
    "total_time": "زمان پاسخ",
    "upload_documents": "آپلود اسناد",
    "upload_pdf": "آپلود PDF",
    "upload_success": "فایل با موفقیت ایندکس شد",
    "upload_error": "خطا در ایندکس کردن",
    "indexed_docs": "اسناد ایندکس شده",
    "no_documents": "هیچ سندی ایندکس نشده",
    "rebuild_index": "بازسازی ایندکس",
    "total_chunks": "کل قطعات",
    "total_docs": "کل اسناد",
    "total_queries": "کل درخواست‌ها",
    "avg_response": "میانگین پاسخ",
    "welcome_desc": "سوال خود را بپرسید",
    "admin_desc": "مدیریت اسناد و تنظیمات",
}

# ============================================================================
# Cached resources (once per session)
# ============================================================================
@st.cache_resource
def get_config():
    return RAGConfig()

@st.cache_resource
def get_agent():
    """Cached RAGAgent - created once."""
    config = get_config()
    return RAGAgent(config=config)

@st.cache_resource
def get_qdrant():
    config = get_config()
    return QdrantDB(config=config)

@st.cache_data(ttl=300)
def get_stats():
    return db.get_stats()

@st.cache_data(ttl=300)
def get_documents():
    return db.get_documents()

def clear_cache():
    st.cache_resource.clear()
    st.cache_data.clear()

# ============================================================================
# Session state init
# ============================================================================
def init_session():
    if "role" not in st.session_state:
        st.session_state.role = None
    if "auth" not in st.session_state:
        st.session_state.auth = False
    if "history" not in st.session_state:
        st.session_state.history = db.get_chat_history()
    if "debug" not in st.session_state:
        st.session_state.debug = {}
    if "config" not in st.session_state:
        cfg = get_config()
        st.session_state.config = {
            "top_k": cfg.default_top_k,
            "min_score": cfg.min_score_threshold,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }

# ============================================================================
# Helper: apply config to agent
# ============================================================================
def apply_config(agent):
    cfg = st.session_state.config
    agent.update_config(top_k=cfg["top_k"])
    agent.retriever.update_config(top_k=cfg["top_k"], min_score=cfg["min_score"])
    agent.llm.update_config(temperature=cfg["temperature"], max_tokens=cfg["max_tokens"])

# ============================================================================
# Indexing functions
# ============================================================================
def index_pdf(file_path: Path, file_name: str) -> bool:
    try:
        config = get_config()
        from embeddings import Embedder
        pages = PDFProcessor.extract_text(file_path)
        cleaned = [{"page_number": p["page_number"], "text": PDFProcessor.clean_text(p["text"])} for p in pages]
        chunks, meta = PDFProcessor.chunk_text(cleaned, config.chunk_size, config.overlap_sentences)
        if not chunks:
            return False
        embedder = Embedder(config=config)
        embeddings = embedder.embed_text(chunks)
        payloads = []
        for ch, m in zip(chunks, meta):
            payloads.append({
                "text": ch,
                "doc_id": file_path.stem,
                "source": file_name,
                "page_number": m["page_number"],
                "chunk_id": m["chunk_id"],
            })
        qd = QdrantDB(config=config)
        if not qd.collection_exists():
            qd.create_collection()
        count = qd.add_vectors(embeddings, payloads)
        db.add_document(file_name, count, file_path.stat().st_size / (1024*1024))
        return True
    except Exception as e:
        st.error(f"خطا: {e}")
        return False

def rebuild_index():
    config = get_config()
    builder = IndexBuilder(config=config)
    builder.build_collection(force_recreate=True)
    results = builder.index_directory(config.data_dir, force_recreate=True)
    db.clear_all_documents()
    for name, cnt in results.items():
        if cnt:
            path = config.data_dir / name
            size = path.stat().st_size / (1024*1024) if path.exists() else 0
            db.add_document(name, cnt, size)
    clear_cache()
    return True

# ============================================================================
# UI components
# ============================================================================
def chat_interface(agent, is_admin):
    # Show last 20 messages
    for msg in st.session_state.history[-20:]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**{TEXTS['user_label']}**\n\n{msg['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**{TEXTS['assistant_label']}**\n\n{msg['content']}")
                if msg.get("sources"):
                    with st.expander(f"📚 {TEXTS['sources_label']}"):
                        for src in msg["sources"][:3]:
                            st.markdown(f'<div class="source-citation">📄 {src["source"]} (صفحه {src["page"]}) - {src["score"]:.2f}</div>', unsafe_allow_html=True)

    query = st.chat_input(TEXTS["query_placeholder"])
    if query:
        # Add user message
        with st.chat_message("user"):
            st.markdown(f"**{TEXTS['user_label']}**\n\n{query}")
        # Process
        with st.chat_message("assistant"):
            with st.spinner(TEXTS["thinking"]):
                try:
                    start = time.perf_counter()
                    apply_config(agent)
                    result = agent.answer(query)
                    elapsed = (time.perf_counter() - start) * 1000
                    db.increment_query(elapsed, True)
                    st.markdown(f"**{TEXTS['assistant_label']}**\n\n{result['answer']}")
                    if result.get("sources"):
                        with st.expander(f"📚 {TEXTS['sources_label']}"):
                            for src in result["sources"][:3]:
                                st.markdown(f'<div class="source-citation">📄 {src["source"]} (صفحه {src["page"]}) - {src["score"]:.2f}</div>', unsafe_allow_html=True)
                    if is_admin:
                        st.session_state.debug = {"time": f"{elapsed:.0f}ms", "sources": len(result.get("sources", []))}
                    db.add_chat_message("user", query)
                    db.add_chat_message("assistant", result["answer"], result.get("sources", []))
                    st.session_state.history = db.get_chat_history()
                except Exception as e:
                    st.error(f"خطا: {e}")
                    db.increment_query(0, False)
        st.rerun()

def admin_sidebar():
    with st.sidebar:
        st.markdown("## 👑 مدیریت")
        if st.button(TEXTS["logout"], use_container_width=True):
            st.session_state.role = None
            st.session_state.auth = False
            st.rerun()
        st.divider()
        st.markdown(f"### 🔍 {TEXTS['retrieval_settings']}")
        top_k = st.slider(TEXTS["top_k"], 1, 10, st.session_state.config["top_k"])
        min_score = st.slider(TEXTS["min_score"], 0.0, 1.0, st.session_state.config["min_score"], 0.05)
        st.markdown(f"### 🧠 {TEXTS['llm_settings']}")
        temp = st.slider(TEXTS["temperature"], 0.0, 1.0, st.session_state.config["temperature"], 0.05)
        max_tok = st.number_input(TEXTS["max_tokens"], 256, 4096, st.session_state.config["max_tokens"], 256)
        if st.button(TEXTS["save_config"], use_container_width=True):
            st.session_state.config.update({"top_k": top_k, "min_score": min_score, "temperature": temp, "max_tokens": max_tok})
            st.success("✅ ذخیره شد")
            time.sleep(0.2)
            st.rerun()
        st.divider()
        stats = get_stats()
        st.metric(TEXTS["total_queries"], stats.get("total_queries", 0))
        st.metric(TEXTS["avg_response"], f"{stats.get('avg_response_time_ms', 0):.0f} ms")
        return {"top_k": top_k, "min_score": min_score, "temperature": temp, "max_tokens": max_tok}

def debug_panel():
    st.markdown(f"### 🔧 {TEXTS['debug_panel']}")
    if st.session_state.debug:
        st.metric(TEXTS["total_time"], st.session_state.debug.get("time", ""))
        st.metric(TEXTS["sources_label"], st.session_state.debug.get("sources", 0))
    else:
        st.info("پس از سوال اول نمایش داده می‌شود")

def document_manager():
    config = get_config()
    st.markdown(f"### 📤 {TEXTS['upload_documents']}")
    files = st.file_uploader(TEXTS["upload_pdf"], type=["pdf"], accept_multiple_files=True)
    if files:
        for f in files:
            with st.spinner(f"در حال ایندکس {f.name}..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = Path(tmp.name)
                dest = config.data_dir / f.name
                dest.parent.mkdir(exist_ok=True)
                shutil.copy(tmp_path, dest)
                ok = index_pdf(dest, f.name)
                tmp_path.unlink()
                if ok:
                    st.success(f"✅ {TEXTS['upload_success']}: {f.name}")
                else:
                    st.error(f"❌ {TEXTS['upload_error']}: {f.name}")
        clear_cache()
        st.rerun()
    st.divider()
    st.markdown(f"### 📄 {TEXTS['indexed_docs']}")
    docs = get_documents()
    if not docs:
        st.info(TEXTS["no_documents"])
    else:
        for d in docs:
            col1, col2, col3 = st.columns([3,1,1])
            col1.write(f"📄 {d['name']}")
            col2.write(f"{d['chunks']} قطعه")
            col3.write(f"{d.get('size_mb',0):.1f} MB")
    st.divider()
    if st.button(f"🔄 {TEXTS['rebuild_index']}", use_container_width=True):
        with st.spinner("در حال بازسازی..."):
            if rebuild_index():
                st.success("بازسازی کامل شد")
                st.rerun()

def login_screen():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown(f"# {TEXTS['app_title']}")
        st.markdown(f"### {TEXTS['app_subtitle']}")
        tab1, tab2 = st.tabs([f"👤 {TEXTS['user_role']}", f"👑 {TEXTS['admin_role']}"])
        with tab1:
            if st.button("🚀 ورود به عنوان کاربر عادی", use_container_width=True):
                st.session_state.role = "user"
                st.session_state.auth = True
                st.rerun()
        with tab2:
            with st.form("admin_login"):
                user = st.text_input("نام کاربری")
                pwd = st.text_input("رمز عبور", type="password")
                if st.form_submit_button(TEXTS["login"]):
                    if db.verify_admin(user, pwd):
                        st.session_state.role = "admin"
                        st.session_state.auth = True
                        st.success(TEXTS["login_success"])
                        time.sleep(0.2)
                        st.rerun()
                    else:
                        st.error(TEXTS["login_failed"])

# ============================================================================
# Main
# ============================================================================
def main():
    init_session()
    if not st.session_state.auth or st.session_state.role is None:
        login_screen()
        return

    agent = get_agent()
    if st.session_state.role == "user":
        st.markdown(f"# {TEXTS['app_title']}")
        st.markdown(f"### {TEXTS['welcome_desc']}")
        col1, col2 = st.columns([0.85, 0.15])
        with col2:
            if st.button(f"🗑️ {TEXTS['clear_chat']}"):
                db.clear_chat_history()
                st.session_state.history = []
                st.rerun()
        st.divider()
        chat_interface(agent, is_admin=False)
    else:
        st.markdown(f"# {TEXTS['app_title']} 👑")
        st.markdown(f"### {TEXTS['admin_desc']}")
        col1, col2 = st.columns([0.85, 0.15])
        with col2:
            if st.button(f"🗑️ {TEXTS['clear_chat']}"):
                db.clear_chat_history()
                st.session_state.history = []
                st.rerun()
        st.divider()
        tab1, tab2, tab3 = st.tabs(["💬 گفتگو", "📁 مدیریت اسناد", "🔧 دیباگ"])
        with tab1:
            admin_sidebar()  # sidebar already shown
            chat_interface(agent, is_admin=True)
        with tab2:
            document_manager()
        with tab3:
            debug_panel()

if __name__ == "__main__":
    main()