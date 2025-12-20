import streamlit as st
import pandas as pd
import os
import shutil
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="غذا و رستوران", page_icon="🥗", layout="centered")

st.markdown("""
<style>
    @import url('https://v1.fontapi.ir/css/Vazir');
    html, body, [class*="css"] { font-family: 'Vazir', 'Tahoma', sans-serif; direction: rtl; text-align: right; }
    .stApp { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
    .stTextInput > div > div > input { direction: ltr; text-align: left; } /* چپ‌چین برای توکن */
    .card { background-color: #1e1e1e; padding: 15px; border-radius: 12px; margin-bottom: 15px; border: 1px solid #333; }
    .title { font-size: 2em; color: #6ee7b7; text-align: right; }
    .result-text { color: #e2e8f0; font-size: 1.1em; line-height: 1.8; text-align: right; direction: rtl; }
</style>
""", unsafe_allow_html=True)

PERSIST_DIRECTORY = "./chroma_db_food_mobile"

# --- بخش دریافت توکن از کاربر (امن) ---
st.markdown('<div class="card"><div class="title">🥗 آشپزیار همراه</div></div>', unsafe_allow_html=True)

# کادر رمز عبور برای گرفتن توکن
hf_token = st.text_input("🔑 توکن HuggingFace را وارد کنید:", type="password", placeholder="hf_...")

if not hf_token:
    st.warning("⚠️ برای شروع، لطفاً توکن را وارد کنید.")
    st.stop() # تا وقتی توکن نباشد، بقیه کد اجرا نمی‌شود

# --- توابع اصلی ---

@st.cache_resource
def load_embedding_model(token):
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=token,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

def create_knowledge_base(urls, token):
    if os.path.exists(PERSIST_DIRECTORY):
        try: shutil.rmtree(PERSIST_DIRECTORY)
        except: pass
    try:
        loader = WebBaseLoader(urls)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        all_splits = text_splitter.split_documents(data)
        
        embedding_model = load_embedding_model(token)
        vector_db = Chroma.from_documents(documents=all_splits, embedding=embedding_model, persist_directory=PERSIST_DIRECTORY)
        return True, len(all_splits)
    except Exception as e:
        return False, str(e)

def perform_rag_search(query, token):
    embedding_model = load_embedding_model(token)
    vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    base_llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=token,
        max_new_tokens=512,
        temperature=0.7
    )
    
    llm = ChatHuggingFace(llm=base_llm)
    messages = [
        {"role": "system", "content": "تو دستیار آشپزی فارسی هستی. کوتاه و خلاصه پاسخ بده."},
        {"role": "user", "content": f"متن:\n{context_text}\n\nسوال: {query}\n\nپاسخ:"}
    ]
    return llm.invoke(messages).content, docs

# --- ادامه رابط کاربری ---

with st.expander("🔗 منابع", expanded=False):
    input_urls = st.text_area("لینک‌ها:", height=100, value="https://fa.wikipedia.org/wiki/آشپزی_ایرانی")
    if st.button("🍳 یادگیری", use_container_width=True):
        if input_urls.strip():
            with st.spinner('⏳ پردازش...'):
                # توکن را به تابع پاس می‌دهیم
                s, r = create_knowledge_base([u.strip() for u in input_urls.split('\n') if u.strip()], hf_token)
            if s: 
                st.success(f"✅ {r} بخش ذخیره شد.")
                st.session_state["db_ready"] = True
            else: st.error(f"❌ {r}")

if st.session_state.get("db_ready"):
    st.markdown("<br>", unsafe_allow_html=True)
    query = st.text_input("سوال:", placeholder="مثلاً: کباب...")
    if st.button("🔎 جستجو", use_container_width=True):
        if query:
            with st.spinner('🤖 ...'):
                try:
                    # توکن را به تابع پاس می‌دهیم
                    res, docs = perform_rag_search(query, hf_token)
                    st.markdown(f'<div class="card"><div class="result-text">{res}</div></div>', unsafe_allow_html=True)
                except Exception as e: st.error(f"خطا: {e}")

