import streamlit as st
import pandas as pd
import os
import shutil

# استفاده از کتابخانه‌های سبک‌تر برای اجرا روی موبایل
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# تغییر مهم: استفاده از API به جای مدل لوکال برای کاهش مصرف رم گوشی
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma

# --- تنظیمات موبایل ---
# تغییر layout به centered برای نمایش بهتر در گوشی‌های عمودی
st.set_page_config(page_title="غذا و رستوران", page_icon="🥗", layout="centered")

# --- استایل‌دهی ریسپانسیو (مخصوص موبایل) ---
st.markdown("""
<style>
    @import url('https://v1.fontapi.ir/css/Vazir');
    
    html, body, [class*="css"] { 
        font-family: 'Vazir', 'Tahoma', sans-serif; 
        direction: rtl; 
        text-align: right; 
    }
    
    .stApp { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
    
    /* تنظیم ورودی‌ها برای موبایل */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea { 
        direction: rtl; 
        text-align: right; 
        font-size: 16px; /* فونت بزرگتر برای تایپ راحت‌تر در گوشی */
    }
    
    .card { 
        background-color: #1e1e1e; 
        padding: 15px; /* کاهش پدینگ برای فضای کم موبایل */
        border-radius: 12px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.4); 
        margin-bottom: 15px; 
        border: 1px solid #333; 
    }
    
    /* استایل‌های واکنش‌گرا (Responsive) */
    @media only screen and (max-width: 600px) {
        .title { font-size: 1.8em !important; }
        .subtitle { font-size: 0.9em !important; }
        .result-text { font-size: 1em !important; line-height: 1.6 !important; }
        div[data-testid="column"] { width: 100% !important; flex: 0 0 100% !important; min-width: 100% !important; }
    }
    
    .title { font-size: 2.2em; font-weight: 800; color: #6ee7b7; text-align: right; }
    .subtitle { color: #a7f3d0; font-size: 1.1em; text-align: right; margin-top: 5px; }
    .result-text { color: #e2e8f0; font-size: 1.1em; line-height: 1.8; text-align: right; direction: rtl; }
    
    /* تنظیم جدول در موبایل */
    [data-testid="stDataFrame"] { direction: rtl; text-align: right; width: 100%; }
    .stDataFrame div[role="columnheader"], .stDataFrame div[role="gridcell"] { text-align: right !important; }
    .stAlert { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

PERSIST_DIRECTORY = "./chroma_db_food_mobile"

# --- تغییر حیاتی برای موبایل ---
# استفاده از API به جای دانلود مدل سنگین روی گوشی
# شما باید توکن را در st.secrets داشته باشید
def get_hf_token():
    # چک کردن توکن از سکرت یا متغیر محیطی
    if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        return st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    elif "HUGGINGFACEHUB_API_TOKEN" in os.environ:
        return os.environ["HUGGINGFACEHUB_API_TOKEN"]
    else:
        st.error("⚠️ توکن HuggingFace یافت نشد. لطفاً آن را تنظیم کنید.")
        return None

@st.cache_resource
def load_embedding_model():
    token = get_hf_token()
    if token:
        # این مدل روی سرور اجرا می‌شود و رم گوشی را اشغال نمی‌کند
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=token,
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return None

def create_knowledge_base(urls):
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
        except:
            pass
    try:
        loader = WebBaseLoader(urls)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        
        all_splits = text_splitter.split_documents(data)
        embedding_model = load_embedding_model()
        
        if embedding_model:
            vector_db = Chroma.from_documents(
                documents=all_splits,
                embedding=embedding_model,
                persist_directory=PERSIST_DIRECTORY
            )
            return True, len(all_splits)
        return False, "مشکل در لود مدل امبدینگ"
    except Exception as e:
        return False, str(e)

def perform_rag_search(query):
    embedding_model = load_embedding_model()
    if not embedding_model:
        return "خطا: توکن یافت نشد", []
        
    vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3}) # کاهش k برای سرعت بیشتر در موبایل
    docs = retriever.invoke(query)
    
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    token = get_hf_token()
    base_llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=token,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.2
    )
    
    llm = ChatHuggingFace(llm=base_llm)
    
    messages = [
        {"role": "system", "content": "تو دستیار آشپزی فارسی هستی. کوتاه و خلاصه پاسخ بده."}, # پرامپت کوتاه‌تر برای موبایل
        {"role": "user", "content": f"متن:\n{context_text}\n\nسوال: {query}\n\nپاسخ:"}
    ]
    
    response = llm.invoke(messages).content
    return response, docs

# --- UI بدنه اصلی ---
st.markdown("""
<div class="card">
    <div class="title">🥗 آشپزیار همراه</div>
    <div class="subtitle">دستیار هوشمند غذا (نسخه موبایل)</div>
</div>
""", unsafe_allow_html=True)

with st.expander("🔗 منابع (کلیک کنید)", expanded=False): # استفاده از Expander برای شلوغ نشدن صفحه گوشی
    input_urls = st.text_area(
        "لینک‌ها:",
        height=100,
        value="https://fa.wikipedia.org/wiki/آشپزی_ایرانی\nhttps://fa.wikipedia.org/wiki/کباب"
    )
    
    if st.button("🍳 یادگیری منابع", use_container_width=True): # دکمه تمام عرض برای موبایل
        if input_urls.strip():
            url_list = [u.strip() for u in input_urls.split('\n') if u.strip()]
            with st.spinner('⏳ پردازش ابری...'):
                success, result = create_knowledge_base(url_list)
            if success:
                st.success(f"✅ {result} بخش ذخیره شد.")
                st.session_state["db_ready"] = True
            else:
                st.error(f"❌ {result}")

if st.session_state.get("db_ready"):
    st.markdown("<br>", unsafe_allow_html=True)
    query = st.text_input("سوال خود را بپرسید:", placeholder="مثلاً: طرز تهیه کباب...")
    
    if st.button("🔎 جستجو", use_container_width=True): # دکمه بزرگ برای لمس راحت‌تر
        if query:
            with st.spinner('🤖 تفکر...'):
                try:
                    ai_response, source_docs = perform_rag_search(query)
                    
                    st.markdown(f"""
                    <div class="card">
                        <h3 style="color:#fbbf24; margin-bottom:10px;">💡 پاسخ:</h3>
                        <div class="result-text">{ai_response}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("📜 مشاهده منابع"):
                        table_data = []
                        for idx, doc in enumerate(source_docs):
                            table_data.append({
                                "منبع": doc.metadata.get('source', 'لینک'),
                                "متن": doc.page_content[:100] + "...",
                            })
                        st.table(pd.DataFrame(table_data))
            
                except Exception as e:
                    st.error(f"خطا: {e}")

