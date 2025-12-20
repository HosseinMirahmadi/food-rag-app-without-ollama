import streamlit as st
import pandas as pd
import os
import shutil

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace  # جدید: ChatHuggingFace اضافه شد
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="غذا و رستوران", page_icon="🥗", layout="wide")

st.markdown("""
<style>
    @import url('https://v1.fontapi.ir/css/Vazir');
    html, body, [class*="css"] { font-family: 'Vazir', 'Tahoma', sans-serif; direction: rtl; text-align: right; }
    .stApp { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea { direction: rtl; text-align: right; }
    .card { background-color: #1e1e1e; padding: 20px; border-radius: 16px; box-shadow: 0 8px 20px rgba(0,0,0,0.4); margin-bottom: 20px; border: 1px solid #333; }
    .title { font-size: 2.4em; font-weight: 800; color: #6ee7b7; text-align: right; }
    .subtitle { color: #a7f3d0; font-size: 1.1em; text-align: right; margin-top: 5px; }
    .result-text { color: #e2e8f0; font-size: 1.1em; line-height: 1.8; text-align: right; direction: rtl; }
    [data-testid="stDataFrame"] { direction: rtl; text-align: right; }
    .stDataFrame div[role="columnheader"], .stDataFrame div[role="gridcell"] { text-align: right !important; justify-content: right !important; }
    .stAlert { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

PERSIST_DIRECTORY = "./chroma_db_food"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

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
        vector_db = Chroma.from_documents(
            documents=all_splits,
            embedding=embedding_model,
            persist_directory=PERSIST_DIRECTORY
        )
        return True, len(all_splits)
    except Exception as e:
        return False, str(e)

def perform_rag_search(query):
    embedding_model = load_embedding_model()
    vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # --- جدید: استفاده از ChatHuggingFace برای مدل conversational ---
    base_llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"],
        temperature=0.7,
        max_new_tokens=512,
        repetition_penalty=1.1
    )
    
    llm = ChatHuggingFace(llm=base_llm)  # این خط ارور task رو کامل حل می‌کنه
    
    messages = [
        {"role": "system", "content": "تو یک متخصص حرفه‌ای غذا و آشپزی ایرانی هستی. فقط و فقط به زبان فارسی استاندارد پاسخ بده. از انگلیسی یا هر زبان دیگری استفاده نکن."},
        {"role": "user", "content": f"اطلاعات مرتبط از منابع:\n{context_text}\n\nسوال کاربر: {query}\n\nپاسخ کامل، دقیق و مفید به فارسی بده:"}
    ]
    
    response = llm.invoke(messages).content  # .content برای گرفتن فقط متن پاسخ
    return response, docs

# رابط کاربری (همون قبلی)
st.markdown("""
<div class="card">
    <div class="title">🥗 دستیار هوشمند غذا و رستوران</div>
    <div class="subtitle">جستجو در منوی رستوران‌ها، دستور پخت‌ها و مقالات غذایی</div>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🔗 مرحله ۱: منابع اطلاعاتی")
with st.container():
    input_urls = st.text_area(
        "لینک‌ها را وارد کنید (هر خط یک لینک):",
        height=100,
        placeholder="https://example.com/menu",
        value="https://fa.wikipedia.org/wiki/آشپزی_ایرانی\nhttps://fa.wikipedia.org/wiki/کباب"
    )

st.markdown("### 👨‍🍳 مرحله ۲: پردازش")
if st.button("🍳 بررسی و یادگیری"):
    if input_urls.strip():
        url_list = [u.strip() for u in input_urls.split('\n') if u.strip()]
        with st.spinner('در حال خواندن منابع و ساخت پایگاه دانش...'):
            success, result = create_knowledge_base(url_list)
        if success:
            st.success(f"✅ انجام شد! {result} بخش متنی ذخیره شد.")
            st.session_state["db_ready"] = True
        else:
            st.error(f"❌ خطا: {result}")
    else:
        st.warning("⚠️ لطفاً حداقل یک لینک وارد کنید.")

if st.session_state.get("db_ready"):
    st.markdown("### 🍽️ مرحله ۳: پرسش و پاسخ")
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("سوال شما:", placeholder="مثلاً: کباب کوبیده خوب چه ویژگی‌هایی دارد؟")
    with col2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        search = st.button("🔎 جستجو", use_container_width=True)

    if search and query:
        with st.spinner('در حال جستجو و تولید پاسخ...'):
            try:
                ai_response, source_docs = perform_rag_search(query)
                
                st.markdown(f"""
                <div class="card">
                    <h3 style="color:#fbbf24; text-align:right; margin-bottom:10px;">🍕 پاسخ هوش مصنوعی:</h3>
                    <div class="result-text">{ai_response}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📜 منابع یافت شده")
                table_data = []
                for idx, doc in enumerate(source_docs):
                    table_data.append({
                        "رتبه": idx + 1,
                        "متن (خلاصه)": doc.page_content[:150] + "...",
                        "لینک منبع": doc.metadata.get('source', 'نامشخص'),
                    })
                df = pd.DataFrame(table_data)
                st.dataframe(
                    df,
                    use_container_width=True,
                    column_config={
                        "لینک منبع": st.column_config.LinkColumn("لینک کامل"),
                        "رتبه": st.column_config.NumberColumn("رتبه", format="%d")
                    },
                    hide_index=True
                )
            except Exception as e:
                st.error(f"خطا در تولید پاسخ: {e}")