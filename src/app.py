import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import re

# --- ARAYÜZ VE SÜSLEMELER ---
st.set_page_config(page_title="AI Stilist Chat", page_icon="💅", layout="centered")

st.markdown("""
    <style>
    .urun-karti {
        background-color: #fff0f5;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff1493;
        margin: 10px 0;
        font-size: 15px;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
    }
    .stChatInputContainer { padding-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- YAPAY ZEKA VE VERİTABANI KURULUMU ---
GEMINI_API_KEY = "BURAYA_KENDI_API_ANAHTARINIZI_GIRIN"
genai.configure(api_key=GEMINI_API_KEY, transport="rest") # O gıcık SSL hatasını engelleyen sihirli kod

# Yapay Zekaya Özel Görev Veriyoruz (Sistem Kimliği)
sistem_kimligi = """Sen enerjik, havalı ve çok tatlı bir moda danışmanısın. Kullanıcıyla Türkçe sohbet et.
Görevlerin:
1. Kullanıcı kombin istediğinde veya "bunu beğenmedim" dediğinde ona şık bir dille yeni öneriler yap.
2. ÇOK ÖNEMLİ: Önermek istediğin her kıyafetin İNGİLİZCE detaylı tarifini mutlaka [ARA: english description of the clothes] formatında metnin içine gizle. 
Örnek cümlen: "Tabii ki tatlım! O pantolonu beğenmediysen sana şunu önerebilirim: [ARA: chunky white sports sneakers]. Üstüne de [ARA: red leather jacket] çok havalı durur!"
"""
llm = genai.GenerativeModel('gemini-2.5-flash', system_instruction=sistem_kimligi)

@st.cache_resource
def load_models():
    df = pd.read_csv('data/cleaned_clothes.csv')
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index('models.nosync/clothes_index.faiss')
    return df, embedder, index

df, embedder, index = load_models()

def find_match_html(query):
    """Gizli kodu alır, FAISS'te arar ve şık bir karta dönüştürür."""
    vec = embedder.encode([query.strip()]).astype('float32')
    _, indices = index.search(vec, 1)
    res = df.iloc[indices[0][0]]
    urun_ozeti = res['cleaned_text']
    kisa_ozet = (urun_ozeti[:100] + '...') if len(urun_ozeti) > 100 else urun_ozeti
    
    return f'<div class="urun-karti"><b>🛍️ Senin İçin Bulduğum H&M Ürünü:</b><br>{kisa_ozet}</div>'

# --- SOHBET HAFIZASINI BAŞLAT ---
if "chat_session" not in st.session_state:
    st.session_state.chat_session = llm.start_chat(history=[])
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SİTE GÖRÜNÜMÜ ---
st.title("💅 Yapay Zeka Stilistim")
st.write("Benimle sohbet et! İstediğin kombini sor, beğenmediklerini değiştir, renkleri uydur...")

# Önceki mesajları ekrana çiz
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Yeni Mesaj Kutusu
user_input = st.chat_input("Örn: Yağmurlu bir günde kahve içmeye gideceğim, ne giysem?")

if user_input:
    # 1. Kullanıcının mesajını ekrana bas ve hafızaya al
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Asistanın cevabını al (Hafızalı şekilde)
    with st.chat_message("assistant"):
        with st.spinner("Stilistin dolapları karıştırıyor... 💖"):
            response = st.session_state.chat_session.send_message(user_input)
            ai_metni = response.text
            
            # 3. Metnin içindeki [ARA: ...] kodlarını bul ve HTML kartlarıyla değiştir
            aranacaklar = re.findall(r'\[ARA:(.*?)\]', ai_metni)
            gosterilecek_metin = ai_metni
            
            for ingilizce_tarif in aranacaklar:
                urun_html = find_match_html(ingilizce_tarif)
                gosterilecek_metin = gosterilecek_metin.replace(f"[ARA:{ingilizce_tarif}]", urun_html)
            
            # 4. Nihai cevabı ekrana bas ve hafızaya kaydet
            st.markdown(gosterilecek_metin, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": gosterilecek_metin})