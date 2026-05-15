import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import re

# --- 1. SAYFA VE STİL AYARLARI ---
st.set_page_config(page_title="AI Stilist Chat", page_icon="💅", layout="wide")

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
        color: #333;
    }
    .stChatInputContainer { padding-bottom: 20px; }
    .sidebar-text { font-size: 14px; color: #555; }
    </style>
""", unsafe_allow_html=True)

# --- 2. YAPAY ZEKA VE VERİTABANI KURULUMU ---
# API anahtarını buraya eklemeyi unutma
GEMINI_API_KEY = "AIzaSyAXDbbGjPw7yUwZXH_dkLAee8UnzjnxKv4"
genai.configure(api_key=GEMINI_API_KEY, transport="rest")

sistem_kimligi = """Sen enerjik, havalı ve çok tatlı bir moda danışmanısın. Kullanıcıyla Türkçe sohbet et.
Görevlerin:
1. Kullanıcı kombin istediğinde veya modunu paylaştığında ona şık bir dille yeni öneriler yap.
2. ÇOK ÖNEMLİ: Önermek istediğin her kıyafetin İNGİLİZCE detaylı tarifini mutlaka [ARA: english description] formatında metnin içine gizle. 
Örnek: "Bu enerjik modun için şunu seçtim: [ARA: colorful floral summer dress]. Üstüne de [ARA: denim jacket] çok yakışır!"
"""
llm = genai.GenerativeModel('gemini-2.5-flash', system_instruction=sistem_kimligi)

@st.cache_resource
def load_models():
    # Dosya yollarının doğruluğundan emin ol (models.nosync veya models)
    df = pd.read_csv('data/cleaned_clothes.csv')
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index('models.nosync/clothes_index.faiss')
    return df, embedder, index

df, embedder, index = load_models()

def find_match_html(query):
    vec = embedder.encode([query.strip()]).astype('float32')
    _, indices = index.search(vec, 1)
    res = df.iloc[indices[0][0]]
    urun_ozeti = res['cleaned_text']
    kisa_ozet = (urun_ozeti[:100] + '...') if len(urun_ozeti) > 100 else urun_ozeti
    return f'<div class="urun-karti"><b>🛍️ Senin İçin Bulduğum H&M Ürünü:</b><br>{kisa_ozet}</div>'

# --- 3. SESSION STATE (BELLEK) YÖNETİMİ ---
if "user_db" not in st.session_state:
    st.session_state.user_db = {"test@gmail.com": "1234"} # Örnek kullanıcı
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {} # { "email": [ {"title": "Başlık", "msgs": []} ] }
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. GİRİŞ / KAYIT EKRANI ---
def login_page():
    st.title("💅 AI Stilist'e Hoş Geldin")
    tab1, tab2 = st.tabs(["Giriş Yap", "Kayıt Ol"])
    
    with tab1:
        email = st.text_input("E-posta")
        password = st.text_input("Şifre", type="password")
        if st.button("Giriş"):
            if email in st.session_state.user_db and st.session_state.user_db[email] == password:
                st.session_state.logged_in_user = email
                st.rerun()
            else:
                st.error("Hatalı e-posta veya şifre!")

    with tab2:
        new_email = st.text_input("Yeni E-posta")
        new_pass = st.text_input("Yeni Şifre", type="password")
        if st.button("Kayıt Ol"):
            if new_email in st.session_state.user_db:
                st.warning("Bu email zaten kayıtlı.")
            else:
                st.session_state.user_db[new_email] = new_pass
                st.success("Kayıt başarılı! Şimdi giriş yapabilirsin.")

# --- 5. ANA UYGULAMA ---
if st.session_state.logged_in_user is None:
    login_page()
else:
    user_email = st.session_state.logged_in_user
    
    # Sidebar: Geçmiş ve Profil
    with st.sidebar:
        st.title(f"Selam {user_email.split('@')[0]}! ✨")
        if st.button("Çıkış Yap"):
            st.session_state.logged_in_user = None
            st.session_state.messages = []
            st.rerun()
            
        st.divider()
        st.subheader("📜 Geçmiş Kombinlerin")
        
        if user_email not in st.session_state.all_chats:
            st.session_state.all_chats[user_email] = []

        if st.button("+ Yeni Stil Sohbeti", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_session = llm.start_chat(history=[])
            st.rerun()

        for idx, chat in enumerate(st.session_state.all_chats[user_email]):
            if st.sidebar.button(f"🗨️ {chat['title']}", key=f"chat_{idx}", use_container_width=True):
                st.session_state.messages = chat['msgs']
                # Hafızayı o geçmişe göre güncelle
                formatted_history = [{"role": m["role"], "parts": [m["content"]]} for m in chat['msgs']]
                st.session_state.chat_session = llm.start_chat(history=formatted_history)
                st.rerun()

    # Sohbet Başlatma (Karşılama Mesajı)
    if not st.session_state.messages:
        karshilama = "Selam tatlım! ✨ Bugün modun nasıl? Nasıl giyinmek istersin, sana nasıl bir kombin yapalım?"
        st.session_state.messages.append({"role": "assistant", "content": karshilama})
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = llm.start_chat(history=[])

    # Arayüz
    st.title("💅 Yapay Zeka Stilistim")
    
    # Mesajları Çiz
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Kullanıcı Girişi
    user_input = st.chat_input("Örn: Bugün enerjik hissediyorum, ofis için şık bir şeyler...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Stilistin dolapları karıştırıyor... 💖"):
                response = st.session_state.chat_session.send_message(user_input)
                ai_metni = response.text
                
                # [ARA: ...] kodlarını HTML kartlarına çevir
                aranacaklar = re.findall(r'\[ARA:(.*?)\]', ai_metni)
                gosterilecek_metin = ai_metni
                
                for ingilizce_tarif in aranacaklar:
                    urun_html = find_match_html(ingilizce_tarif)
                    gosterilecek_metin = gosterilecek_metin.replace(f"[ARA:{ingilizce_tarif}]", urun_html)
                
                st.markdown(gosterilecek_metin, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": gosterilecek_metin})

                # Eğer bu yeni bir sohbetse (karşılama + soru + cevap), geçmişe kaydet
                if len(st.session_state.messages) == 3:
                    chat_title = user_input[:25] + "..."
                    st.session_state.all_chats[user_email].append({
                        "title": chat_title,
                        "msgs": list(st.session_state.messages)
                    })
                else:
                    # Mevcut sohbeti güncelle
                    for chat in st.session_state.all_chats[user_email]:
                        if chat['msgs'][1]['content'] == st.session_state.messages[1]['content']:
                            chat['msgs'] = list(st.session_state.messages)