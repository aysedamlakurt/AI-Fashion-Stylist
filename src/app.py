import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import re
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# .env dosyasını yükle (python-dotenv olmadan)
_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from database import (
    init_db, create_user, verify_user, user_exists,
    save_item, get_user_items, delete_item, save_uploaded_image
)

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
    .dolap-karti {
        background-color: #fff0f5;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #ffb6d9;
        margin: 5px 0;
        font-size: 13px;
        box-shadow: 1px 1px 4px rgba(0,0,0,0.08);
        color: #333;
    }
    .stChatInputContainer { padding-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. VERİTABANI BAŞLATMA ---
init_db()

# --- 3. YAPAY ZEKA KURULUMU ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY ortam değişkeni ayarlanmamış. Lütfen .env dosyası oluşturun veya ortam değişkenini tanımlayın.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY, transport="rest")

BASE_SISTEM_KIMLIGI = """Sen enerjik, havalı ve çok tatlı bir moda danışmanısın. Kullanıcıyla Türkçe sohbet et.
Görevlerin:
1. Kullanıcı kombin istediğinde veya modunu paylaştığında ona şık bir dille yeni öneriler yap.
2. Kullanıcının dolabındaki kıyafeleri kullanarak kombin önerileri yap (varsa).
3. ÇOK ÖNEMLİ: Önermek istediğin her kıyafetin İNGİLİZCE detaylı tarifini mutlaka [ARA: english description] formatında metnin içine gizle.
Örnek: "Bu enerjik modun için şunu seçtim: [ARA: colorful floral summer dress]. Üstüne de [ARA: denim jacket] çok yakışır!"
"""


@st.cache_resource
def load_models():
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    df = pd.read_csv(os.path.join(base_dir, 'data', 'cleaned_clothes.csv'))
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(os.path.join(base_dir, 'models.nosync', 'clothes_index.faiss'))
    return df, embedder, index


df, embedder, index = load_models()


def find_match_html(query):
    vec = embedder.encode([query.strip()]).astype('float32')
    _, indices = index.search(vec, 1)
    res = df.iloc[indices[0][0]]
    urun_ozeti = res['cleaned_text']
    kisa_ozet = (urun_ozeti[:100] + '...') if len(urun_ozeti) > 100 else urun_ozeti
    return f'<div class="urun-karti"><b>🛍️ Senin İçin Bulduğum H&M Ürünü:</b><br>{kisa_ozet}</div>'


def get_llm_with_wardrobe(user_email):
    items = get_user_items(user_email)
    if items:
        dolap_metni = "\n".join([
            f"- {it['ana_kategori']} / {it['alt_kategori']}"
            + (f", {it['renk']}" if it.get('renk') else "")
            + (f", {it['marka']}" if it.get('marka') else "")
            + (f", {it['beden']} beden" if it.get('beden') else "")
            for it in items
        ])
        sistem = BASE_SISTEM_KIMLIGI + f"\n\nKullanıcının dolabındaki kıyafetler:\n{dolap_metni}\n"
    else:
        sistem = BASE_SISTEM_KIMLIGI
    return genai.GenerativeModel('gemini-2.5-flash', system_instruction=sistem)


# --- 4. SESSION STATE ---
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. GİRİŞ / KAYIT EKRANI ---
KATEGORILER = {
    "Üst Giyim": {
        "alt": ["T-Shirt", "Gömlek", "Bluz", "Kazak", "Sweatshirt", "Crop Top", "Atlet"],
        "extra": ["kol_tipi", "yaka_tipi", "boy"]
    },
    "Alt Giyim": {
        "alt": ["Jean", "Pantolon", "Etek", "Şort", "Tayt", "Eşofman Altı"],
        "extra": ["bel_tipi", "paca_tipi", "paca_boyu"]
    },
    "Dış Giyim": {
        "alt": ["Mont", "Kaban", "Ceket", "Blazer", "Trençkot", "Hırka", "Yelek"],
        "extra": ["kol_tipi", "yaka_tipi", "boy"]
    },
    "Elbise & Tulum": {
        "alt": ["Mini Elbise", "Midi Elbise", "Maxi Elbise", "Tulum", "Şort Tulum"],
        "extra": ["kol_tipi", "yaka_tipi", "boy"]
    },
    "Ayakkabı": {
        "alt": ["Sneaker", "Topuklu", "Bot/Çizme", "Sandalet", "Loafer", "Oxford"],
        "extra": []
    },
    "Aksesuar": {
        "alt": ["Çanta", "Kemer", "Şapka", "Eşarp", "Takı", "Güneş Gözlüğü"],
        "extra": []
    },
}

RENKLER = ["Siyah", "Beyaz", "Lacivert", "Mavi", "Kırmızı", "Pembe", "Sarı", "Yeşil",
           "Gri", "Bej", "Kahverengi", "Mor", "Turuncu", "Krem", "Diğer"]
DESENLER = ["Düz", "Çizgili", "Kareli", "Çiçekli", "Baskılı", "Benekli", "Diğer"]
MATERYALLER = ["Pamuk", "Denim", "Polyester", "Viskon", "Yün", "Keten", "Deri", "Kadife", "Diğer"]
KALIPLAR = ["Regular", "Slim", "Oversize", "Skinny", "Wide Leg", "Straight", "Crop", "Fitted"]
KOL_TIPLERI = ["Uzun Kol", "Kısa Kol", "Kolsuz", "Yarım Kol", "Balon Kol"]
YAKA_TIPLERI = ["Yuvarlak Yaka", "V Yaka", "Polo Yaka", "Boğazlı", "Kayık Yaka", "Gömlek Yaka"]
BOY_SECENEKLERI = ["Crop (Kısa)", "Bel Hizası", "Kalça Hizası", "Uzun"]
BEL_TIPLERI = ["Yüksek Bel", "Normal Bel", "Düşük Bel", "Elastik Bel"]
PACA_TIPLERI = ["Dar Paça", "Normal Paça", "Geniş Paça", "Flare"]
PACA_BOYLARI = ["Kısa (Şort)", "Bermuda", "Kapri", "Tam Boy"]


def login_page():
    st.title("💅 AI Stilist'e Hoş Geldin")
    tab1, tab2 = st.tabs(["Giriş Yap", "Kayıt Ol"])

    with tab1:
        email = st.text_input("E-posta", key="login_email")
        password = st.text_input("Şifre", type="password", key="login_pass")
        if st.button("Giriş Yap", key="btn_login"):
            if verify_user(email, password):
                st.session_state.logged_in_user = email
                st.session_state.messages = []
                st.rerun()
            else:
                st.error("Hatalı e-posta veya şifre!")

    with tab2:
        new_email = st.text_input("E-posta", key="reg_email")
        new_pass = st.text_input("Şifre", type="password", key="reg_pass")
        if st.button("Kayıt Ol", key="btn_register"):
            if not new_email or not new_pass:
                st.warning("E-posta ve şifre boş bırakılamaz.")
            elif user_exists(new_email):
                st.warning("Bu e-posta zaten kayıtlı.")
            else:
                create_user(new_email, new_pass)
                st.success("Kayıt başarılı! Şimdi giriş yapabilirsin.")


# --- 6. DOLAP SAYFASI ---
def wardrobe_page():
    user_email = st.session_state.logged_in_user
    st.title("👗 Dolabım")

    tab1, tab2 = st.tabs(["📦 Kıyafetlerim", "➕ Kıyafet Ekle"])

    with tab1:
        items = get_user_items(user_email)
        if not items:
            st.info("Dolabın henüz boş. 'Kıyafet Ekle' sekmesinden kıyafet ekleyebilirsin! ✨")
        else:
            st.write(f"Dolabında **{len(items)}** kıyafet var.")
            cols = st.columns(3)
            for i, item in enumerate(items):
                with cols[i % 3]:
                    with st.container():
                        if item.get('image_path') and os.path.exists(item['image_path']):
                            st.image(item['image_path'], use_container_width=True)
                        else:
                            st.markdown("🧥")

                        bilgi = f"**{item['ana_kategori']} — {item['alt_kategori']}**"
                        if item.get('marka'):
                            bilgi += f"  \n🏷️ {item['marka']}"
                        if item.get('renk'):
                            bilgi += f"  \n🎨 {item['renk']}"
                        if item.get('beden'):
                            bilgi += f"  \n📏 {item['beden']} beden"
                        if item.get('desen') and item['desen'] != "Düz":
                            bilgi += f"  \n✨ {item['desen']}"

                        st.markdown(f'<div class="dolap-karti">{bilgi}</div>', unsafe_allow_html=True)

                        if st.button("🗑️ Sil", key=f"del_{item['id']}"):
                            delete_item(item['id'], user_email)
                            st.rerun()

    with tab2:
        st.subheader("Yeni Kıyafet Ekle")

        foto = st.file_uploader("Fotoğraf Yükle (isteğe bağlı)", type=["jpg", "jpeg", "png", "webp"])

        col1, col2 = st.columns(2)
        with col1:
            ana_kat = st.selectbox("Ana Kategori *", list(KATEGORILER.keys()))
        with col2:
            alt_kat = st.selectbox("Alt Kategori *", KATEGORILER[ana_kat]["alt"])

        col3, col4 = st.columns(2)
        with col3:
            marka = st.text_input("Marka")
            renk = st.selectbox("Renk", ["(Seçiniz)"] + RENKLER)
            materyal = st.selectbox("Materyal", ["(Seçiniz)"] + MATERYALLER)
        with col4:
            beden = st.text_input("Beden (örn: S, M, L, 38, 40)")
            desen = st.selectbox("Desen", ["(Seçiniz)"] + DESENLER)
            kalip = st.selectbox("Kalıp", ["(Seçiniz)"] + KALIPLAR)

        extra_fields = KATEGORILER[ana_kat]["extra"]
        if extra_fields:
            st.markdown("---")
            ecol1, ecol2 = st.columns(2)
            with ecol1:
                kol_tipi = st.selectbox("Kol Tipi", ["(Seçiniz)"] + KOL_TIPLERI) if "kol_tipi" in extra_fields else None
                bel_tipi = st.selectbox("Bel Tipi", ["(Seçiniz)"] + BEL_TIPLERI) if "bel_tipi" in extra_fields else None
                paca_tipi = st.selectbox("Paça Tipi", ["(Seçiniz)"] + PACA_TIPLERI) if "paca_tipi" in extra_fields else None
            with ecol2:
                yaka_tipi = st.selectbox("Yaka Tipi", ["(Seçiniz)"] + YAKA_TIPLERI) if "yaka_tipi" in extra_fields else None
                boy = st.selectbox("Boy", ["(Seçiniz)"] + BOY_SECENEKLERI) if "boy" in extra_fields else None
                paca_boyu = st.selectbox("Paça Boyu", ["(Seçiniz)"] + PACA_BOYLARI) if "paca_boyu" in extra_fields else None
        else:
            kol_tipi = yaka_tipi = boy = bel_tipi = paca_tipi = paca_boyu = None

        if st.button("✅ Dolaba Ekle", type="primary"):
            image_path = None
            if foto:
                image_path = save_uploaded_image(user_email, foto)

            data = {
                'username': user_email,
                'image_path': image_path,
                'ana_kategori': ana_kat,
                'alt_kategori': alt_kat,
                'marka': marka or None,
                'beden': beden or None,
                'renk': renk if renk != "(Seçiniz)" else None,
                'desen': desen if desen != "(Seçiniz)" else None,
                'materyal': materyal if materyal != "(Seçiniz)" else None,
                'kalip': kalip if kalip != "(Seçiniz)" else None,
                'boy': boy if boy and boy != "(Seçiniz)" else None,
                'bel_tipi': bel_tipi if bel_tipi and bel_tipi != "(Seçiniz)" else None,
                'paca_tipi': paca_tipi if paca_tipi and paca_tipi != "(Seçiniz)" else None,
                'paca_boyu': paca_boyu if paca_boyu and paca_boyu != "(Seçiniz)" else None,
                'kol_tipi': kol_tipi if kol_tipi and kol_tipi != "(Seçiniz)" else None,
                'yaka_tipi': yaka_tipi if yaka_tipi and yaka_tipi != "(Seçiniz)" else None,
            }
            save_item(data)
            st.success(f"✨ {ana_kat} — {alt_kat} dolabına eklendi!")
            st.rerun()


# --- 7. ANA UYGULAMA ---
if st.session_state.logged_in_user is None:
    login_page()
else:
    user_email = st.session_state.logged_in_user

    with st.sidebar:
        st.title(f"Selam {user_email.split('@')[0]}! ✨")

        if st.button("💬 Stilist Chat", use_container_width=True,
                     type="primary" if st.session_state.current_page == "chat" else "secondary"):
            st.session_state.current_page = "chat"
            st.rerun()

        if st.button("👗 Dolabım", use_container_width=True,
                     type="primary" if st.session_state.current_page == "wardrobe" else "secondary"):
            st.session_state.current_page = "wardrobe"
            st.rerun()

        if st.button("🚪 Çıkış Yap", use_container_width=True):
            st.session_state.logged_in_user = None
            st.session_state.messages = []
            st.session_state.current_page = "chat"
            st.rerun()

        if st.session_state.current_page == "chat":
            st.divider()
            st.subheader("📜 Geçmiş Kombinlerin")

            if user_email not in st.session_state.all_chats:
                st.session_state.all_chats[user_email] = []

            if st.button("+ Yeni Stil Sohbeti", use_container_width=True):
                st.session_state.messages = []
                llm = get_llm_with_wardrobe(user_email)
                st.session_state.chat_session = llm.start_chat(history=[])
                st.rerun()

            for idx, chat in enumerate(st.session_state.all_chats[user_email]):
                if st.sidebar.button(f"🗨️ {chat['title']}", key=f"chat_{idx}", use_container_width=True):
                    st.session_state.messages = chat['msgs']
                    llm = get_llm_with_wardrobe(user_email)
                    formatted_history = [{"role": m["role"], "parts": [m["content"]]} for m in chat['msgs']]
                    st.session_state.chat_session = llm.start_chat(history=formatted_history)
                    st.rerun()

    if st.session_state.current_page == "wardrobe":
        wardrobe_page()
    else:
        # Sohbet başlatma
        if not st.session_state.messages:
            karshilama = "Selam tatlım! ✨ Bugün modun nasıl? Nasıl giyinmek istersin, sana nasıl bir kombin yapalım?"
            st.session_state.messages.append({"role": "assistant", "content": karshilama})
            if "chat_session" not in st.session_state:
                llm = get_llm_with_wardrobe(user_email)
                st.session_state.chat_session = llm.start_chat(history=[])

        st.title("💅 Yapay Zeka Stilistim")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

        user_input = st.chat_input("Örn: Bugün enerjik hissediyorum, ofis için şık bir şeyler...")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Stilistin dolapları karıştırıyor... 💖"):
                    response = st.session_state.chat_session.send_message(user_input)
                    ai_metni = response.text

                    aranacaklar = re.findall(r'\[ARA:(.*?)\]', ai_metni)
                    gosterilecek_metin = ai_metni

                    for ingilizce_tarif in aranacaklar:
                        urun_html = find_match_html(ingilizce_tarif)
                        gosterilecek_metin = gosterilecek_metin.replace(
                            f"[ARA:{ingilizce_tarif}]", urun_html
                        )

                    st.markdown(gosterilecek_metin, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": gosterilecek_metin})

                    if len(st.session_state.messages) == 3:
                        chat_title = user_input[:25] + "..."
                        if user_email not in st.session_state.all_chats:
                            st.session_state.all_chats[user_email] = []
                        st.session_state.all_chats[user_email].append({
                            "title": chat_title,
                            "msgs": list(st.session_state.messages)
                        })
                    else:
                        for chat in st.session_state.all_chats.get(user_email, []):
                            if len(chat['msgs']) > 1 and chat['msgs'][1]['content'] == st.session_state.messages[1]['content']:
                                chat['msgs'] = list(st.session_state.messages)
