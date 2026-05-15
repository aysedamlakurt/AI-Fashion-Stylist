import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai # type: ignore

print("🚀 RAG Tabanlı Akıllı Kombin Motoru Başlatılıyor...")

# 1. Gemini Yapay Zeka Kurulumu (KENDİ API ANAHTARINI BURAYA YAZ)
GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel('gemini-2.5-flash')

# 2. Veritabanı ve Vektör Modeli Yükleniyor
print("🗄️ Veriler ve FAISS beyni yükleniyor...")
df = pd.read_csv('data/cleaned_clothes.csv')
df['cleaned_text'] = df['cleaned_text'].fillna('')
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('models/clothes_index.faiss')

def find_best_match(query):
    """FAISS kullanarak verilen İngilizce tarife en uygun tek ürünü bulur."""
    query_vector = embedder.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, 1)
    idx = indices[0][0]
    
    # Elimizdeki tek bilgi article_id ve cleaned_text
    art_id = df.iloc[idx]['article_id']
    ozet = df.iloc[idx]['cleaned_text']
    
    # Çok uzun metni kısaltalım ki ekranda güzel dursun
    kisa_ozet = (ozet[:75] + '...') if len(ozet) > 75 else ozet
    
    return f"Ürün ID: {art_id} | Tanım: {kisa_ozet}"

def kombin_yap(kullanici_istegi):
    print(f"\n👗 İsteğin: '{kullanici_istegi}'")
    print("🧠 Gemini düşünüyor ve kombin parçalarını belirliyor...")
    
    # Gemini'ye verdiğimiz özel talimat (Prompt Engineering)
    prompt = f"""
    Sen profesyonel bir moda danışmanısın. Kullanıcının şu isteğine göre bir kombin oluştur: '{kullanici_istegi}'
    Bana sadece arama motorunda aratacağım İngilizce kıyafet tariflerini ver. 
    Lütfen sadece şu formatta yanıt ver (başka hiçbir kelime ekleme):
    TOP: [Üst giyim için İngilizce detaylı tarif]
    BOTTOM: [Alt giyim için İngilizce detaylı tarif]
    SHOES: [Ayakkabı için İngilizce detaylı tarif]
    """
    
    # Gemini'den yanıtı al
    response = llm_model.generate_content(prompt)
    ai_cevabi = response.text.strip().split('\n')
    
    print("\n✨ İŞTE SİHİRLİ KOMBİNİN ✨")
    print("-" * 50)
    
    # Gemini'nin verdiği 3 parçayı tek tek FAISS veritabanımızda aratıyoruz
    for satir in ai_cevabi:
        if ":" in satir:
            kategori, ingilizce_tarif = satir.split(":", 1)
            kategori = kategori.strip()
            ingilizce_tarif = ingilizce_tarif.strip()
            
            # FAISS'ten gerçek H&M ürününü bul
            gercek_urun = find_best_match(ingilizce_tarif)
            
            print(f"📍 {kategori} (AI'ın aradığı: {ingilizce_tarif})")
            print(f"   👉 Bulunan H&M Ürünü: {gercek_urun}\n")

print("✅ Sistem Hazır!\n")

# --- TEST ZAMANI ---
kombin_yap("Bahar aylarında ofiste giyebileceğim şık ama rahat bir kombin")
kombin_yap("Spor salonuna giderken giyeceğim rahat ve havalı siyah ağırlıklı bir takım")