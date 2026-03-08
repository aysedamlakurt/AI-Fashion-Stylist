# KÜTÜPHANELERDEN BİLE ÖNCE İLK BU ÇALIŞACAK!
print("🚀 Sistem uyanıyor, yapay zeka kütüphaneleri RAM'e yükleniyor... (Bu işlem 10-20 saniye sürebilir, imleç boşlukta beklerse korkma!)")

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

print("✅ Kütüphaneler başarıyla yüklendi! Vektörleme (Embedding) işlemi başlıyor...")

# 1. Veriyi yükle
df = pd.read_csv('data/cleaned_clothes.csv')
df['cleaned_text'] = df['cleaned_text'].fillna('')
sentences = df['cleaned_text'].tolist()

# 2. Modeli yükle
print("🧠 Yapay Zeka dil modeli yükleniyor (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print(f"⏳ Toplam {len(sentences)} adet kıyafetin vektörü çıkarılacak.")
print("İşlem adımları aşağıda akacak, böylece donmadığını görebileceksin...\n")

# 3. Veriyi 5000'erlik gruplar (batch) halinde işleyip ekrana yazdırıyoruz
batch_size = 5000
all_embeddings = []

for i in range(0, len(sentences), batch_size):
    batch_sentences = sentences[i : i + batch_size]
    
    # Vektöre çevir
    batch_embeddings = model.encode(batch_sentences, show_progress_bar=False)
    all_embeddings.extend(batch_embeddings)
    
    # Ekrana ilerlemeyi bas
    islenen_sayisi = min(i + batch_size, len(sentences))
    print(f"✅ {islenen_sayisi} / {len(sentences)} kıyafet işlendi...")

print("\n🗄️ Bütün veriler başarıyla vektörlendi! FAISS veritabanı inşa ediliyor...")

# 4. FAISS veritabanına kaydet
embeddings_array = np.array(all_embeddings).astype('float32')
embedding_dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)

index.add(embeddings_array)

os.makedirs('models.nosync', exist_ok=True)
faiss.write_index(index, 'models.nosync/clothes_index.faiss')

print("🎉 HARİKA! Kıyafetlerin yapay zeka beyni oluşturuldu ve 'models/clothes_index.faiss' olarak kaydedildi.")