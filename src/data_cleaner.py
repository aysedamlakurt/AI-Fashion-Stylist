import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import ssl # MAC İÇİN EKLENEN YENİ KÜTÜPHANE

# MAC CİHAZLARDAKİ SSL GÜVENLİK ENGELİNİ AŞAN SİHİRLİ KOD:
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download stopwords on first run
nltk.download('stopwords')

# We use English stopwords since the H&M dataset is in English
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # 1. Handle missing values (NaN)
    if pd.isna(text):
        return ""
    
    # 2. Lowercasing
    text = str(text).lower()
    
    # 3. Remove punctuation, numbers, and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # 4. Remove stopwords
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]
    
    # 5. Join cleaned words back into a single string
    return " ".join(cleaned_words)

print("🚀 Veri okuma işlemi başlıyor... (Dosya büyük, biraz sürebilir)")

# 1. Load the raw data
df = pd.read_csv('data/ham_veri.csv')
print(f"✅ Veri başarıyla yüklendi! Toplam {len(df)} adet kıyafet bulundu.")

print("🧹 Metinler birleştiriliyor ve NLP temizliği yapılıyor... (Arkana yaslan, bu işlem 1-2 dakika sürebilir)")

# 2. Define target columns that contain useful NLP features
target_columns = ['prod_name', 'product_type_name', 'colour_group_name', 'department_name', 'detail_desc']

# 3. Combine these columns into a single rich text feature
df['combined_text'] = df[target_columns].fillna('').agg(' '.join, axis=1)

# 4. Apply the NLP cleaning function
df['cleaned_text'] = df['combined_text'].apply(clean_text)

# 5. Extract only the necessary columns for the Vector Database
final_df = df[['article_id', 'cleaned_text']]

# 6. Save the processed data
final_df.to_csv('data/cleaned_clothes.csv', index=False)

print("🎉 MÜKEMMEL! Verin pırıl pırıl oldu ve 'data' klasörüne 'cleaned_clothes.csv' olarak kaydedildi.")