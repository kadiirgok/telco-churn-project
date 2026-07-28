## Data içe aktarma
import pandas as pd 
path = "telco.csv"
df = pd.read_csv(path)



print("-"*30+"Data Type"+"-"*30)
df.info()                        # Tablo tiplerini görme

#Fonksiyonlar
def kolon_kesfetme(kolonlar):
    for kolon in kolonlar:
        print(f"---{kolon} Kolonu---")

        secenek_sayisi = df[kolon].nunique()
        print(f"Toplam farklı seçenek sayısı {secenek_sayisi}")

        print(df[kolon].value_counts())
        print("\n" + "="*30 + "\n")

#Kolon silme
def kolon_silme(kolonlar):
    if isinstance(kolonlar , str):
        kolonlar = [kolonlar]
    df.drop(kolonlar, axis=1, inplace=True)
    print(f"Silinen Kolonlar: {kolonlar}")
    print(f"Kalan güncel kolon sayısı: {df.shape[1]}")

#Binary Dönüştürme
def binary_donusturucu(string_kolonlar):
    for kolon in string_kolonlar:
        benzersiz_degerler = df[kolon].unique()
        if len(benzersiz_degerler) == 2:
            val1, val2 = benzersiz_degerler[0], benzersiz_degerler[1]
            df[kolon] = df[kolon].map({val1: 0, val2: 1})
            print(f"'{kolon}' kolonu otomatik olarak sayısala çevrildi: {val1}=0, {val2}=1")

            
# Çoklu yappıyı int yapma one encoding
def one_encondig(kolon_adi):
    global df 
    df = pd.get_dummies(df , columns=[kolon_adi] , drop_first=True , dtype=int)
    print(f"{kolon_adi} kolonu One-Hot Encoding ile başarıyla dönüştürüldü")
    print(f"Güncel toplam kolon sayısı {df.shape[1]}")


#outlier tespit etme 
# outlier tespit etme (Düzeltilmiş Hali)
def outlier_temizle_ve_baskila(kolonlar):
  for kolon in kolonlar:
    # Sadece sayısal kolonlarda çalış
    if df[kolon].dtype in ["int64", "float64"]:
      Q1 = df[kolon].quantile(0.25)
      Q3 = df[kolon].quantile(0.75)
      IQR = Q3 - Q1

      alt_sinir = Q1 - 1.5 * IQR
      ust_sinir = Q3 + 1.5 * IQR

      # Sınırların dışındakileri baskıla (Clipping)
      # Alt sınırdan düşük olanları alt sınıra, üst sınırdan yüksek olanları üst sınıra eşitle
      df[kolon] = df[kolon].clip(lower=alt_sinir, upper=ust_sinir)
      print(
          f"'{kolon}' kolonundaki aykırı değerler alt ({alt_sinir:.2f}) ve üst"
          f" ({ust_sinir:.2f}) sınırlarına baskılandı."
      )



#Outlier Baskılama 
print("Outlier Değerler Kontrol ediliyor...")
# Sadece sayısal (int ve float) kolonları otomatik seç
sayisal_kolonlar = df.select_dtypes(include=["number"]).columns

for kolon in sayisal_kolonlar:
  # Eğer hedef değişkenimiz (Churn veya 0-1 olan kolon) varsa onu outlier analizinden hariç tutalım
  if kolon == "Churn":
    continue

  Q1 = df[kolon].quantile(0.25)
  Q3 = df[kolon].quantile(0.75)
  IQR = Q3 - Q1

  alt_sinir = Q1 - 1.5 * IQR
  ust_sinir = Q3 + 1.5 * IQR

  # Sınırların dışında kalan kaç tane değer var kontrol edelim
  aykiri_sayisi = df[
      (df[kolon] < alt_sinir) | (df[kolon] > ust_sinir)
  ].shape[0]

  if aykiri_sayisi > 0:
    print(
        f"'{kolon}' kolonunda {aykiri_sayisi} adet aykırı değer tespit edildi."
        " Baskılanıyor..."
    )
    # Veriyi silmek yerine sınırların içine çekiyoruz (Clipping)
    df[kolon] = df[kolon].clip(lower=alt_sinir, upper=ust_sinir)
  else:
    print(f"'{kolon}' kolonunda aykırı değer bulunamadı.")

print("--- Outlier Temizliği Tamamlandı ---\n")

#Genel Bilgi 
print("---Tablo İnt Çevirilmiş Hali Genel Bilgi---")
print(df.select_dtypes(include=['int64', 'float64']).describe()) # genel bilgilere bakıyoruz 


# Encoing yapma
print("Encoidng yapılıyor...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
  if col == 'customerID':
    kolon_silme([col])  # Costumer id direk sil
    continue

  # TotalCharges sütununu yakala, sayısal tipe çevir ve döngüden çıkar
  if col == 'TotalCharges':
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Boş kalan (NaN) değerleri medyan ile dolduralım ki patlamasın
    df['TotalCharges'] = df['TotalCharges'].fillna(
        df['TotalCharges'].median()
    )
    print("'TotalCharges' kolonu sayısal (float) tipe dönüştürüldü.")
    continue  # Artık bu döngü turunu burada bitiriyoruz, encoding'e sokmuyoruz!

  # Benzersiz değer sayısına bak
  unique_count = df[col].nunique()

  # Eğer tam 2 seçenek varsa binary dönüştürürüz
  if unique_count == 2:
    binary_donusturucu([col])

  # Eğer 2'den fazlaysa One-Hot Encoding uygularız
  elif unique_count > 2:
    one_encondig(col)

#Corr ilşkisi kontrol ediliyor 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# 1. Hedef sütununuzun adını buraya yazın
hedef_sutun = "Churn"

# 2. X ve y olarak veriyi ayırın
X = df.drop(columns=[hedef_sutun])
y = df[hedef_sutun]

# Sadece sayısal sütunlarla çalışalım
X_num = X.select_dtypes(include=["number"])

# 3. Random Forest modeli kurup fit edelim
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_num, y)

# 4. Tablo haline getir ve sırala
feature_importance = pd.DataFrame(
    {"Degisken": X_num.columns, "Onem_Derecesi": model.feature_importances_}
).sort_values(by="Onem_Derecesi", ascending=False)

# Konsol çıktısı
print("Müşteri Kaybını Etkileyen En Önemli Faktörler:")
print(feature_importance.head(25))

# 5. Görselleştirme - Profesyonel ve Cezbedici Tasarım
plt.figure(figsize=(14, 9))
top_features = feature_importance.head(15).copy()  # İlk 15 en net görünenidir

# Modern bir tema ve renk paleti seçelim
sns.set_theme(style="whitegrid")
ax = sns.barplot(
    x="Onem_Derecesi",
    y="Degisken",
    data=top_features,
    palette="mako",
    hue="Degisken",
    legend=False,
    edgecolor="black",
    linewidth=0.6,
)

# Çubukların üzerine tam değerleri (yüzde/oran) yazdıralım
for p in ax.patches:
  width = p.get_width()
  ax.annotate(
      f"{width:.3f}",
      xy=(width, p.get_y() + p.get_height() / 2),
      xytext=(5, 0),  # Çubuğun biraz sağında dursun
      textcoords="offset points",
      ha="left",
      va="center",
      fontsize=10,
      fontweight="bold",
      color="#333333",
  )

plt.title(
    "Müşteri Kaybını (Churn) Tetikleyen En Kritik 15 Faktör",
    fontsize=16,
    fontweight="bold",
    pad=15,
    color="#1f77b4",
)
plt.xlabel("Önem Derecesi (Feature Importance)", fontsize=12, fontweight="bold")
plt.ylabel("Değişkenler", fontsize=12, fontweight="bold")

# Grafiği şık bir şekilde sıkıştırıp dosyaya kaydedelim
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)  
print(
    "📊 Özellik önem grafiği 'feature_importance.png' olarak yüksek"
    " çözünürlükte kaydedildi."
)
plt.close()


#Model Eğitme Ksımı
print("Model Eğitiliyor...")
#Model Seçme Değerlendirme 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

y= df['Churn']
X = df.drop('Churn' , axis=1)


X_tr , X_te , y_tr , y_te = train_test_split(
    X,y , test_size=0.2 , random_state=42 , stratify=y
)




models = {
    'LogisticRegression': (
        Pipeline([('sc', StandardScaler()), ('m', LogisticRegression(max_iter=1000))]),
        {'m__C': [0.1, 1, 10]},
    ),
    'KNN': (
        Pipeline([('sc', StandardScaler()), ('m', KNeighborsClassifier())]),
        {'m__n_neighbors': [3, 5, 7, 11]},
    ),
    'DecisionTree': (
        Pipeline([('m', DecisionTreeClassifier(random_state=42))]),
        {'m__max_depth': [3, 5, 10, None]},
    ),
    'RandomForest': (
        Pipeline([('m', RandomForestClassifier(random_state=42, n_jobs=-1))]),
        {'m__n_estimators': [50, 100, 200], 'm__max_depth': [5, 10, None]},
    ),
    'SVM': (
        Pipeline([('sc', StandardScaler()), ('m', SVC())]),
        {'sc__with_mean': [True, False], 'm__C': [0.1, 1, 10], 'm__kernel': ['rbf', 'linear']},
    ),
}

# 1. DEĞİŞİKLİK: Döngünün hemen üstüne modelleri saklamak için boş bir sözlük ekleyin
egitilen_modeller = {}

results = []
for name, (p, params) in models.items():
    g = GridSearchCV(p, params, cv=5, scoring='f1', n_jobs=-1)
    g.fit(X_tr, y_tr)
    results.append({
        'Model':      name,
        'Best CV F1': g.best_score_,
        'Test F1':    g.score(X_te, y_te),
        'Best Params': g.best_params_,
    })
    # 2. DEĞİŞİKLİK: Her modelin eğitilmiş halini ismini anahtar yaparak sözlüğe kaydedin
    egitilen_modeller[name] = g.best_estimator_

# Sizin orijinal tablonuz (Sıralamayı isterseniz 'Test F1'e göre de değiştirebilirsiniz)
df_sonuc = pd.DataFrame(results).sort_values('Best CV F1', ascending=False)
print("\n🏆 Model Karşılaştırma — ML-04 Modül Final")
print("=" * 70)
print(df_sonuc[['Model', 'Best CV F1', 'Test F1']].to_string(index=False))
print("\nEn iyi modelin parametreleri:")
print(f"  {df_sonuc.iloc[0]['Model']}: {df_sonuc.iloc[0]['Best Params']}")


# 3. DEĞİŞİKLİK: 'g.best_estimator_' yerine tablonun 0. indeksindeki (en üstteki) modeli çağırın
en_iyi_model_ismi = df_sonuc.iloc[0]['Model']
en_iyi_model = egitilen_modeller[en_iyi_model_ismi]

# Kodun geri kalan tahmin ve metrik hesaplama kısımları orijinal haliyle aynen devam eder:
y_pred = en_iyi_model.predict(X_te)



print("="*70)
print("---Genel Değerlendirme --- ")
# Her bir metriği ayrı ayrı hesaplıyoruz
acc = accuracy_score(y_te, y_pred)
prec = precision_score(y_te, y_pred)
rec = recall_score(y_te, y_pred)
f1 = f1_score(y_te, y_pred)

print(f"Accuracy (Doğruluk): {acc:.4f}")
print(f"Precision (Hassasiyet): {prec:.4f}")
print(f"Recall (Duyarlılık / Kaçak Yakalama): {rec:.4f}")
print(f"F1-Score (Dengeli Skor): {f1:.4f}")

# Veya tek satırda hepsini tablo şeklinde veren efsane komut:
print("\n--- DETAYLI SINIFLANDIRMA RAPORU ---")
print(classification_report(y_te, y_pred))


#Model Hata ve Yanılma Payı 
import pandas as pd

# Test verisini ve tahminleri birleştiriyoruz
analiz_df = X_te.copy()
analiz_df['Gercek_Churn'] = y_te.values
analiz_df['Modelin_Tahmini'] = y_pred

# Modelin kaçırdığı müşteriler (False Negative - En tehlikelisi)
kacanlar = analiz_df[(analiz_df['Gercek_Churn'] == 1) & (analiz_df['Modelin_Tahmini'] == 0)]

print(f"Modelin fark edemediği kaçak müşteri sayısı: {len(kacanlar)}")
# Bu 'kacanlar' tablosundaki müşterilerin fatura tutarlarına veya sözleşme tiplerine bakarak 
# "Demek ki model şu tip müşterilerde kaçırıyor" diye yorum yapabilirsin!
# Kaçırılan müşterilerin ortalama özelliklerine bir bak (Örn: Sözleşme süreleri, aylık ödemeleri vb.)
print("\n--- Kaçırılan Müşterilerin Ortalamaları ---")
print(kacanlar.select_dtypes(include=["number"]).mean())

print("="*70)
print("Tamamlandı..")