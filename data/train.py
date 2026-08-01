## Data içe aktarma
import pandas as pd 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC




## Verinin Okunması 
path = "telco.csv"
df = pd.read_csv(path)
print("-"*30+"Data Type"+"-"*30)
df.info() 


#Outlier Değerler
class IQRClipper(BaseEstimator, TransformerMixin):
    # Bu class'ı sadece 2 "an" olarak düşünün:
    #   1) fit()      -> SADECE train verisine bakıp sınırları hesaplar, hafızaya yazar
    #   2) transform() -> hafızadaki sınırları kullanarak veriyi kırpar (train'e de test'e de aynı sınırlarla)

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.kolonlar_ = [c for c in X.columns if X[c].nunique() > 2]  # sadece sürekli kolonlar
        Q1 = X[self.kolonlar_].quantile(0.25)
        Q3 = X[self.kolonlar_].quantile(0.75)
        IQR = Q3 - Q1
        self.alt_ = Q1 - 1.5 * IQR   # <-- öğrenilen bilgi burada saklanıyor
        self.ust_ = Q3 + 1.5 * IQR   # <-- öğrenilen bilgi burada saklanıyor
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for kolon in self.kolonlar_:
            X[kolon] = X[kolon].clip(lower=self.alt_[kolon], upper=self.ust_[kolon])
        return X.values                       # Tablo tiplerini görme


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
  if col == 'TotalCharges':
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print("'TotalCharges' kolonu sayısal (float) tipe dönüştürüldü (fillna sonraya bırakıldı).")
    continue      
  unique_count = df[col].nunique()
  if unique_count == 2:
    binary_donusturucu([col])
  elif unique_count > 2:
    one_encondig(col)



## Verinin X ve Y olarak ayrılması 
y= df['Churn']
X = df.drop('Churn' , axis=1)
X_tr , X_te , y_tr , y_te = train_test_split(
    X,y , test_size=0.2 , random_state=42 , stratify=y
)


#Corr ilşkisi kontrol ediliyor 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

X_num_tr = X_tr.select_dtypes(include=["number"]).copy()
X_num_tr = X_num_tr.fillna(X_num_tr.median())   # sadece bu grafik için geçici doldurma

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_num_tr, y_tr)   # sadece train

# 4. Tablo haline getir ve sırala
feature_importance = pd.DataFrame(
    {"Degisken": X_num_tr.columns, "Onem_Derecesi": model.feature_importances_}
).sort_values(by="Onem_Derecesi", ascending=False)

# Konsol çıktısı
print("Müşteri Kaybını Etkileyen En Önemli Faktörler:")
print(feature_importance.head(25))

plt.figure(figsize=(14, 9))
top_features = feature_importance.head(15).copy()  # İlk 15 en net görünenidir

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
models = {
    'LogisticRegression': (
        Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clip', IQRClipper()),
            ('sc', StandardScaler()),
            ('m', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ]),
        {'m__C': [0.1, 1, 10]},
    ),
    'KNN': (
        Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clip', IQRClipper()),
            ('sc', StandardScaler()),
            ('m', KNeighborsClassifier())   # class_weight yok, KNN'de bu parametre desteklenmiyor
        ]),
        {'m__n_neighbors': [3, 5, 7, 11]},
    ),
    'DecisionTree': (
        Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clip', IQRClipper()),
            ('m', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
        ]),
        {'m__max_depth': [3, 5, 10, None]},
    ),
    'RandomForest': (
        Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clip', IQRClipper()),
            ('m', RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'))
        ]),
        {'m__n_estimators': [50, 100, 200], 'm__max_depth': [5, 10, None]},
    ),
    'SVM': (
        Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clip', IQRClipper()),
            ('sc', StandardScaler()),
            ('m', SVC(class_weight='balanced',probability=True))
        ]),
        {'sc__with_mean': [True, False], 'm__C': [0.1, 1, 10], 'm__kernel': ['rbf', 'linear']},
    ),
}

##Modelin Eğitilmesi Ve Karşılaştırması
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
    egitilen_modeller[name] = g.best_estimator_

df_sonuc = pd.DataFrame(results).sort_values('Best CV F1', ascending=False)
print("\n🏆 Model Karşılaştırma — ML-04 Modül Final")
print("=" * 70)
print(df_sonuc[['Model', 'Best CV F1', 'Test F1']].to_string(index=False))
print("\nEn iyi modelin parametreleri:")
print(f"  {df_sonuc.iloc[0]['Model']}: {df_sonuc.iloc[0]['Best Params']}")
en_iyi_model_ismi = df_sonuc.iloc[0]['Model']
en_iyi_model = egitilen_modeller[en_iyi_model_ismi]



##Olasılık Tahmini Ve Eşik 

y_proba = en_iyi_model.predict_proba(X_te)[:, 1]
y_pred = en_iyi_model.predict(X_te)

print("="*70)

# --- ÖNCE: farklı eşikleri dene ---
print("\n--- FARKLI EŞİKLERLE KARŞILAŞTIRMA ---")
esikler = [0.5, 0.4, 0.3, 0.2]
for esik in esikler:
    y_pred_esik = (y_proba >= esik).astype(int)
    p = precision_score(y_te, y_pred_esik)
    r = recall_score(y_te, y_pred_esik)
    f = f1_score(y_te, y_pred_esik)
    print(f"Eşik={esik:.1f} -> Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")

# --- SONRA: tabloya bakarak eşiği seç ---
secilen_esik = 0.3   # tabloya bakıp recall ~0.78'e en yakın olanı seçtik
y_pred_final = (y_proba >= secilen_esik).astype(int)

print(f"\n--- Seçilen eşik ({secilen_esik}) ile Genel Değerlendirme ---")
acc = accuracy_score(y_te, y_pred_final)
prec = precision_score(y_te, y_pred_final)
rec = recall_score(y_te, y_pred_final)
f1 = f1_score(y_te, y_pred_final)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\n--- DETAYLI SINIFLANDIRMA RAPORU (seçilen eşik ile) ---")
print(classification_report(y_te, y_pred_final))


#Hata Anlaizi ( Kaçan Müşteri )
import pandas as pd

# Test verisini ve tahminleri birleştiriyoruz
analiz_df = X_te.copy()
analiz_df['Gercek_Churn'] = y_te.values
analiz_df['Modelin_Tahmini'] = y_pred_final   # y_pred değil — seçtiğiniz eşiğin tahminini kullanın

# Modelin kaçırdığı müşteriler (False Negative - En tehlikelisi)
kacanlar = analiz_df[(analiz_df['Gercek_Churn'] == 1) & (analiz_df['Modelin_Tahmini'] == 0)]

print(f"Modelin fark edemediği kaçak müşteri sayısı: {len(kacanlar)}")
# Bu 'kacanlar' tablosundaki müşterilerin fatura tutarlarına veya sözleşme tiplerine bakarak 
# "Demek ki model şu tip müşterilerde kaçırıyor" diye yorum yapabilirsin!
# Kaçırılan müşterilerin ortalama özelliklerine bir bak (Örn: Sözleşme süreleri, aylık ödemeleri vb.)
print("\n--- Kaçırılan Müşterilerin Ortalamaları ---")
print(kacanlar.select_dtypes(include=["number"]).mean())

from sklearn.metrics import precision_score, recall_score, f1_score

print("\n--- FARKLI EŞİKLERLE KARŞILAŞTIRMA ---")
esikler = [0.5, 0.4, 0.3, 0.2]

for esik in esikler:
    y_pred_esik = (y_proba >= esik).astype(int)   # olasılık >= eşik ise 1, değilse 0
    
    p = precision_score(y_te, y_pred_esik)
    r = recall_score(y_te, y_pred_esik)
    f = f1_score(y_te, y_pred_esik)
    
    print(f"Eşik={esik:.1f} -> Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")
print("="*70)
print("Tamamlandı..")