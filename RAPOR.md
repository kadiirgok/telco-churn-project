# Churn Projesi — Rapor

> Bu şablonu doldur ve dosyayı `RAPOR.md` olarak kaydet. Teknik olmayan bir yöneticinin
> anlayacağı **iş diliyle** yaz. Kısa ve net olması yeterli — uzunluk değil, netlik önemli.
> Başlıkları silme; altlarını kendi cümlelerinle doldur.

## 1. Özet (3-4 cümle)
Ne yaptın, ne buldun, sonuç ne? Yönetici sadece bunu okusa fikir sahibi olabilmeli.
-Veri üzerinde data temizlme ve outlier kontrol gerçekleştirdim .
-Null değerler ve bozuk datalar temzilendi.
-Tablo çoğunlu str'di int çevirildi model eğitimi için .
-GridSearchCV kullanarak doğru model seçimi yaptım en çok Random Forset olduğu bulundu . 
-Model hata payı ve kaçırdığı yerler kontrol ettim . 

## 2. Veriyi tanıma ve temizlik
Veride hangi sorunları (eksik/bozuk/tutarsız değerler, tip uyumsuzlukları, dengesizlik vb.)
fark ettin? Her birini **nasıl** ve **neden** öyle çözdün?
-Data üzerinde genel olarak çok bozuk bir yapı yoktu sadece kullanılmaz customerId gibi tabloları sildim genel anlamda data temizdi.
-Müşterilerin sadece az bir kısmı ayrıldığı için datada bir dengesizlik vardı bunun içinde class_weight='balanced' kullanıldı 


## 3. Kurduğun modeller ve karşılaştırma
Hangi modelleri denedin? Nasıl karşılaştırdın? Hangisini seçtin, neden?
-Genel anlamda GridSearchCV kullanarak  model seçimi yapıldı . 
- Genel analamda en başarılı olan Random Forset oldu veri sızıntısıda kapatınca.


## 4. Model ne kadar güvenilir?
Hangi metrik(ler)e baktın ve **neden** onları seçtin? "Accuracy yüksek" demek burada neden
tek başına yeterli/yanıltıcı olabilir? Sonuçları sayılarla ver.
-Müşetilerin bir çoğu şirkette kaldığı için Accuracy skor yüksek çıkması normal bir durumdu amacımız giden az saydıdaki müşterileri tespit etmek .
-Giden Müşterili tespit etmek için recall skorlarına baktım orda veri sızıntısı kapattık dan sonra yüzde 50 den yüzde 80 - 86 seviyesine çıkardım .
-Karar eşikleri ile beraber ne kadar iyi sonuc vereceği de kod içerisinde yazdım.


## 5. Model nerede yanılıyor?
Hangi tür müşterilerde hata yapıyor? Ayrılacak bir müşteriyi kaçırmanın (yanlış "kalır" demenin)
işe maliyeti ne olur?
-Model gözden kaçırdığı 49 kişi var . uzun süredir bizimle olan kişileri ayrılmaz sadık olarak nitelendirmekte en büyük yanılgısı bu .
-Ayrılacak  müşteriyi gözden kaçırmanın şirkete maliyeti, kalacak  müşteriye yanlışlıkla indirim yapmaktan çok daha fazladır.
-Bu yüzden modelin gidenleri yakalaması çok önemlidir.


## 6. Tavsiyen
Bu modelle/analizle yönetime **ne yapmasını** öneriyorsun? Somut bir aksiyon yaz.
Müşterilerin en önemli ayrılma nedenleri toplam fatura  ,aylık fatura , ve müşteri olam süresi ,
Uzun süredir müşteri olanlar daha ve belli bir kotadan üstüne indirim uygulanması veya indirim teklifleri ile kalmasını sağlamak .


## 7. (Opsiyonel) Daha fazla zamanın olsa
Neyi denerdin / neyi geliştirirdin?
-Şirkete gelen şikayet kayıları gibi durumlarda daha detay analiz yapılmasını isterdim.
-Anket değerlendirme gibi arastırmalarla daha derine indirgemek isterim elimizdeki verlilerden fatura ödeme ve kalma süresi göstersede farklı etgenlerle olmakta bunların tespitini isterdim .