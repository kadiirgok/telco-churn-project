# Churn Projesi — Rapor

> Bu şablonu doldur ve dosyayı `RAPOR.md` olarak kaydet. Teknik olmayan bir yöneticinin
> anlayacağı **iş diliyle** yaz. Kısa ve net olması yeterli — uzunluk değil, netlik önemli.
> Başlıkları silme; altlarını kendi cümlelerinle doldur.

## 1. Özet (3-4 cümle)
Ne yaptın, ne buldun, sonuç ne? Yönetici sadece bunu okusa fikir sahibi olabilmeli.
-Veri üzerinde data temizlme ve outlier kontrol gerçekleştirdim .
-Null değerler ve bozuk datalar temzilendi.
-Tablo öoğunlu str'di int çevirildi model eğitimi için .
-GridSearchCV kullanarak doğru model seçimi yaptım en çok LogisticRegression olduğu bulundu . 
-Model hata payı ve kaçırdığı yerler kontrol ettim . 

## 2. Veriyi tanıma ve temizlik
Veride hangi sorunları (eksik/bozuk/tutarsız değerler, tip uyumsuzlukları, dengesizlik vb.)
fark ettin? Her birini **nasıl** ve **neden** öyle çözdün?
-Data üzerinde genel olarak çok bozuk bir yapı yoktu sadece kullanılmaz customerId gibi tabloları sildim genel anlamda data temizdi.
-Sadece encoding drn sonra bir kaç null değerler oldu onlarda çok fazla olmması nedeni ile sıfırla değiştirdim


## 3. Kurduğun modeller ve karşılaştırma
Hangi modelleri denedin? Nasıl karşılaştırdın? Hangisini seçtin, neden?
-Genel anlamda GridSearchCV kullanarak hem model seçimi yapmak hemde genel basarı görmek için kullandım 
LogisticRegression en basarılı model oldu ardından Svm , KNN , RandomForest , DecisionTree gelmekte 


## 4. Model ne kadar güvenilir?
Hangi metrik(ler)e baktın ve **neden** onları seçtin? "Accuracy yüksek" demek burada neden
tek başına yeterli/yanıltıcı olabilir? Sonuçları sayılarla ver.
-Eğitimden sonra Accury , Precission ,Recall ve F1-Score baktım .
-Accury skor %79 veriyor ancak Recall kısmı %53 veriyor yani data üzerinde müşteri kaybının yüzde ellisini yakaladığımızı gösteriyor. Buda beklentimizin altında kalan bir durum 



## 5. Model nerede yanılıyor?
Hangi tür müşterilerde hata yapıyor? Ayrılacak bir müşteriyi kaçırmanın (yanlış "kalır" demenin)
işe maliyeti ne olur?
-Model genel anlamda accury skorda yanılıyor aslında doğru ama kaçan müşterileri yakala istatistiği istediğimiz seviyede değil.
-Ayrılmayı düşünen azınlık kitleler yakalama olarak %50  - %45 seviyelerinde . 

## 6. Tavsiyen
Bu modelle/analizle yönetime **ne yapmasını** öneriyorsun? Somut bir aksiyon yaz.
Genel anlamda güvenlik ve fiyatlandırma olumsuz etkilenmekte bunun üzerinden daha teşvik edici yatırımlar yapılmalıdır.
Uzun sürdir abone olan kişiler daha ayrılma olasılığı daha fazla burda gidecek kişilere elde tutulması için daha basit indirimlerle elde tutmamızı gerek . 


## 7. (Opsiyonel) Daha fazla zamanın olsa
Neyi denerdin / neyi geliştirirdin?
Araşatıram kapsamını daha derine indirgemek isterdim Kulalnııcı sadece belirtielen durumlardan mı yada farklı bir neden den mi ayrılıyor onları kalmasını sağlıycak minimum veya maksimum aralık fiyatlandırma ve testler yapılması gerek ve hem şirket çıkarları hemde müşterileri daha kalıcı olmalarını sağlardım .