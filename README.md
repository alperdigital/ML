# Ames Housing · Üç sürümde ev fiyatı tahmini

Eksik konut bilgilerinden satış fiyatını tahmin etmekle başlayan proje üç soruyu ele alıyor. Hangi veriye tahmin anında erişebiliriz? Modelleri nasıl adil karşılaştırırız? Ölçtüğümüz fark ne kadar güvenilir? [Etkileşimli proje sunumu](https://alperdigital.github.io/ML/)

## V1, V2 ve V3 aynı 292 evde

2006–2010 arasında Ames’te gerçekleşen aynı 292 satışta V3’ün ortalama mutlak hatası **12.764,41 dolar**. Ev başına hata düzeltilmiş V1’e göre **2.283,25 dolar (%15,17)**, yayımlanmış V2’ye göre **2.229,10 dolar (%14,87)** azaldı.

| Sürüm · model | RMSLE ↓ | Önceki sürüme göre RMSLE | MAE ↓ (USD) | Önceki sürüme göre MAE (USD) | R² ↑ | Önceki sürüme göre R² |
|---|---:|---:|---:|---:|---:|---:|
| V1 · LightGBM (düzeltilmiş) | 0,13601 | — | 15.047,65 | — | 0,92592 | — |
| V2 · CatBoost | 0,12346 | −%9,23 | 14.993,51 | −54,15 | 0,89969 | −0,02623 |
| V3 · TabPFNv2 | **0,11810** | −%4,34 | **12.764,41** | −2.229,10 | **0,94024** | +0,04055 |

RMSLE, gerçek ve tahmin edilen fiyatların logaritmik ölçekteki sapmasını özetler. MAE, mutlak sapmaların dolar cinsinden ortalamasıdır. R², modelin satış fiyatlarındaki değişkenliğin ne kadarını açıkladığını ölçer. V2’nin V1’e göre ortalama dolar hatası azalırken büyük sapmalara daha duyarlı olan R² geriledi. Tablodaki farklar tam hassasiyetli değerlerden hesaplandı. RMSLE farkı göreli yüzde, R² farkı mutlak puandır.

**Farkın belirsizliği.** Aynı evin tahminlerini birlikte tutarak yapılan 20.000 yeniden örneklemede (tohum 42), V3’ün V2’ye karşı RMSLE kazancının koşullu %95 aralığı **−%3,96 ile +%14,03** oldu. Sıfır bu aralığın içinde. Ortalama dolar hatasındaki azalmanın aralığı **969–3.685 dolar**. Bu aralıklar sabit modeller ve daha önce incelenmiş 292 satış için hesaplandı. Güncel piyasaya ilişkin bir tahmin için yeni satışlarda doğrulama gerekir. [Ev bazında tahminler](ames-v3/results/three-version-holdout.csv) · [Ölçümler, farklar ve aralıklar](ames-v3/results/three-version-holdout.json)

## Sürümlerde ne değişti?

- **V1 · İlk ağaç modelleri.** Tablodaki V1 geriye dönük, düzeltilmiş karşılaştırmadır. 1.460 ham kaydın 292’si en başta ayrılır. Eksik değerler ve kategoriler eğitim verisinden öğrenilir. Aykırı değer kuralı 1.168 eğitim kaydından ikisini çıkarır. Ev yaşı takvim yıllarıyla hesaplanır. Toplam oda sayısı yatak odalarını zaten içerir. Örneğin 1 numaralı ev satış sırasında 5 yaşındadır ve 8 odalıdır. Beş katlı geliştirme doğrulaması iki ağaç modeli arasından LightGBM seçti. [V1 ve V2 karşılaştırması](ames-v2/README.md)
- **V2 · Tekrarlanabilir model karşılaştırması.** XGBoost, LightGBM, CatBoost ve Ridge beş katlı doğrulamada model başına 20 Optuna denemesinden geçti. Portföy ve `--version v2` yayımlanan CatBoost seçimini kullanır. Otomatik test, 292 evin her biri için komutun ürettiği fiyatı sitedeki değerle sente kadar karşılaştırır. V3’ün üç iç katlı araştırma karşılaştırıcısı `v2-nested` adıyla sunulur. [Yayımlanmış V2 ayarları](ames-v2/results.json)
- **V3 · Ayrı değerlendirme ve model seçimi.** Karışım ağırlıkları, her iç doğrulama katının bir yarısında öğrenilip diğer yarısında puanlanır. Son kullanıma yönelik ağırlıklar seçimden sonra eğitim verisindeki kat dışı tahminlerle hesaplanır. On dış bölüm, 2010 sınaması ve son seçim düzeltilmiş V1 ile yeniden çalıştırıldı. TabPFNv2 kaynak sütunları doğrudan alır. Özel eksik değer ve alan/banyo/yaş dönüşümleri CatBoost ve Ridge adaylarına aittir. Farklı modellerle fiyat hatasını karşılaştırmak, eksik bir özelliğin gerçek değerini ne kadar iyi geri kazandığımızı ayrıca ölçmez. [Düzeltilmiş protokol](ames-v3/protocol.md) · [Deney raporu](ames-v3/README.md)

Ayrı bir değerlendirmede 1.168 geliştirme evinin her biri iki kez eğitim dışında tahmin edildi. Bu 2.336 tahminde RMSLE düzeltilmiş V3 için **0,11506**, iç içe doğrulamadaki V2 karşılaştırıcısı için **0,14102** oldu. Yukarıdaki tablo yayımlanmış sürümlerin 292 evlik sonuçlarını gösteriyor.

## Sitedeki fiyatı yeniden üretmek

Depo kökünde, [deneyde kullanılan Python ortamıyla](ames-v3/README.md#reproduce-from-the-repository-root) çalıştırın.

```sh
python -m pip install -r ames-v3/requirements.txt
python ames-v3/predict.py --version v2 --holdout-id 899
```

899 numaralı ev için komut ve site **412.405,27 dolar** verir. Aynı evde `--version v2-nested` **456.536,36 dolar** verir. `--holdout-id` yalnızca tarihsel test kümesinden seçer ve gerçek satış fiyatını model girdilerinden çıkarır. Kendi Ames biçimli dosyanızı `--input ev.csv` ile verebilirsiniz.

```sh
python ames-v2/compare_v1_v2.py --data data/train.csv
python ames-v3/compare_releases.py
python -m unittest discover -s ames-v2
python -m unittest discover -s ames-v3
```

İlk komut aynı beş katta düzeltilmiş V1’i ve yayımlanmış V2 ayarlarını değerlendirir. İkinci komut, V1 ve V2’yi yeniden eğitir, kayıtlı V3 tahminlerini ev kimliği ve satış fiyatıyla eşleştirir, sitenin 292 kaydını ve karşılaştırma aralıklarını üretir. Tam V3 deneyini yeniden eğitmek için teknik rapordaki iki Python ortamı ve sabit TabPFNv2 ağırlığı gerekir. [Teknik kurulum](ames-v3/README.md#reproduce-from-the-repository-root)

## İlk sürümün arşivi

İlk sürümün README’sindeki 0,9378 değeri R²’dir. Regresyonda bu değer “evlerin %93,78’inin fiyatı doğru tahmin edildi” anlamına gelmez. Eski özet tablosunda XGBoost daha yüksek R², LightGBM daha düşük RMSLE gösteriyordu. Eski deneylerin örneklemi, dönüşümleri ve bazı metrik etiketleri farklıydı. Kökteki `Untitled1*.ipynb` dosyaları, `proje-main/` ve `proje-deriniz/` eski çalışma kopyalarını içerir. Sürüm ölçümlerinin kaynağı depo kökünden çalışan yukarıdaki komutlar ve eşlenmiş satışlardır. [Özgün proje, kullanım örnekleri ve arşivlenmiş README](https://github.com/alperdigital/ML/tree/a1cf99b304c604125019a10b587613a5d3b1ee5c) · [Önceki 292 evlik rapor](https://github.com/alperdigital/ML/blob/1f8864c518794adbb4217ace49b246d8c18f326d/README.md)
