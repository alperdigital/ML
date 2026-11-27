# Ames Housing · Üç sürümde ev fiyatı tahmini

**Deneyler duraklatıldı ve ilerleme bu dala kaydedildi.** Düzeltilmiş V1 ve V2, aynı 292 satışta tamamlandı. V3 için 960 iç denemenin 640’ı ve 48 TabPFN tahmin grubunun 15’i kayıtlı. Kaldığınız noktadan başlamak için [devam kaydını](DEVAM.md) açın.

| Aynı 292 ev | RMSLE ↓ | MAE ↓ | R² ↑ |
|---|---:|---:|---:|
| V1 · LightGBM | 0,13679 | 15.225,01 $ | 0,92109 |
| V2 · CatBoost | 0,12451 | 15.042,20 $ | 0,90390 |
| V3 · son değerlendirme bekliyor | — | — | — |

V2’nin V1’e göre RMSLE’si %8,98, ev başına ortalama mutlak hatası 182,81 dolar azaldı. R² ise 0,01719 puan geriledi. Güncel V1 karşılaştırması ham dolar fiyatlarına doğru RMSLE uygulayan düzeltilmiş modeli kullanır. [V1 ve V2 deney raporu](ames-v2/README.md)

[Canlı portföy](https://alperdigital.github.io/ames-housing-price-prediction/) son tamamlanmış üçlü karşılaştırmayı gösteriyor. Bu tablonun kaynak kodu ve dönemin tahmin komutu [son tam yayında](https://github.com/alperdigital/ames-housing-price-prediction/tree/ad18630fc5a074538049aa80a4d19203d7d0d211) incelenebilir. Yeni ayrıntılı anlatım [portföy taslağında](ames-v3/portfolio-draft/) saklanıyor.
