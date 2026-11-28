# Ames Housing V3 · Devam kaydı

27 Kasım 2026, Türkiye saati. Kullanıcının isteği üzerine eğitimler durduruldu. **Buradan devam etmek için önce bu dosyayı okuyun.** Amaç, üç düzeltilmiş sürümü aynı 292 evde RMSLE, MAE ve R² ile karşılaştırmak, Türkçe proje anlatımını güncellemek ve yalnızca `v3` dalına yayımlamak.

## Nerede kaldık?

| Tamamlanan yeni kıyas · 292 ev | RMSLE ↓ | MAE ↓ | R² ↑ |
|---|---:|---:|---:|
| V1 · LightGBM | 0,13679 | 15.225,01 $ | 0,92109 |
| V2 · CatBoost | 0,12451 | 15.042,20 $ | 0,90390 |
| V3 · değerlendirme bekliyor | — | — | — |

- V1 ve beş katlı V2 tamamlandı. Model başına 20 gerçek denemeli V2 araması, kısa referans deneyleri ve bağımsız yeniden eğitim sonuçları `ames-v2/` içinde. Sitede gösterilecek güncel V2 fiyatları tek satırdan okunan 292/292 satışta toplu tahminle sentine kadar eşleşti.
- Üç düzeltme uygulandı. V1 görülmemiş kategoriye ayrı kod veriyor. Okuyucular kaplama olmadığını belirten 864 `None` ile sekiz bilinmeyen `NA` değerini ayırıyor. RMSLE önce dizileri ev başına bire bir eşliyor, gerçek ve tahmin edilen dolar fiyatlarının her birine bir kez logaritma uyguluyor. V1’deki kimlik girdisi de çıkarıldı. Arşivdeki çift logaritmalı skor bu tabloya girmedi.
- V3 iç aramalarının **8/12’si**, yani **640/960 deneme** eksiksiz kayıtlı. Tamamlananlar `r0f0…r0f4`, `r1f0…r1f2`. Kalan dört bölüm `r1f3`, `r1f4`, `temporal` ve `final`. Kesilen kısmi aramalar ilgili bölümün başından çalışır.
- TabPFN’de **15/48** tahmin grubu kayıtlı. `r0f0`, `r0f1` ve `r0f3` için üç iç ve bir dış grup tamamlandı. `r0f2` için üç iç grup tamamlandı. Kalan **33** grup, yedi klasik adayın kayıtları ve ortam bilgisi [manifestte](ames-v3/checkpoint/manifest.json). İki ek tek-ev kontrolü bu sayılara dahil değil.
- V3 için nihai seçim, 2010 sınaması ve yeni 292 evlik karşılaştırma bekliyor. İsteğe bağlı TabPFN kontrolündeki dizin sorunu giderildi. TabPFN’siz eski deneme çıktısını kullanmayın. [Önceki yayımlanan üçlü sonuç](https://github.com/duayenalper/ames-housing-price-prediction/tree/ad18630fc5a074538049aa80a4d19203d7d0d211) bu duraklama sırasında sitenin veri kaynağıdır.

## Deneyleri sürdürmek için

Depo kökünde çalışın. Tamamlanmış sayısal kayıtlar dalda `ames-v3/checkpoint/progress.json.gz` içinde saklanıyor. Açılan sekiz aramanın, yedi klasik adayın ve 15 TabPFN grubunun değerleri özgün kayıtlarla karşılaştırıldı. Aynı bilgisayarda `.ames-v3-work/` altında da geri yüklenmiş yerel kopya bulunuyor. Geri yükleme mevcut dosyaları korur. Önbellek anahtarları veri, bölüm, kod ve çalışma ortamına bağlıdır. Başka bir ortam eşleşmezse ilgili işler yeniden hesaplanır.

Bu bilgisayardaki doğrulanmış Python 3.9.6 ortamı ile bağımsız TabPFN Python 3.12.14 ortamı duruyorsa aşağıdaki komutlar kullanılabilir. Yeni kurulumda karşılık gelen sürümleri [ana ortama](ames-v3/requirements.txt) ve [TabPFN ortamına](ames-v3/requirements-tabpfn.txt) ayrı yükleyin. [Önceki kurulumdaki model indirme adımı](https://github.com/duayenalper/ames-housing-price-prediction/blob/ad18630fc5a074538049aa80a4d19203d7d0d211/ames-v3/README.md#reproduce-from-the-repository-root) aynı sabit ağırlığı indirir.

```sh
export PYTHONPATH=/tmp/ames-v1-deps:/tmp/ames-v2-full-deps:/tmp/ames-corrected-examples-deps
export AMES_TABPFN_PYTHON=/tmp/ames-v3-pfn/bin/python
export TABPFN_MODEL_CACHE_DIR=/tmp/ames-v3-pfn-weights
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=/tmp/ames-mpl
export AMES_TABPFN_THREADS=3
W="$PWD/.ames-v3-work"
export AMES_TABPFN_PRED_CACHE="$W/foundation-cache"
python3 ames-v3/checkpoint/restore.py --work "$W"
python3 -c 'import sys; sys.path.insert(0,"ames-v3"); from run import foundation_availability; result=foundation_availability(); print(result); assert result["available"]'
python3 ames-v3/run.py --data data/train.csv --work "$W" --out "$W/out" --jobs 2
```

## Yayımdan önce

1. 12 bölümde 960 Optuna denemesi ve TabPFN adayının bulunduğunu, 48 grubun tamamlandığını doğrulayın. Ayrı bir çıktı dizini kullanın. Tam koşudan sonra `python3 ames-v3/checkpoint/finish_receipts.py --work "$W" --out "$W/out"` komutu bu dosyaları kontrol eder, süre ve tek-ev kayıtlarını oluşturur. Nihai reçeteyi, 2.336 dış tahmini, ileri tarihli 136 satışı ve 292 saklı satışı kontrol edin. 899 numaralı ev için tarihsel komutu toplu sonuçla karşılaştırın. Ayrı bir giriş CSV’sinin tek satır tahminini ve etiketsiz 1461 örneğini de kaydedin.
2. Önceki kayıtların üzerine yalnızca doğrulanmış yeni sonuçları alın. Beş katlı V2 ile üç iç katlı `v2-nested` ayrı kalmalı. `python3 ames-v3/compare_releases.py` yeni ortak CSV, JSON ve sitedeki ev kayıtlarını üretir.
3. [Portföy taslağından](ames-v3/portfolio-draft/) beş web dosyasını (`index.html`, `style.css`, `case-study.js`, `proje-detaylari.html`, `details.css`) tekrar `docs/` altına alın. `python3 ames-v3/render_portfolio.py` ile doğrulanmış değerleri yazın. Ana ve teknik README için `python3 ames-v3/portfolio-draft/render_final_readmes.py --write` çalıştırın ve çıktıları okuyun. Proje 1, 2, 3 bölümlerinin karar ve sonuçları açıkladığını, eski rakam veya yer tutucu kalmadığını kontrol edin.
4. V1 temel kontrollerini, `python3 -m unittest discover -s ames-v2`, `python3 -m unittest discover -s ames-v3` ve `python3 ames-v3/render_portfolio.py --check` çalıştırın. Yeni tam sonuçlar gelene kadar yayın tutarlılığı testleri son aşamayı bekliyor. Mobil ve masaüstünde 292 satırı ve sayfa bağlantılarını karşılaştırın.
5. Yalnızca `v3` dalına gönderin. Yerel commit, uzak `v3` ve GitHub Pages dağıtımını doğrulayın. Başlangıç referansları `main` için `a1cf99b304c604125019a10b587613a5d3b1ee5c`, `v2` için `7473f585685b15cf886daa32041f8f293e887d1e`.
