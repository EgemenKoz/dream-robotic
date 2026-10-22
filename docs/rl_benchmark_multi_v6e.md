# RL Benchmark Sonuçları — `multi_v6e` (SAC, 600k adım)

> Tarih: 2026-10-20 · Robot: Moborobo · Branch: `rl_training` (commit'lenmemiş çalışma ağacı)
>
> Ham sonuçlar: `results/benchmark/benchmark_20261020_223336/`
> (`episodes.csv`, `summary.txt`, `worlds.txt`) · Konsol logu: `results/benchmark/rl_multi_v6e_bench.log`

## Özet

| | Değer |
|---|---|
| Başarı | **42 / 50 dünya (%84)** |
| Çarpışma | 7 dünya (%14) |
| Zaman aşımı | 1 dünya (%2), dünya 90 |
| Başarılı bölümlerde ortalama süre | 50.4 s |
| Başarılı bölümlerde ortalama hız | 0.186 m/s (robotun tavanı 0.2 m/s) |
| Başarılı bölümlerde ortalama yol | 9.38 m (A* referans yolunun ~%83'ü) |

Politika hedefe vardığı her bölümde neredeyse tam hızla ve dolaysız gidiyor. Başarısızlıkların hepsi
parkurun **ilk 3–5 metresinde**, sık engel bölgesinde oluyor. Zorluk arttıkça başarı düşüyor:
kolay %93 → orta %89 → zor %72.

## Ne çalıştırıldı

```bash
python3 -m benchmark.run_benchmark --robot moborobo --planners rl_sac \
    --rl-model results/rl/runs/multi_v6e/model_final.zip
```

- **Model:** `results/rl/runs/multi_v6e/model_final.zip` (SAC, LayerNorm critic, 4 robotlu çoklu-robot
  ortamında 600k adım, kolay → orta → tüm dünyalar curriculum'u).
- **Dünyalar:** standart Moborobo değerlendirme seti `MOBOROBO_EVAL_WORLDS` = 0, 3, 6, …, 147 (50 dünya).
- **Ayarlar:** her dünyada tek bölüm, deterministik aksiyon (politikanın ortalaması), 120 s simülasyon
  zaman sınırı, hedef toleransı 1.0 m.
- **Süre:** yaklaşık 60 dakika (22:33 → 23:33).

## Zorluk seviyesine göre

Seviyeler `rl.curriculum.difficulty_tiers("moborobo")` ile hesaplandı (153 dünya zorluk skoruna göre üçe
bölünüyor).

| Seviye | N | Başarı | Çarpışma | Zaman aşımı | Ort. süre (başarılı) | Ort. hız |
|---|---|---|---|---|---|---|
| Kolay | 14 | %93 (13) | 1 | 0 | 50.6 s | 0.186 m/s |
| Orta | 18 | %89 (16) | 2 | 0 | 49.8 s | 0.188 m/s |
| Zor | 18 | %72 (13) | 4 | 1 | 51.0 s | 0.185 m/s |
| **Toplam** | **50** | **%84 (42)** | **7** | **1** | **50.4 s** | **0.186 m/s** |

## Başarısız dünyalar

| Dünya | Seviye | Sonuç | Bittiği an | Gidilen yol | Hedefe kalan | Eğitim-ortamı eval'i |
|---|---|---|---|---|---|---|
| 60 | kolay | çarpışma | 21.3 s | 3.40 m | 7.43 m | 4/4 başarılı |
| 78 | zor | çarpışma | 21.1 s | 3.23 m | 6.86 m | **0/4** (hepsi çarpışma) |
| 81 | orta | çarpışma | 16.4 s | 2.91 m | 7.18 m | 4/4 başarılı |
| 90 | zor | zaman aşımı | 120 s | 5.06 m | 8.63 m | **0/4** (hepsi çarpışma) |
| 93 | zor | çarpışma | 24.3 s | 3.86 m | 6.72 m | 4/4 başarılı |
| 105 | orta | çarpışma | 18.9 s | 3.49 m | 6.97 m | 4/4 başarılı |
| 117 | zor | çarpışma | 30.7 s | 4.19 m | 6.35 m | yok (eval bu dünyaya gelmeden durdu) |
| 144 | zor | çarpışma | 28.0 s | 4.92 m | 5.42 m | yok |

**Bu tablodan çıkanlar:**

- **Çarpışmalar hep erken oluyor.** 7 çarpışmanın hepsi bölümün 16–31. saniyesinde, başlangıçtan
  2.6–4.6 m ileride, yani engel alanının içinde. Engel alanı bir kez geçilince hiçbir bölüm
  kaybedilmemiş.
- **Dünya 90'da robot takılı kaldı.** 120 s boyunca 5 m yol gitti ama hedefe yalnızca ~1.4 m yaklaştı.
  Bir engel önünde ileri-geri gidip geldiği anlaşılıyor. Eğitimde bu durum 20 s sonra "stuck"
  bitişiyle kesiliyor; benchmark'ta öyle bir kural yok, bu yüzden zaman aşımına kadar sürdü.
- **78 ve 90 iki ortamda da çözülemiyor.** Bunlar politikanın zayıf olduğu dünyalar.
- **60, 81, 93 ve 105 eğitim ortamında 4/4 çözülüyor, benchmark'ta çarpıyor.** Bunlar politikanın
  zayıflığından çok iki ortam arasındaki farktan kaynaklanıyor olabilir (aşağıda).

## Eğitim-ortamı değerlendirmesiyle karşılaştırma

`tools/eval_policy.py` aynı modeli, eğitildiği çoklu-robot ortamında değerlendirmişti
(`results/rl/eval/multi_v6e_final/results.csv`). Her dünyada 4 robot vardı. Değerlendirme 153 dünyanın
116'sında durdu; bu benchmark'ın 50 dünyasından 39'u o 116'nın içinde.

| Bu 39 dünyada | Eğitim ortamı (eval_policy) | Benchmark |
|---|---|---|
| Başarı | %91 (bölüm bazında, 156 bölüm) | %85 (33 / 39) |
| Ortalama başarılı süre | 48.8 s | ~50 s |

Dünya bazında (eval'de ≥3/4 başarı "çözüldü" sayıldı):

| | Benchmark başarılı | Benchmark başarısız |
|---|---|---|
| **Eval'de çözüldü** | 32 | 4 (60, 81, 93, 105) |
| **Eval'de çözülmedi** | 1 (dünya 9) | 2 (78, 90) |

Benchmark eğitim ortamından ~6 puan daha kötü. Fark küçük ama tutarlı.

### Neden fark var — iki ortam aynı değil

| | Eğitim / eval_policy | Benchmark |
|---|---|---|
| Simülasyon akışı | Her adımda **duraklatılıyor**: unpause → komut → 0.1 s → pause → gözlem. Politika düşünürken dünya donuk. | **Gerçek zamanlı**: dünya hiç durmuyor, 10 Hz döngü duvar saatiyle. Gözlem ile aksiyon arasında gecikme ve tarama bayatlığı var. |
| Lidar (RS-16 dikey tarama) | `MULTI_LIDAR_PROFILE`: 64 dikey örnek, −45°…0°, 10 m menzil | Stok profil: 900 dikey örnek, ±155°, 150 m menzil (`/scan`'e çevrilince 10 m) |
| Robot sayısı | 4 robot (birbirine "hayalet") | 1 robot |
| Çarpışma testi | Geometrik: gövde kutusu ↔ silindirler, 1 cm pay | Gazebo temas sensörü (`collision_publisher`) |
| Takılma | 20 s'de 0.3 m'den az ilerlerse bölüm biter (stuck) | Kural yok, 120 s zaman aşımına kadar sürer |

Erken çarpışmaların hepsi sık engel bölgesinde, engellere çok yakın manevralar sırasında oluyor. Bu
yüzden en olası iki sebep: (1) gerçek zamanlı çalışmanın getirdiği gecikme, (2) farklı lidar
profilinin yakın mesafede farklı bir `/scan` üretmesi. Hangisinin baskın olduğu bu koşuyla
ayrıştırılamıyor.

## Metrikleri okurken dikkat

- **`NavMetric` (BARN skoru) Moborobo için anlamsız derecede düşük.** Formül
  `optimal_time = yol / NOMINAL_SPEED` kullanıyor ve `NOMINAL_SPEED = 2.0 m/s` Jackal'a göre ayarlı.
  Moborobo en fazla 0.2 m/s gittiği için süre her zaman üst sınır olan `8 × optimal_time`'a kırpılıyor
  ve başarılı her bölüm tam **0.125** alıyor. Ortalama 0.1055 = 0.84 × 0.125, yani başarı oranından
  başka bilgi taşımıyor. Moborobo'da bu skoru karşılaştırma için kullanmamak lazım.
- **SPL tüm başarılı bölümlerde 1.0.** Referans yol, 8-bağlantılı A* "merdiveni"; robot ondan kısa bir
  yol izliyor (ortalama %83'ü). SPL `opt / max(gerçek, opt)` olduğu için 1'de doyuyor. Bu yüzden SPL de
  burada başarıyla aynı şeyi söylüyor.
- **`MinClr` ≈ 0.45 m gerçek bir mesafe değil.** `/scan`'i üreten `pointcloud_to_laserscan`
  `range_min: 0.45` ile ayarlı (`robots/moborobo/launch/barn_move_base.launch`). Bu yüzden 0.45 m'den
  yakın hiçbir şey okunmuyor ve neredeyse tüm bölümlerde değer bu tabanda kalıyor.
- **Bu dünyalar eğitimde görüldü.** Moborobo veri setinde 153 dünya var ve curriculum 200k adımdan
  sonra hepsini kullanıyor. Yani bu benchmark *öğrenilmiş* dünyalardaki performansı ölçüyor,
  genellemeyi değil. Genelleme için eğitimde hiç görülmemiş bir harita gerekiyor (bkz. `tammap`).
- **Her dünyada tek bölüm.** Aksiyon deterministik olsa da simülatör gürültüsü sonucu değiştirebiliyor
  (örneğin dünya 12 ve 21 eval'de 3/4). Tek koşuluk sonuçları dünya bazında kesin hüküm gibi okumamak
  lazım; toplam oranın ±5 puan oynaması normal.

## Sonraki adımlar

1. **Klasik planlayıcılarla karşılaştırma.** `results/benchmark/` altında henüz klasik planlayıcı
   sonucu yok. Aynı 50 dünyada `dwa mppi elastic_band` koşmak %84'ün iyi mi kötü mü olduğunu
   gösterir (~2–3 saat):
   ```bash
   python3 -m benchmark.run_benchmark --robot moborobo --planners dwa mppi elastic_band rl_sac \
       --rl-model results/rl/runs/multi_v6e/model_final.zip
   ```
2. **Ortam farkını ayrıştırmak.** Başarısız 4 dünyayı (60, 81, 93, 105) benchmark'ta önce eğitimdeki
   lidar profiliyle koşmak. Çarpışmalar kalkarsa sebep lidar, kalkmazsa gerçek zamanlı gecikme.
   Gecikme çıkarsa eğitime rastgele gecikme eklemek (latency randomization) iyi bir çözüm olur.
3. **Genelleme testi: `tammap_barn.world`.** ~29 × 32 m, 1872 silindir, engeller BARN'dakiyle aynı tip
   (r = 0.075 m, h = 1 m). Başlangıç ve hedef noktaları belirlenmeli. Dikkat: gözlemdeki hedef mesafesi
   10 m'de kırpılıyor (`dist_norm = clip(d / 10, 0, 1)`), yani 10 m'den uzak bir hedef politikaya
   "10 m" gibi görünür. Politika yönü yine bilir ama bu durum eğitimde hiç görülmedi. Uzun rotalarda
   ara hedefler (waypoint) vermek gerekebilir.
4. **Benchmark'a takılma (stuck) kuralı eklemek.** Dünya 90 gibi bölümler 120 s boşa harcıyor;
   ayrı bir `stuck` sonucu sayılmaları raporu da netleştirir.
