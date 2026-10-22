# RL — Soru / Cevap (rl_deep_dive.md'nin devamı)

> Bu dosya şu sorulara tek tek cevap verir (sırası soruldukları gibi):
>
> 1. [`unsafe` nedir?](#1-unsafe-nedir)
> 2. [Timeout reward ve running reward neden yok? Olsa iyi olmaz mı?](#2-timeout-reward-ve-running-reward-neden-yok)
> 3. [Step başı 0.1 s tam olarak nasıl oluyor? Fizik durup iş bitince mi devam ediyor?](#3-step-başı-01-s-tam-olarak-nasıl-oluyor)
> 4. [0.1 s'nin yeterli olduğunu nasıl biliyoruz?](#4-01-snin-yeterli-olduğunu-nasıl-biliyoruz)
> 5. [150 adım görülmeme (SCAN_MEM_MAX_AGE) fazla değil mi? Dürüst yorum](#5-150-adım-görülmeme-fazla-mı--dürüst-yorum)
> 6. [Robot neden geri gidemiyor? (kısa — detayı sonra)](#6-robot-neden-geri-gidemiyor-kısa-cevap)
> 7. [Timeout / truncate tam olarak ne?](#7-timeout--truncate-tam-olarak-ne)
> 8. [Hiperparametreler — detaylı](#8-hiperparametreler--detaylı)
> 9. [Headless yerine GUI ile çalıştırıp robotu izleyebilir miyim?](#9-headless-yerine-guiyle-izleyebilir-miyim)
>
> Satır numaraları `rl_training` branch'inin şu anki (commit'lenmemiş) çalışma ağacına göredir.

---

## 1. `unsafe` nedir?

**Kısaca:** Robot bir engele **fiziksel olarak çarpmadan önce**, "çok yaklaştın" diye episode'u
erken bitiren yapay bir güvenlik sınırı. Çarpışmanın "yumuşak" versiyonu.

**Kod:** `rl/env/barn_env.py:502-508`
```python
elif self._safety_terminate and min_range_m < self._safety_dist:
    reward += self._safety_penalty      # -40
    terminated = True
    outcome = "unsafe"
```

**Parametreler** (`configs/rl_config.py:166-176`):

| Parametre | Değer | Anlamı |
|---|---|---|
| `SAFETY_TERMINATE` | `True` | Özellik açık/kapalı |
| `SAFETY_DIST` | **0.42 m** | Bu mesafenin altı = unsafe |
| `SAFETY_PENALTY` | **−40** | Collision'dan (−50) daha hafif |
| `SAFETY_ARC` | 2.0944 rad = **±120°** | Sadece önde/yanlarda bakılır, arka 120° hariç |

**`min_range_m` nedir?** Scan memory ile birleştirilmiş (fused) mesafelerin, robotun önündeki
±120°'lik yaydaki minimumu (metre). `barn_env.py:450, 943-947`.

**Neden var?** (`rl_config.py:150-165` yorumu)
1. **Kör bölge problemi.** RS-16 LiDAR 0.45 m'den yakını göremiyor. Moborobo'nun ön tamponu
   base_link'ten 0.387 m'de. Yani gerçek temas, sensörün **göremediği** bir halkanın içinde
   oluyor. Critic'ten "gözlemde görünmeyen" bir olayı tahmin etmesi isteniyordu → öğrenmesi zor.
   0.42 m ise scan memory sayesinde **gözlemde ifade edilebilen** bir mesafe.
2. **Nadir olay problemi.** Çarpışmalar geçişlerin sadece %0.126'sıydı (204k adımda 258).
   100k buffer + 256 batch ile gradient update'lerinin ~%72'sinde hiç çarpışma örneği yoktu.
   Unsafe daha sık tetiklenir → value fonksiyonu "tehlike" kavramını daha iyi öğrenir.

**Neden −40 (collision'dan hafif)?** "Durup/kaçınıp unsafe olmak" her zaman "çarpmaktan" iyi
olsun diye. Sıralama: success (+100) > timeout (0, ama bootstrap) > unsafe (−40) > collision (−50).

**Neden arka 120° hariç?** Geçilmiş engeller robotun arkasında kör halkada kalıp hafızada
duruyor; tüm 360° baksaydık robot bir engeli geçtikten sonra sürekli unsafe tetiklenirdi.

**Dikkat edilecekler (dürüst notlar):**
- Marj çok ince: ön tampon 0.387 m, sınır 0.42 m → **~3.3 cm**. Moborobo tam hızda adım başına
  2 cm ilerliyor → sınır ile temas arası **~1.5 adım**. Yani unsafe "erken uyarı" değil,
  "son saniye" sınırı.
- Dönme yarıçapı 0.4528 m > 0.42 m. Robot yerinde dönerken köşesi 0.42–0.45 m arası bir
  engele sürtebilir ve bu unsafe'i tetiklemeden collision olur. Yan tarafta koruma zayıf.
- `SCAN_MEMORY=False` yapılırsa fused mesafe asla 0.45'in altına inmez → **unsafe hiç
  tetiklenmez** (deep dive §5).
- Benchmark'ta unsafe **yok** — orada sadece gerçek collision/success/timeout var
  (`benchmark/benchmark_env.py:218-230`). Yani eğitimde "unsafe" diye biten bir episode,
  benchmark'ta devam edip ya kurtulur ya da çarpar.

---

## 2. Timeout reward ve running reward neden yok?

### 2a. "Running reward" aslında VAR

`outcome = "running"` sadece bir etiket; o adımın ödülü **dense reward**'ın kendisidir
(`barn_env.py:494`):
```
r = r_step (−0.01) + r_progress + r_prox + r_heading + r_path
```
Yani her "running" adımda robot zaten ödül/ceza alıyor. Ayrı bir "running reward" ile
kastedilen muhtemelen **"hayatta kalma bonusu"** (her adım +c). Bu **bilinçli olarak yok**, çünkü:

- Navigasyonda pozitif alive-bonus, **mümkün olduğunca uzun yaşamayı** ödüllendirir →
  robot yerinde durur/döner, hedefe gitmez (klasik "donma" problemi). Bu projede zaten
  donma bir kez yaşanmış (`rl_config.py:225-232`: timeout %2.5 → %17).
- Tam tersi, **küçük negatif** step cezası (−0.01) var: "oyalanma" diye.

### 2b. Timeout reward neden 0?

Timeout'a ceza vermek sezgisel olarak mantıklı görünür ama **teknik olarak yanlış** olur:

1. **Timeout MDP'nin parçası değil.** Gözlemde "kalan süre" bilgisi yok (`obs["state"]` =
   dist, sin, cos, last_lin, last_ang). Robot aynı durumda 500. adımda da 999. adımda da aynı
   gözlemi görür. Eğer 1000. adımda −X verirsen, critic aynı gözlem için bazen −X bazen 0
   görür → bu ceza critic için **açıklanamayan gürültü** olur (non-Markov reward).
2. **Truncation'da bootstrap yapılıyor** (bkz. §7). SB3, timeout adımında "episode bitti,
   gelecek = 0" demiyor, "gelecek = V(s')" diyor. Timeout'a ceza eklemek bu mantıkla çelişir:
   hem "devam ediyormuş gibi" değer hesaplarsın hem de "bitti" cezası verirsin.
3. **Zaten dolaylı ceza var.** Timeout'a giden bir episode 1000 adım boyunca step (−0.01),
   heading, path ve prox cezası biriktiriyor ve terminal +100'ü **hiç almıyor**. Başarıya giden
   yol her zaman daha kârlı.

**Olsa iyi olur muydu?** Bence mevcut haliyle **hayır**. Eğer ileride "robot yavaş/donuyor"
diye timeout oranı artarsa doğru müdahaleler şunlar olur:
- `REWARD_STEP`'i biraz büyütmek (−0.01 → −0.02 gibi),
- ya da gözleme **kalan süre oranı** (`1 − step/max_steps`) eklemek; o zaman timeout
  Markov olur ve `terminated=True` + ceza meşru hale gelir. Ama bu, benchmark'ta (120 s sim
  zamanı, adım sayısı değil) farklı bir değerle çalışacağı için deploy tarafında da uyum ister.

---

## 3. Step başı 0.1 s tam olarak nasıl oluyor?

Kod (`barn_env.py:433-437`):
```python
self._gazebo_sim.unpause()        # 1) fiziği başlat  (ROS servis çağrısı)
self._pub_cmd(lin, ang)           # 2) /cmd_vel yayınla
time.sleep(self.step_duration)    # 3) 0.1 s DUVAR SAATİ bekle
self._gazebo_sim.pause()          # 4) fiziği durdur  (ROS servis çağrısı)
# 5) bundan sonrası fizik DURMUŞKEN olur:
#    get_model_state, scan memory, obs, reward, SB3 gradient update...
```

**Sorunun cevabı: evet, tam olarak tahmin ettiğin gibi.** Fizik sadece `sleep` penceresinde
akıyor. Sonra Gazebo **pause** ediliyor ve:
- pose okuma, obs üretme, reward hesabı,
- SB3'ün replay buffer'a yazması ve **gradient update**'i (SAC'ta her adımda 1 update),
- loglama

hepsi **fizik donmuşken** yapılıyor. Bunlar ne kadar uzun sürerse sürsün simülasyon zamanı
ilerlemiyor; robot da bu arada hareket etmiyor (hızı dahil tüm durum donuyor). Bir sonraki
`step()`'te unpause edilince kaldığı yerden devam ediyor.

Yani "yetişebildiği kadar devam" **değil**: iş ne kadar sürerse sürsün, beklenir. Hesap
süresi sadece **eğitimin duvar saati süresini** uzatır, robotun davranışını etkilemez.

**Ama bu 0.1 s kesin değil — yaklaşık.** Neden:

| Etken | Etki |
|---|---|
| `time.sleep` **duvar saati** ölçer, sim zamanı değil | Sim zamanı = 0.1 s × RTF (real-time factor). Dünya dosyaları RTF=1 hedefliyor (`max_step_size=0.001`, `real_time_update_rate=1000`), ama CPU yetişemezse RTF<1 olur → robot adım başına **daha az** yol alır. |
| unpause/pause servis çağrılarının gecikmesi | Birkaç ms. unpause ile cmd_vel arasında fizik birkaç ms **önceki komutla** akar; pause çağrısı dönene kadar da birkaç ms fazladan akar. |
| `time.sleep` jitter'ı | Linux'ta genelde < 1 ms, ihmal edilebilir. |

**LiDAR senkronizasyonu (önemli, dürüst not):** RS-16 LiDAR **10 Hz** yayın yapıyor
(`robots/moborobo/urdf/RS-16.urdf.xacro:51`), step de 10 Hz. Kod, step sonrası **taze bir scan
beklemiyor** ve scan'in zaman damgasına bakmıyor; `_latest_scan` neyse onu kullanıyor
(`barn_env.py:~875-896`). Sonuçları:
- Obs'taki scan, pose'a göre **0–100 ms eski** olabilir (0.1 s'lik pencerenin herhangi bir
  anında çekilmiş olabilir). Hatta pencerede hiç scan gelmezse bir önceki adımınki kullanılır.
- Scan memory noktaları dünya koordinatına **şu anki** pose ile çeviriyor → küçük kayma:
  düz giderken ≤ 2 cm, 1 rad/s dönerken 0.45 m'de ~4.5 cm yanal hata.
- Bu, "fresh" maskesinin de bazen eski bir scan'i "taze" işaretlemesi demek.

Bu bir bug değil ama ölçülmesi gereken bir belirsizlik (bkz. §4'teki ölçüm scripti).

**Benchmark'ta farklı:** `benchmark_env.py:212-285` fiziği **durdurmuyor**. Gerçek zamanlı bir
döngü: her 0.1 s'de bir (`_CTRL_DT = 0.1`, `benchmark_env.py:54`) pose + scan okunur, action
hesaplanır, yayınlanır, periyodun kalanı kadar uyunur. Orada hesaplama süresi **gerçekten**
önemli (hesap uzarsa robot eski komutla gitmeye devam eder). SAC inference ~1 ms, scan memory
~0.3 ms olduğu için pratikte sorun değil.

---

## 4. 0.1 s'nin yeterli olduğunu nasıl biliyoruz?

**Dürüst cevap: kesin olarak bilmiyoruz; ölçülmemiş, makul varsayım olarak seçilmiş.** Repoda
0.1'i doğrulayan bir test veya deney yok; gerekçesi sadece `rl_config.py:76-80` yorumu
("~10 Hz policy frequency"). Yine de neden **makul** olduğunu şu argümanlar destekliyor:

1. **Sensörle eşleşiyor.** LiDAR 10 Hz. Daha sık karar vermek (örn. 20 Hz) yeni bilgi
   getirmez; iki adımdan biri aynı scan'i görür.
2. **Hareket çözünürlüğü.** Moborobo 0.2 m/s → adım başına **2 cm**; 1 rad/s → adım başına
   **5.7°**. BARN koridorları ~0.5–1 m; 2 cm'lik adımlarla manevra için bol çözünürlük var.
3. **Literatür.** BARN/DRL navigasyon çalışmalarının çoğu 5–10 Hz kullanıyor.
4. **Benchmark'la tutarlı.** Benchmark da 0.1 s ile kontrol ediyor (`_CTRL_DT`), yani
   eğitim ve deploy aynı frekansta.

**Zayıf nokta:** Güvenlik marjı. Unsafe sınırı ile temas arası ~3.3 cm = tam hızda **~1.5
adım**. Yani politikanın "tehlikeyi gör → tepki ver" döngüsü için neredeyse hiç yedek adım yok.
0.1 s'den çok daha büyük step (örn. 0.2 s) seçilseydi bu marj tek adıma inerdi — **bu yüzden
büyütmemek doğru**. Küçültmek (0.05 s) ise sensör 10 Hz olduğu için anlamsız.

**Nasıl ölçeriz?** Şunu gerçek bir koşuda ölçmek önerilir (script değil, fikir; istersen
yazarım):
```python
# step() içine geçici olarak:
t_sim0 = rospy.get_time()          # use_sim_time=True → sim saati
scan_count0 = self._scan_msg_count # _scan_cb içinde sayaç artır
self._gazebo_sim.unpause(); self._pub_cmd(lin, ang)
time.sleep(self.step_duration); self._gazebo_sim.pause()
dt_sim = rospy.get_time() - t_sim0          # beklenen ≈ 0.100
n_scans = self._scan_msg_count - scan_count0 # beklenen ≈ 1
```
Birkaç bin adım boyunca `dt_sim` dağılımı (ortalama/min/max) ve `n_scans == 0` oranı
loglanırsa: RTF gerçekten ~1 mi, scan'ler kaçıyor mu net görülür. `dt_sim` belirgin şekilde
0.1'in altındaysa (CPU yetişmiyor) veya `n_scans == 0` sık oluyorsa, daha sağlam çözüm:
`sleep` yerine **sim saatine göre beklemek** (`/clock` 0.1 s ilerleyene kadar) ve step
sonunda **yeni bir scan gelmesini beklemek**.

---

## 5. 150 adım görülmeme fazla mı? — Dürüst yorum

**Parametre:** `SCAN_MEM_MAX_AGE = 150` adım = 15 s (`rl_config.py:74`). Bir hafıza noktası
150 adım boyunca **yeniden doğrulanmazsa** (aynı 5 cm voxel'e yeni bir ölçüm düşmezse)
silinir (`scan_memory.py:216-218`).

**Önce şunu netleştirelim: bu yaş sınırı çoğu nokta için hiç devreye girmiyor.** Noktalar
başka yollarla zaten siliniyor:
- **Raycast temizleme:** 0.50 m'den uzaktaki bir nokta, içinden bir ışın geçerse hemen silinir.
- **Mesafe budama:** robottan 4 m'den uzaklaşan nokta silinir.
- **Yeniden görülme:** LiDAR 360° olduğu için görüş alanındaki gerçek engeller her adım
  yeniden doğrulanır; yaşları sürekli sıfırlanır.

Yani yaş sınırı pratikte sadece **kör halkanın içindeki (< 0.45–0.50 m)** ve **perdelenmiş**
noktalar için önemli — yorum da bunu söylüyor: "safety valve against phantoms parked inside
the blind ring" (`rl_config.py:71-73`).

**Burada gerçek bir takas var:**

| Yaş kısa olursa (örn. 30 = 3 s) | Yaş uzun olursa (150 = 15 s) |
|---|---|
| ✗ Robot bir duvarın dibinde **takılıp yerinde döndüğünde** (tam da en tehlikeli an), kör halkadaki gerçek duvar 3 s sonra unutulur → robot "önüm boş" sanıp çarpar. Bu, scan memory'nin varlık sebebini baltalar. | ✗ Kör halkaya düşmüş **hayalet/kaymış** noktalar 15 s kalır → gereksiz prox cezası / yanlış unsafe. |

**Benim dürüst yorumum:** 150 **fazla değil**, hatta BARN için mantıklı tarafta.
- BARN **statik** bir ortam: hareket eden engel yok. "Gerçekte olmayan bir engeli hatırlamak"
  sadece **kayıt hatasından** (pose–scan zaman kayması, §3) doğabilir; bu da gerçek duvarın
  birkaç cm yanında bir "iz" bırakır, boş alanın ortasında hayalet değil. Etkisi: engeller
  birkaç cm **daha yakın** görünür → muhafazakâr taraf.
- Kısa yaşın zararı (gerçek engeli unutmak → çarpışma) uzun yaşın zararından (birkaç cm fazla
  temkin) çok daha büyük.

**Asıl riskim şurada:** 0.52 m'lik en dar boğazlarda, birkaç cm'lik kayma izi ±120° yayda
0.42'nin altına düşerse **yanlış unsafe** tetiklenebilir. Bu, 150'nin değil, §3'teki zaman
kaymasının sorunu; ama yaş uzun olduğu için iz birikir.

**Öneri (tahmin değil ölçüm):** Unsafe ile biten episode'larda tetikleyen sektörün
`fresh=0` (hafızadan) olup olmadığını ve o noktanın yaşını logla. Eğer unsafe'lerin büyük
kısmı "eski hafıza noktası" kaynaklıysa 150'yi 50–75'e indirip (5–7.5 s) ablation yap. Aksi
halde dokunma.

---

## 6. Robot neden geri gidemiyor? (kısa cevap)

Detayına sonra gireceğimizi söyledin; sadece "ne ve nerede" kısmı:

- **Tasarım kararı:** action space'in lineer alt sınırı 0 (`barn_env.py:281-285`):
  `low=[0.0, -max_angular]`.
- **Gerekçe** (`docs/rl_structure.md:141-142`): deploy edilen diğer planner'larla aynı olmak
  (MPPI'de de `MIN_V = 0.0`, `configs/mppi_config.py:72`) ve RL'nin "her köşeden geri geri
  çıkma" gibi patolojik davranışlar öğrenmesini engellemek.
- **Bedeli:** Sıkıştığında geri çıkamıyor; tek kaçış yolu yerinde dönmek. Dönme yarıçapı
  (0.4528 m) kör halkadan büyük olduğu için bu da riskli (§1).
- Yan etki: SAC'ın tanh çıktısı [0, 0.2]'ye lineer ölçeklendiği için ağ çıktısı 0 → 0.1 m/s.
  Yani "sıfır aksiyon" aslında **yavaş ileri**.

---

## 7. Timeout / truncate tam olarak ne?

Gymnasium'da `step()` iki ayrı "bitti" bayrağı döner:

| Bayrak | Anlamı | Bu projede | Değer hesabında |
|---|---|---|---|
| `terminated=True` | Episode **ortamın doğası gereği** bitti; gerçek bir son durum. Sonrası yok. | collision, unsafe, success | Hedef = `r` (gelecek = 0) |
| `truncated=True` | Episode **dışarıdan kesildi** (süre doldu); aslında devam edebilirdi. | timeout (`_step_count ≥ 1000`) | Hedef = `r + γ·V(s')` (bootstrap) |

**Neden ayrım önemli?** Robot 1000. adımda hedefe 3 m uzakta, güzel ilerliyorsa, o durumun
değeri **yüksek** olmalı (biraz daha zaman olsa varacaktı). Eğer timeout'u `terminated` gibi
işleseydik critic "bu durumdan sonra hiçbir şey gelmez" diye öğrenirdi → aynı gözlem bazen
"gelecek var" bazen "gelecek yok" → değer fonksiyonu bozulur.

**Kodda nasıl işliyor:**
1. `BarnEnv.step()` timeout'ta `truncated=True, terminated=False` döner (`barn_env.py:513-517`).
2. SB3 2.4.1 `DummyVecEnv` bunu `info["TimeLimit.truncated"] = truncated and not terminated`
   olarak işaretler.
3. SAC'ın replay buffer'ı (`handle_timeout_termination=True`, SB3 varsayılanı) bu geçişi
   `done=False` gibi saklar → `r + γ·Q(s', a')` ile bootstrap yapılır.
   PPO'da SB3 aynı işi rollout toplarken `r += γ·V(terminal_obs)` ekleyerek yapar.
4. Terminal potansiyel düzeltmesi (`+12.5 × d`) truncation'da **uygulanmaz**
   (`barn_env.py:541`), çünkü kalan potansiyeli zaten `V(s')` hesaba katıyor.

**Pratik sonuç:** Log'lardaki `ep_rew_mean` / `total_reward` timeout episode'ları için
**eksik** bir sayıdır (bootstrap kısmı yok). O yüzden timeout episode'larının return'ü bazen
çarpışmalardan düşük görünür; bu bug değil. Başarıyı **outcome oranlarından** oku.

**Adım sayma detayı:** Sayaç sadece terminal olmayan adımlarda artar, timeout 1000. `step()`
çağrısında tetiklenir (detay: `rl_deep_dive.md §3.4`).

---

## 8. Hiperparametreler — detaylı

Üç grup var: **(A) SAC algoritması**, **(B) ağ mimarisi**, **(C) ortam/reward**. Ayrıca
**(D) PPO**. Hepsinin CLI'dan değişip değişmediği belirtildi.

### (A) SAC — `train_sac.py:328-343`

| Parametre | Değer | Ne yapar | Bu değer neden / yorum |
|---|---|---|---|
| `learning_rate` (`--lr`) | 3e-4 | Adam optimizer'ın adım büyüklüğü (actor, critic ve α için aynı). | SAC makalesinin ve SB3'ün standart değeri. Büyütmek hızlandırır ama critic'i kararsızlaştırır. |
| `batch_size` (`--batch-size`) | 256 | Her gradient update'te buffer'dan rastgele çekilen geçiş sayısı. | Standart. Nadir olayların (collision/unsafe) batch'e girme olasılığını belirler (bkz. §1: çarpışma %0.126 iken batch'lerin %72'si çarpışmasızdı). |
| `buffer_size` (`--buffer-size`) | 300 000 (2026-10-13'e kadar 100 000) | Replay buffer kapasitesi (FIFO). Dolunca en eski geçiş silinir. | 200k adımlık eğitimde son **100k** adım hatırlanır → eğitimin ilk yarısı unutulur. ~0.3 GB RAM. Tüm eğitimi tutmak için 200k yapılabilir (~0.6 GB). |
| `learning_starts` (`--learning-starts`) | 1 000 | İlk 1000 adımda hiç update yapılmaz, **rastgele** aksiyonla buffer doldurulur. | ≈ 1–2 episode. Başlangıçta çeşitli veri toplamak için. |
| `gamma` γ (`--gamma`) | **0.99** (2026-10-13'e kadar 0.999; o ayarda kritik Q ≈ +716'ya kaydı) | Gelecek ödüllerin indirim katsayısı. Efektif ufuk ≈ 1/(1−γ) = 100 adım. | En kritik ayar. 0.99 → ufuk 100 adım = 10 s = **2 m**; hedef ~500 adım ötede → görünmez, robot donar. 0.999 → ufuk = episode bütçesi. Bedeli: Q-değerleri büyük ve daha gürültülü, öğrenme daha yavaş. |
| `tau` τ (`--tau`) | 0.005 | Target critic'in yumuşak güncelleme oranı: `θ_target ← τ·θ + (1−τ)·θ_target`. | Standart. Target ağ ~200 update'te yetişir; critic'i stabilize eder. |
| `ent_coef` α (`--ent-coef`) | `auto` | Entropi (keşif) ağırlığı. `auto` → α otomatik ayarlanır ki politikanın entropisi hedefte kalsın. | Hedef entropi = −dim(A) = **−2** (SB3 varsayılanı). Sabit sayı verilebilir (örn. `0.1`). |
| `train_freq` | 1 (SB3 default, CLI yok) | Kaç env adımında bir update. | Her adımda 1. |
| `gradient_steps` | 1 (SB3 default, CLI yok) | Her update'te kaç gradient adımı. | Update-to-data oranı 1:1. |
| `total_timesteps` (`--total-steps`) | 200 000 | Toplam env adımı. | ≈ 200–400 episode; duvar saatiyle ≥ 5.6 saat. |
| `n_envs` | 1 | Paralel ortam sayısı. | Tek Gazebo. |
| `target_update_interval` | 1 (SB3 default) | Target ağın kaç update'te bir güncellendiği. | — |

**Eğitim akışına etki eden diğer CLI parametreleri** (`train_sac.py:288-327`):

| Parametre | Varsayılan | Anlamı |
|---|---|---|
| `--robot` | moborobo | jackal / moborobo |
| `--worlds` | robotun 50 eval dünyası | Eğitim dünyaları (⚠ benchmark ile aynı liste — genelleme ölçülmez) |
| `--episodes-per-world` | 5 | Bir dünyada kaç episode sonra sıradakine geçilir |
| `--curriculum` | kapalı | easy → easy+medium → hepsi, 3 eşit faz |
| `--save-every` | 20 000 | Checkpoint (+ replay buffer) sıklığı |
| `--log-every` | 2 000 | Konsol ilerleme satırı sıklığı |
| `--resume` | — | Checkpoint'ten devam |

### (B) Ağ mimarisi — `configs/rl_config.py` `RLModelConfig` (CLI yok)

| Parametre | Değer | Anlamı |
|---|---|---|
| `CONV_LAYERS` | `[(32,5,2), (32,3,2)]` | Scan üzerinde 2 adet 1D-conv (kanal, kernel, stride). 90 sektör → 23 pozisyon, alıcı alan ≈ 9 sektör ≈ 36°. |
| `SCAN_FEATURES` | 128 | CNN kolunun çıkış genişliği |
| `STATE_FEATURES` | 32 | State (5,) kolunun çıkış genişliği |
| `NET_ARCH` | `[256, 256]` | Actor ve her iki critic'in gizli katmanları |

### (C) Ortam / reward — `configs/rl_config.py` `RLEnvConfig` (CLI yok)

**Gözlem:**

| Parametre | Değer | Anlamı |
|---|---|---|
| `NUM_SCAN_RAYS` | 90 | Ham scan (~722 ışın) min-pool ile 90 sektöre (4°) indirilir |
| `SCAN_STACK` | 3 | Son 3 frame (0.3 s geçmiş) — yaklaşma hızını çıkarmak için |
| `SCAN_CLIP_RANGE` | 5.0 m | Normalize paydası; 5 m ötesi = 1.0 |
| `SCAN_MEMORY` | True | Kör halka için nokta hafızası + fresh mask kanalı |
| `SCAN_BLIND_RADIUS` | 0.45 m | Bu yarıçap içindeki noktalar raycast ile silinmez |
| `SCAN_MEM_RADIUS` | 4.0 m | Daha uzak noktalar unutulur |
| `SCAN_MEM_VOXEL` | 0.05 m | Hafıza tekilleştirme ızgarası |
| `SCAN_MEM_MAX_AGE` | 150 adım | Yeniden görülmeyen nokta ömrü (§5) |

**Zamanlama:**

| Parametre | Değer | Anlamı |
|---|---|---|
| `STEP_DURATION` | 0.1 s | Adım başı fizik penceresi (§3, §4) |
| `MAX_EPISODE_STEPS_MOBOROBO` | 1000 | 100 s bütçe |
| `MAX_EPISODE_STEPS_JACKAL` | 500 | 50 s bütçe |

**Reward** (tam formül: `rl_deep_dive.md §7`):

| Parametre | Değer | Anlamı |
|---|---|---|
| `REWARD_GOAL` | +100 | Hedefe ≤ 1 m |
| `REWARD_COLLISION` | −50 | Gerçek temas |
| `SAFETY_PENALTY` | −40 | Unsafe (§1) |
| `REWARD_STEP` | −0.01 | Her adım oyalanma cezası |
| `REWARD_PROGRESS_SCALE_MOBOROBO` | 12.5 | `12.5 × (önceki mesafe − şimdiki mesafe)`; tam hızda +0.25/adım |
| `REWARD_PROGRESS_SCALE_JACKAL` | 5.0 | Aynı +0.25/adım'ı jackal'da vermek için |
| `REWARD_PROX_SCALE` | 0.15 | Engel yakınlık cezası maksimumu |
| `PROX_RAMP_START_M` / `PROX_FLOOR_M` | 0.90 / 0.50 m | Ceza 0.9 m'de başlar, 0.5 m'de doyar (karesel, hızla ağırlıklı) |
| `SAFETY_DIST` / `SAFETY_ARC` | 0.42 m / ±120° | §1 |
| `REWARD_HEADING_SCALE` | 0.02 | `−0.02 × |hedef açısı|/π` |
| `PATH_FOLLOW_SHAPING` | True | A* yolundan sapma cezası açık |
| `REWARD_PATH_SCALE` / `PATH_MAX_DEV` | 0.15 / 1.0 m | `−0.15 × min(sapma, 1)/1` |
| `REWARD_SHAPING` | True | False → sadece step + terminal (sparse ablation) |
| `TERMINAL_POTENTIAL_ZERO` | **False** | Sadece ablation. Açıkken terminalde `+12.5 × d` eklenir ve ilerleme teşvik olmaktan çıkar |
| `STUCK_TERMINATION` / `STUCK_WINDOW_STEPS` / `STUCK_MIN_DISPLACEMENT_M` | True / 200 / 0.3 m | 200 adımda 0.3 m'den az yer değiştirme → `stuck` terminali |
| `REWARD_STUCK` | −50 | Stuck terminal ödülü (çarpışmayla eşit: pes etmek denemekten ucuz olmamalı) |
| `REVERSE_SPEED_RATIO` | 0.3 | Aralık [−0.06, 0.2]; simetrik aralıkta eğitilmemiş politika yerinde titriyor |

**Ölçek karşılaştırması (neyin baskın olduğu):** Bir adımda tipik değerler:
progress **+0.25** (tam hız, hedefe doğru) · path **−0.067** (ortalama 0.45 m sapma) ·
prox **0 … −0.15** · heading **0 … −0.02** · step **−0.01**. Yani iyi sürüşte progress net
baskın; engel dibinde prox progress ile aynı mertebeye çıkıyor — tasarım hedefi bu.

### (D) PPO — `train_ppo.py:215-245`

| Parametre | Değer | Anlamı |
|---|---|---|
| `n_steps` | 2048 | Her update öncesi toplanan rollout uzunluğu (~2 episode) |
| `batch_size` | 64 | Minibatch; `n_steps`'i bölmeli |
| `n_epochs` | 10 | Her rollout üzerinden kaç tur gradient |
| `learning_rate` | 3e-4 | — |
| `gamma` | 0.99 | SAC ile aynı gerekçe |
| `gae_lambda` | 0.95 | Avantaj tahmininde bias/varyans takası. Not: γ·λ ≈ 0.949 → avantaj efektif olarak ~20 adıma bakar; uzun vade value fonksiyonundan gelir. |
| `clip_range` | 0.2 | PPO'nun politika değişim sınırı |
| `vf_coef` | 0.5 | Value loss ağırlığı |
| `ent_coef` | 0.01 | Sabit entropi bonusu (SAC'ın aksine otomatik değil) |
| `total_steps` | 500 000 | PPO on-policy olduğu için daha çok veri ister |

---

## 9. Headless yerine GUI'yle izleyebilir miyim?

**Evet.** Üç yol var; en kolayı kod değiştirmeden:

### Yol 1 — Çalışan eğitime sonradan `gzclient` bağlamak (kod değişikliği yok) ✅ önerilen

Eğitim headless `gzserver` başlatıyor. Gazebo'da GUI (`gzclient`) ayrı bir süreç ve çalışan
bir server'a **istediğin an** bağlanabilir. Eğitim çalışırken ikinci bir terminalde:
```bash
source /opt/ros/noetic/setup.bash
# (projenin/robotun Gazebo ortam değişkenleri gerekiyorsa aynı şekilde source et —
#  modeller/mesh'ler bulunamazsa robot gri kutu görünür, fonksiyonel sorun değil)
gzclient
```
Kapatınca eğitim etkilenmez. Dikkat:
- Her **dünya değişiminde** (5 episode'da bir) gzserver yeniden başlatılıyor
  (`barn_env.py:603-610`, `_kill_processes` gzclient'ı da öldürüyor: `:746`). O zaman
  `gzclient`'ı yeniden açman gerekir.
- Fizik adımlar arasında pause olduğu için robotu **0.1 s'lik kesik hareketlerle** görürsün;
  gradient update sırasında donuk durur. Bu normal.
- GUI GPU/CPU yer; RTF düşebilir (§3) → izlerken eğitim birebir aynı koşulda olmayabilir.

### Yol 2 — `gui=True` ile başlatmak (küçük kod değişikliği)

`BarnEnv` zaten `gui` parametresini destekliyor (`barn_env.py:176, 214, 637-648` →
`gui:=true headless:=false`). Ama trainer'lar sabit `gui=False` geçiyor:
`train_sac.py:393`, `train_ppo.py:291`. Bir `--gui` flag'i eklemek 2 satırlık iş; istersen
ekleyeyim. Bu yolla dünya değişince GUI de otomatik yeniden açılır.

Benchmark için: `run_benchmark.py:214` `gui=False` sabit geçiyor **ve** `BenchmarkEnv`
`self.gui`'yi hiç kullanmıyor; launch'a doğrudan `gui:=false headless:=true` yazıyor
(`benchmark_env.py:371-382`). Orada GUI için hem flag hem launch satırları düzeltilmeli.
Yol 1 (sonradan gzclient bağlamak) benchmark'ta da çalışır.

Klasik planner'lar için zaten hazır: `python3 run_barn_custom.py --world_idx 0 --gui`.

### Yol 3 — RViz ile robotun "gördüğünü" izlemek

Gazebo robotu dışarıdan gösterir; **politikanın gördüğünü** görmek için RViz daha faydalı:
```bash
rviz   # Fixed Frame: odom (veya base_link), Add → LaserScan → /scan
```
Ama dikkat: RViz ham `/scan`'i gösterir, **scan memory ile birleştirilmiş 90 sektörlük
obs'u değil**. Politikanın gerçek girdisini görmek için:
```bash
python3 train_sac.py --robot moborobo --debug-obs
```
Bu, her adımda obs dilimlerini, `min_range_m`, `fresh_frac` ve reward bileşenlerini basar
(`train_sac.py:192-279`). Hafızayı/fused scan'i RViz'de canlı görmek istersen, env'den
fused mesafeleri bir `LaserScan` topic'i olarak yayınlayan küçük bir debug publisher
eklenebilir — ayrıca konuşabiliriz.

### Eğitilmiş modeli izlemek

Eğitim değil de eğitilmiş politikayı izlemek istiyorsan: benchmark'ı `rl_sac` planner'ıyla
çalıştırıp (headless) Yol 1'deki gibi `gzclient` bağla:
```bash
python3 benchmark/run_benchmark.py --robot moborobo --planners rl_sac \
    --rl-model results/rl/runs/<run_id>/model_final.zip --world-list 0 3 6
# başka terminalde: gzclient
```
Benchmark fiziği durdurmadan gerçek zamanlı çalıştığı için burada hareket akıcı görünür.
