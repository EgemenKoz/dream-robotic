# RL Deep Dive — kanıtlı referans (Bölüm 1)

> Amaç: RL yapısının **her parçasını kod kanıtıyla** anlatmak; "episode length ne kadar, nasıl
> sayılıyor?", "reward nedir?", "obs shape ne?" gibi sorulara **dosya:satır** göstererek cevap
> verebilmek.
>
> **Kanıt notasyonu:** `dosya.py:123` → o satıra bak. "SB3 varsayılanı" yazan yerler repoda
> açıkça set edilmemiş, Stable-Baselines3 kütüphanesinin default değeridir.
>
> **Dikkat:** Bu doküman `rl_training` branch'inin **commit'lenmemiş** çalışma ağacına göre
> yazıldı (`barn_env.py`, `rl_config.py`, `train_sac.py`, `train_ppo.py`,
> `rl_sac_controller.py`, `rl_logger.py`, `benchmark_env.py` değişmiş, `scan_memory.py` yeni).
> Commit'ten sonra satır numaraları kayabilir.
>
> `docs/rl_structure.md` mevcut tasarım dokümanı; içinde **eskimiş** birkaç değer var —
> §15'te listelendi. Çelişki olduğunda **kod** esas alınmıştır.

---

## 0. Kopya kâğıdı (en çok sorulacak sayılar)

| Soru | Cevap (moborobo) | Cevap (jackal) | Kanıt |
|---|---|---|---|
| Episode max uzunluğu | **1000 step** | **500 step** | `configs/rl_config.py:89-98`, `rl/env/barn_env.py:209-213` |
| 1 step süresi | 0.1 s (`time.sleep`) | 0.1 s | `rl_config.py:80`, `barn_env.py:436` |
| Episode süre bütçesi | 1000 × 0.1 = **100 s** | 50 s | aynı |
| Kontrol frekansı | 10 Hz | 10 Hz | aynı |
| Başlangıç pozu | (-2.25, 3.0), yaw 1.57 rad (hedefe bakıyor) | aynı | `configs/config.py:147` |
| Hedef | (-2.25, 13.0) → düz mesafe **10 m** | aynı | `config.py:150`, `barn_env.py:250-255` |
| Hedef toleransı | **1.0 m** | aynı | `config.py:233`, `barn_env.py:509` |
| Max hız | lin 0.2 m/s, ang 1.0 rad/s | 0.5 m/s, 1.57 rad/s | `config.py:66-67, 100-101` |
| Action | `[linear_x ∈ [0, vmax], angular_z ∈ [-wmax, wmax]]` | aynı | `barn_env.py:281-285` |
| Obs `scan` shape | **(4, 90)** (3 frame + 1 fresh mask) | (4, 90) | `barn_env.py:272-279` |
| Obs `state` shape | **(5,)** | (5,) | `barn_env.py:274-279, 935-941` |
| Algoritma | SAC (SB3), `MultiInputPolicy` | aynı | `train_sac.py:467-480` |
| γ (gamma) | **0.99** | 0.99 | `train_sac.py:332-340` |
| lr / batch / buffer | 3e-4 / 256 / 300 000 | aynı | `train_sac.py:328-331` |
| learning_starts | 1000 | aynı | `train_sac.py:331` |
| Toplam step (default) | 200 000 (SAC), 500 000 (PPO) | aynı | `train_sac.py:296-299`, `train_ppo.py:186-189` |
| Dünya başına episode | 5 | 5 | `train_sac.py:300-303` |
| Varsayılan eğitim dünyaları | 0,3,6,…,147 (50 dünya) | 0,6,…,294 (50 dünya) | `config.py:135-136`, `train_sac.py:374-375` |
| Terminal ödüller | goal +100, collision −50, unsafe −40 | aynı | `rl_config.py:102-105, 168` |
| Safety sınırı | ileri ark (±120°) min mesafe < **0.42 m** | aynı | `rl_config.py:166-176`, `barn_env.py:502` |
| Pose kaynağı | Gazebo ground-truth (`/gazebo/get_model_state`) | aynı | `barn_env.py:440-443` |

---

## 1. Dosya haritası ve veri akışı

```
configs/config.py            Robot limitleri, dünya/dataset yolları, start/goal, GOAL_TOLERANCE
configs/rl_config.py         RLEnvConfig (env+reward), RLModelConfig (ağ), RLLogConfig, RLEvalConfig
rl/env/barn_env.py           BarnEnv(gym.Env): Gazebo yönetimi, obs, reward, termination
rl/env/scan_memory.py        ScanMemory: LiDAR kör halkasını (0.45 m) dolduran kısa süreli nokta hafızası
rl/models/feature_extractor.py  ScanStateExtractor: 1D-CNN (scan) + FC (state)
rl/curriculum.py             BARN metriklerinden kolay/orta/zor ayrımı, --curriculum fazları
rl/controllers/rl_sac_controller.py  Eğitilmiş modeli benchmark'ta çalıştıran controller
rl/logging/rl_logger.py      RLLogger (eğitim scriptlerinde KULLANILMIYOR, bkz. §11)
train_sac.py / train_ppo.py  Giriş noktaları: WorldCyclingEnv + SB3 + callback'ler
benchmark/benchmark_env.py   Değerlendirme ortamı (gym değil) — RL + klasik planner'lar
benchmark/run_benchmark.py   --planners rl_sac --rl-model PATH
gazebo/gazebo_simulation.py  pause/unpause/teleport/get_model_state/collision servis sarmalayıcısı
rl/tests/test_barn_env.py    Birim testleri (+ gazebo işaretli entegrasyon testleri)
```

Eğitim sırasında bir adımın veri akışı:

```
SB3 SAC.predict(obs) ──► action [lin, ang]
        │
        ▼
BarnEnv.step()  (barn_env.py:424)
  clip(action) → unpause → /cmd_vel publish → sleep 0.1 s → pause
  get_model_state()  ──► (x, y, yaw)  [ground truth]
  _build_obs()       ──► ScanMemory.update(/scan) → min-pool 90 sektör → /5 m → 3'lü stack + fresh mask
  get_hard_collision() ──► /collision topic'inden bayrak
  _forward_min_m()   ──► ileri ark (±120°) min mesafe [m]
  reward = step + progress + prox + heading + path  (+ terminal, + Φ düzeltmesi)
        │
        ▼
Monitor → DummyVecEnv → SAC replay buffer → her step 1 gradient update (SB3 varsayılanı)
```

---

## 2. Dünya, başlangıç ve hedef geometrisi

- **Başlangıç:** `STATIC_INIT_POSITION = [-2.25, 3, 1.57]` → x=-2.25, y=3.0, yaw=1.57 rad ≈ 90°
  (+y yönüne, yani hedefe bakıyor). `configs/config.py:147`
- **Hedef:** başlangıç + `STATIC_GOAL_OFFSET = [0, 10]` → **(-2.25, 13.0)**. `config.py:150`,
  `barn_env.py:250-254`
- **`_max_dist`** = start→goal düz mesafe = **10.0 m**. `dist_norm`'un paydası budur. `barn_env.py:255`
- **Tüm dünyalarda start/goal aynı**; sadece engel yerleşimi değişir. `barn_env.py:249` yorum.
- **Dünya sayısı:** jackal 300 (0–299), moborobo **153** (0–152). `config.py:129-132`
- **Dünya dosyaları:** moborobo için `~/Desktop/moborobo_dataset/world_files/world_N.world`,
  A* referans yol `path_files/path_N.npy`, zorluk metrikleri `metrics_files/metrics_N.npy`.
  `config.py:26-33, 161-188`
- **Hedefe varış:** robot merkezi hedefe **≤ 1.0 m** olunca başarı. `config.py:233`, `barn_env.py:509`.
  Yani başarı için en az **9 m** yol gerekir.

---

## 3. Episode yaşam döngüsü

### 3.1 `reset()` — ne olur, hangi sırayla? (`barn_env.py:348-422`)

1. Sayaçlar sıfırlanır: `_episode += 1`, `_step_count = 0`, `_episode_reward = 0`,
   `_last_action = [0,0]`, scan history temizlenir, `ScanMemory.reset()`. `:356-362`
2. **Dünya seçimi:** `_fixed_world_idx` varsa o; yoksa `np_random` ile rastgele. `:365-368`
   (Eğitimde `WorldCyclingEnv` her zaman sabit dünya verir, bkz. §9.)
3. **Gazebo yalnızca dünya değişince yeniden başlatılır** (`managed=True`). `:370-374`
   Yeniden başlatma: persistent `roscore` korunur, `gzserver` + (moborobo için)
   `barn_move_base.launch` (pointcloud_to_laserscan → `/scan`) başlatılır. `:603-702`
4. Dünya değiştiyse A* referans yolu yüklenir (`path_N.npy` → Gazebo koordinatları, başa start
   sonuna goal eklenir). `:379-380, 1009-1030`
5. Fizik unpause edilir (sensörler aksın diye). `:389`
6. **Teleport:** `/gazebo/set_model_state` ile start pozuna, 1 s bekle, konum 0.1 m
   toleransta değilse tekrar dene (max 10 deneme). `:392-400`, `gazebo_simulation.py:64-74`,
   `config.py:216-217`
7. Eski çarpışma bayrağı boşaltılır, robot durdurulur. `:403-404`
8. Taze bir LaserScan beklenir (max 30 s). `:407-412, 72`
9. İlk obs üretilir; `_prev_dist` gerçek başlangıç mesafesine set edilir (~10 m). `:414-417`
10. Dönen info: `{"world_idx", "episode_idx"}`. `:422`

> Not: reset bittiğinde fizik **unpause** durumda kalır; ilk `step()` zaten unpause→…→pause
> yapar. Arada robot 0 hız komutuyla durur.

### 3.2 `step()` — sıralı akış (`barn_env.py:424-576`)

| # | Ne yapılır | Satır |
|---|---|---|
| 1 | Action, action_space sınırlarına **clip** edilir; `_last_action` güncellenir | `:427-431` |
| 2 | `unpause` → `/cmd_vel` publish → **`time.sleep(0.1)`** → `pause` | `:434-437` |
| 3 | `get_model_state()` → pose (ground truth), yaw quaternion'dan | `:440-443` |
| 4 | `_build_obs()` (scan memory güncellemesi dahil) | `:444` |
| 5 | Hedefe mesafe `curr_dist` | `:445` |
| 6 | Çarpışma bayrağı (`/collision` topic sayacı, son çağrıdan beri) | `:446`, `gazebo_simulation.py:40-48` |
| 7 | İleri ark min mesafe `min_range_m` (metre) | `:450, 943-947` |
| 8 | Dense reward bileşenleri (bkz. §6) | `:453-494` |
| 9 | Termination kontrolü (öncelik sırası aşağıda) | `:497-519` |
| 10 | `terminated` ise terminal potansiyel düzeltmesi | `:541-543` |
| 11 | `info` sözlüğü (reward bileşenleri dahil) | `:547-570` |
| 12 | Bitti ise robot durdurulur + `episodes.csv`'ye satır yazılır | `:572-574` |

### 3.3 Episode nasıl biter? — öncelik sırası (`barn_env.py:497-519`)

`if / elif` zinciri olduğu için **ilk tutan kazanır**:

| Sıra | Outcome | Koşul | Ek reward | Gym bayrağı |
|---|---|---|---|---|
| 1 | `collision` | `/collision` topic'inde temas | −50 | `terminated=True` |
| 2 | `unsafe` | `SAFETY_TERMINATE` ve ileri ark min mesafe **< 0.42 m** | −40 | `terminated=True` |
| 3 | `success` | `curr_dist ≤ 1.0 m` | +100 | `terminated=True` |
| 4 | `timeout` | `_step_count ≥ max_episode_steps` | yok | `truncated=True` |
| – | `running` | hiçbiri | – | ikisi de False |

Sonuç: Aynı adımda hem çarpışma hem hedef olursa **collision** sayılır.

### 3.4 ⭐ Episode length — tam cevap

**Değer:**
- `RLEnvConfig.MAX_EPISODE_STEPS_MOBOROBO = 1000`, `MAX_EPISODE_STEPS_JACKAL = 500`
  (`rl_config.py:89-90`).
- `RLEnvConfig.max_episode_steps(robot)` robota göre seçer (`rl_config.py:94-98`).
- `BarnEnv.__init__`'e `max_episode_steps=None` gelirse bu fonksiyon kullanılır
  (`barn_env.py:209-213`). `train_sac.py:388-395` bu parametreyi **geçmiyor** → robota göre
  default geçerli. **CLI'dan değiştirilemez**; değiştirmek için config'i veya kodu değiştirmek gerekir.
- `MAX_EPISODE_STEPS = 500` alias'ı (`rl_config.py:92`) sadece geri uyumluluk içindir, BarnEnv
  onu kullanmaz.

**Neden 1000?** (`rl_config.py:82-87` yorumu) 1000 × 0.1 s = 100 s = standart BARN süre bütçesi;
0.2 m/s max hızla 20 m gidilebilir; gerçekçi ~0.13 m/s ortalamayla 10 m'lik parkur ~75 s sürer.

**Nasıl sayılır?** (`barn_env.py:513-519`)
```python
else:                              # collision/unsafe/success DEĞİLSE
    self._step_count += 1
    if self._step_count >= self.max_episode_steps:
        truncated = True; outcome = "timeout"
```
- Sayaç **sadece terminal olmayan adımlarda** artar.
- Timeout episode'u: tam olarak **1000 adet `env.step()`** çağrısı (1000. çağrıda `truncated`).
- Terminated episode'lar (collision/unsafe/success) için son adım sayılmaz →
  **`episodes.csv`'deki `steps` = gerçek step sayısı − 1**. SB3 `Monitor`'ün `l` değeri
  (TensorBoard `rollout/ep_len_mean`, konsoldaki `mean_len(100)`) ise son adımı da sayar.
  İki kaynak arasında **1 adımlık fark** normaldir.

**Süre anlamı:**
- Her adımda fizik yalnızca `time.sleep(0.1)` boyunca açık (`:434-437`). Servis çağrıları,
  obs hesabı, SAC gradient update'i fizik **pause** iken olur → bu süreler sim zamanına eklenmez.
- **Sim zamanı/adım ≈ 0.1 s × Gazebo real-time factor** (RTF ≈ 1 varsayımı, `rl_config.py:77`).
  RTF < 1 ise robot adım başına daha az yol alır.
- **Duvar saati/adım ≥ 0.1 s** + servis + gradient overhead. 200k step ≥ 5.6 saat
  (+ her 5 episode'da bir dünya değişince ~10+ s Gazebo restart).

**Fiziksel olarak mümkün minimum başarı uzunluğu:**
- moborobo: 9 m / 0.2 m/s = 45 s ≈ **450 step** (düz çizgi, engelsiz). A* yolları ~11 m
  civarında olduğu için (yorum: `rl_config.py:193-194`) gerçekte ~500+ step.
- jackal: 9 m / 0.5 m/s = 18 s ≈ 180 step.
- Yani 1000 step'lik bütçe moborobo için kabaca **2× pay** bırakıyor.

**Truncation ≠ terminal:** Timeout `truncated=True` döner; SB3 (replay buffer
`handle_timeout_termination=True` — SB3 varsayılanı) bu geçişte `V(s')`'den bootstrap yapar.
Terminal potansiyel düzeltmesi de truncation'da **uygulanmaz** (`barn_env.py:538-541`).

**Benchmark'ta episode uzunluğu farklıdır** (bkz. §12): adım sayısı değil, **120 s sim zamanı**
timeout (`run_benchmark.py:153-154`, `benchmark_env.py:228`). `RLEvalConfig.MAX_EVAL_STEPS=600`
(`rl_config.py:306`) okuduğum hiçbir eğitim/benchmark dosyasında kullanılmıyor.

---

## 4. Observation

`observation_space = Dict({"scan": Box(0,1,(4,90)), "state": Box(low,high,(5,))})`
— `barn_env.py:272-279`. Tek doğruluk kaynağı: `_build_obs()` `barn_env.py:898-941`.

### 4.1 `obs["scan"]` — (SCAN_STACK + 1, NUM_SCAN_RAYS) = **(4, 90)**

Boru hattı (pipeline):

1. **ROS callback** `_scan_cb` (`barn_env.py:794-815`): `inf`, `< range_min`, `> range_max`
   olan ışınlar **NaN** yapılır. NaN = "dönüş yok" (ya çok uzak ya da 0.45 m'den yakın).
2. **Scan memory füzyonu** `_fuse_scan` → `ScanMemory.update()` (`barn_env.py:875-896`,
   `scan_memory.py:146-242`): 90 sektörde metre cinsinden `fused_m` ve `fresh` maskesi.
   Detay §5.
3. **Normalize:** `clip(fused_m / 5.0, 0, 1)` (`barn_env.py:909`). Payda
   `min(SCAN_CLIP_RANGE=5.0, sensör max=10.0) = 5.0 m` (`barn_env.py:247`, `rl_config.py:44`).
   Yani 5 m ve ötesi = 1.0; 0.45 m ≈ 0.09.
4. **Frame stack:** deque(maxlen=3). Episode'un ilk frame'inde **3 kopya** ile doldurulur,
   sonra her adım bir frame eklenir. Satır 0 = en eski, satır 2 = en yeni. (`barn_env.py:913-918`)
5. **Satır 3 = fresh mask:** o sektör **bu adımda doğrudan ölçüldüyse 1.0**, hafızadan
   geldiyse / bilinmiyorsa 0.0. (`barn_env.py:919-921`) Sadece `SCAN_MEMORY=True` iken var;
   kapalıysa shape (3, 90).

Sektör geometrisi (memory açıkken): 90 sektör robot çerçevesinde **[-π, π)** aralığını kaplar,
her sektör **4°**. Sektör 45 = tam ön (bearing 0), sektör 0 = arka (-π). (`scan_memory.py:103-104,
259-274`, test: `test_barn_env.py:395`)

Neden min-pool: ince bir engele tek ışın çarpsa bile sektör minimumu onu korur
(`barn_env.py:1052-1067`, test `test_barn_env.py:274-283`).

### 4.2 `obs["state"]` — (5,)  (`barn_env.py:923-941`)

| İndeks | Özellik | Formül | Aralık |
|---|---|---|---|
| 0 | `dist_norm` | `clip(dist / 10.0, 0, 1)` | [0, 1] |
| 1 | `sin(angle_to_goal)` | robot çerçevesinde hedef açısı: `wrap(atan2(dy,dx) − yaw)` | [-1, 1] |
| 2 | `cos(angle_to_goal)` | aynı açı | [-1, 1] |
| 3 | `last_lin_norm` | `clip(last_lin / vmax, 0, 1)` | [0, 1] |
| 4 | `last_ang_norm` | `clip(last_ang / wmax, -1, 1)` | [-1, 1] |

- Reset anında: `dist_norm ≈ 1.0`, `sin ≈ 0`, `cos ≈ +1` (robot hedefe bakıyor), last action = 0.
- A* yolu obs'ta **yok** — sadece reward'da kullanılıyor (`rl_config.py:182-186`).
- Pose **ground truth** (Gazebo). Gürültü/odometri yok.

---

## 5. ScanMemory — neden ve nasıl (`rl/env/scan_memory.py`)

**Problem** (`scan_memory.py:1-27`, `rl_config.py:46-57`):
- RS-16 LiDAR 0.40 m'den yakını göremez; `pointcloud_to_laserscan` `range_min=0.45 m`
  (`config.py:94`). Bu ışınlar `inf` gelir.
- Eski kod `inf → 10 m` yapıyordu → **çarpmak üzere olan engel "10 m boş" görünüyordu**.
- Moborobo ön tamponu base_link'ten 0.387 m, dönme yarıçapı 0.4528 m, kör halka 0.45 m →
  son görüş ile temas arasında ~6 cm (~3 adım @ 0.2 m/s).

**Algoritma** (`update()`, `scan_memory.py:146-242`), her adım:
1. Ölçüm: geçerli ışınlar sektörlere min-pool edilir (`meas`), ayrıca **dünya koordinatında
   noktalara** çevrilir. `:179-194`
2. **Raycast temizleme:** Mevcut hafıza noktaları şu anki robot çerçevesine projekte edilir.
   Bir noktanın olduğu sektörde şu an ölçülen mesafe noktadan ≥ 0.15 m **uzaksa** (içinden
   görülüyorsa) nokta silinir — **ancak nokta 0.45 + 0.05 = 0.50 m'den uzaksa**. Kör halkadaki
   noktalar ölçümle asla silinmez. `:198-214`
3. **Budama:** robottan > 4.0 m uzak noktalar ve 150 adımdır (15 s) yeniden görülmeyenler
   silinir. `:216-220`, `rl_config.py:66, 74`
4. **Ekleme:** bu frame'in noktaları eklenir, 5 cm voxel ile tekilleştirilir (en yeni görüş
   kalır), max 8000 nokta. `:222-229, 286-306`
5. **Füzyon:** tüm hafıza sektörlere projekte edilir; sektör başına
   `fused = min(ölçüm, hafıza)`; hiçbir bilgi yoksa `range_max` (10 m). `:231-242`

`arc_min(fused, 2.0944)` — sektör merkezi ±120° içindeki minimum (`scan_memory.py:244-257`).
Hem prox penalty hem safety sınırı bunu kullanır (`barn_env.py:943-947`). Arka 120° hariç
tutulur çünkü geçilmiş engeller kör halkada kalıp safety'yi sonsuza kadar tetiklerdi.

**Kritik çıkarım:** `SAFETY_DIST = 0.42 m` sensör tabanının (0.45) **altında**; bu mesafe
**sadece hafıza sayesinde** görülebilir (`rl_config.py:161-162`). `SCAN_MEMORY=False` yapılırsa
fused değerler asla 0.45'in altına inmez (NaN → 10 m) → **unsafe hiçbir zaman tetiklenmez** ve
arc yerine tüm 360° min kullanılır (`barn_env.py:947`).

Performans: ~0.3 ms/step iddiası (doküman); test <10 ms/step garanti ediyor
(`test_barn_env.py:491-503`).

---

## 6. Action

- `Box(low=[0, −wmax], high=[vmax, +wmax])`, float32. `barn_env.py:281-285`
  - moborobo: `[0, 0.2] × [−1.0, 1.0]`; jackal: `[0, 0.5] × [−1.57, 1.57]`.
- **Geri gitme yok** (lin ≥ 0).
- `step()` action'ı yine de clip'liyor (`:427`).
- SAC tarafında (SB3 davranışı): actor çıktısı `tanh` ile [-1,1]'e sıkıştırılır ve action space
  sınırlarına lineer ölçeklenir. Yani ağ çıktısı 0 → lin = 0.1 m/s, ang = 0.
- Komut `/cmd_vel` (Twist) olarak doğrudan yayınlanır; **move_base kullanılmıyor**
  (`barn_env.py:4, 653-658` yorum: move_base açılıyor ama hiç goal almıyor, sadece
  pointcloud_to_laserscan için).

---

## 7. Reward — tam formül

Kod: `barn_env.py:452-545`. Değerler moborobo için.

### 7.1 Her adım (dense)

```
r = r_step + r_progress + r_prox + r_heading + r_path
```

| Bileşen | Formül | Değer aralığı (moborobo) | Kod | Config |
|---|---|---|---|---|
| step | `−0.01` | sabit −0.01 | `:453` | `rl_config.py:108` |
| progress | `12.5 × (prev_dist − curr_dist)` | tam hızda hedefe doğru ≈ **+0.25**; uzaklaşırken negatif | `:462` | `rl_config.py:119-128` |
| prox | `−0.15 × t² × (0.5 + 0.5·lin/vmax)`, `t = clip((0.90 − d)/(0.90 − 0.50), 0, 1)` | d ≥ 0.9 m → 0; d ≤ 0.5 m & tam hız → **−0.15**; d ≤ 0.5 m & duruk → −0.075 | `:470-476` | `rl_config.py:145-147` |
| heading | `−0.02 × |angle_to_goal| / π` | 0 … **−0.02** | `:479-482` | `rl_config.py:180` |
| path | `−0.15 × min(d_perp, 1.0) / 1.0` | 0 … **−0.15** | `:486-492` | `rl_config.py:187, 209-210` |

- `d` = ileri ark (±120°) füzyonlu min mesafe, **metre** (`:450`).
- `lin` = bu adımda komut edilen lineer hız (prox hızla ağırlıklanıyor).
- `d_perp` = robotun A* polyline'ına dik mesafesi (`:1102-1123`). `path_N.npy` yoksa path
  terimi 0 (`:1016-1018`).
- Jackal için progress ölçeği 5.0 (0.5 m/s × 0.1 s × 5 = +0.25) — iki robotta max progress
  eşitlenmiş (`rl_config.py:110-120`).
- `REWARD_SHAPING=False` → progress/prox/heading/path hepsi kapanır, sadece step + terminal
  kalır (`:460`, `rl_config.py:212-214`).

### 7.2 Terminal

| Outcome | Base | Φ düzeltmesi (terminated'da) | Kod |
|---|---|---|---|
| success | +100 | `+12.5 × curr_dist` (≤ +12.5, çünkü dist ≤ 1 m) | `:509-512, 541-543` |
| collision | −50 | `+12.5 × curr_dist` | `:498-501` |
| unsafe | −40 | `+12.5 × curr_dist` | `:502-508` |
| timeout | 0 | **uygulanmaz** | `:513-517` |

### 7.3 Terminal potansiyel düzeltmesi (`TERMINAL_POTENTIAL_ZERO`, artık kapalı)

> **Güncelleme (2026-10-13):** `TERMINAL_POTENTIAL_ZERO` artık **False**, γ **0.99**, `REVERSE_SPEED_RATIO` **0.3**, buffer **300 000**, ve hareketsizlik `stuck` terminali (−25) ile bitiyor. Sebep: γ=0.999 + terminal potansiyel ile `reverse_v1` koşusunda kritik Q ≈ +716'ya kaydı (fiziksel maksimum 225), politika sıfır ortalamalı titremeye döndü. Aşağıdaki eski gerekçeler tarihçe olarak duruyor.

> **Güncelleme (2026-10-14):** `REWARD_STUCK` **−50** (çarpışmayla eşit; −25'te politika engel önünde ileri-geri titreyip stuck ile bitirmeyi öğrendi). Referans path artık doğru yerde: `WorldConfig.GRID_ORIGIN = (-4.425, 5.175)` (BARN'ın dönüşümü 0.15 m / 0.10 m kaymıştı, grid hücreleri silindirlere oturtularak ölçüldü). `PATH_SMOOTHING` ile path, orijinalinden daha dar bir yerden geçmeyecek şekilde ip çekme ile kısaltılıyor (ortalama 34 → 5 nokta, 11.15 → 10.43 m).

> **Güncelleme (2026-10-15):** `multi_v4` (300k) hiç başarı almadı: kritik ıraksadı (Q 20k'da +40, 300k'da +7000; buffer'daki ödüller hiç +0.35'i geçmemişti), ent_coef 0.03 → 8.7. Çözüm: kritiklerde LayerNorm (`RLModelConfig.CRITIC_LAYER_NORM`, `rl/models/critic.py`), log satırında `q_mean` ve > 200 uyarısı. Ayrıca dünyalar artık karıştırılmış sırada dolaşılıyor (`WorldCycler`); `multi_v3` sıralı dolaşım yüzünden 214k adımda 51 kolay dünyanın sadece 16'sını görmüştü. `--seed` eklendi.

- Progress, potansiyel tabanlı shaping: `Φ(s) = −12.5 × d(s)`; adım terimi `Φ(s') − Φ(s)`.
  Episode boyunca toplam teleskoplanır: `Φ(s_T) − Φ(s_0)`.
- Ng et al. (1999) politika değişmezliği için terminalde `Φ(s_T) = 0` olmalı → terminal adımda
  `+12.5 × d_T` **eklenir**. (`barn_env.py:521-543`, `rl_config.py:216-242`)
- Sonuç: **terminated her episode'da progress toplamı = 12.5 × d₀ ≈ +125 (sabit)**. Shaping
  sonuçları sıralayamaz; sıralamayı terminal ödüller belirler.
- Tarihçe (yorumlardan): işaret önce ters (çıkarma) yazılmış; bu, erken başarısızlığı çok
  pahalı yapıp robotun yolda donmasına yol açmış (timeout %2.5 → %17, başarı %29).
  `rl_config.py:225-232`

### 7.4 ⚠ Episode return'lerini okurken dikkat

Güncel (doğru işaretli) kodla kaba episode toplamları (dense cezalar ihmal edilmeden önce):

| Senaryo (d₀=10 m) | progress | Φ düzeltme | terminal | ≈ toplam (dense cezalar hariç) |
|---|---|---|---|---|
| Başarı (d_T≈1) | +112.5 | +12.5 | +100 | **≈ +225** |
| 4 m gidip çarpma (d_T=6) | +50 | +75 | −50 | **≈ +75** |
| 4 m gidip unsafe (d_T=6) | +50 | +75 | −40 | **≈ +85** |
| 4 m gidip timeout (d_T=6) | +50 | yok | 0 | **≈ +50**, üstüne 1000 adım × (−0.01 + heading + path + prox) → belirgin şekilde daha düşük |

Buradan iki önemli sonuç:
1. `docs/rl_structure.md:242-253`'teki "crash 4 m in ≈ −85, unsafe ≈ −65" rakamları **eski
   (ters işaretli) sürüme ait**; güncel kodla geçerli değil.
2. Loglardaki `total_reward` / `ep_rew_mean`'e bakınca **timeout episode'ları çarpışmalardan
   daha düşük return gösterebilir**. Bu bug değil: timeout truncation olduğu için gerçek değer
   `V(s_final)` bootstrap'ı ile hesaplanır; episode toplamı tam resmi göstermez. Başarıyı
   yorumlarken **outcome oranlarına** (success/collision/unsafe/timeout) bakın, return'e değil.

### 7.5 `info` ile gelen debug alanları (`barn_env.py:547-570`)

`world_idx, episode_idx, step, dist_to_goal, collided, outcome, episode_reward, min_range_m,
fresh_frac` ve `reward_components = {step, progress, prox, heading, path, path_dev_m, terminal_phi}`.
`python3 train_sac.py --robot moborobo --debug-obs` bunları adım adım basar (`train_sac.py:192-279`).

---

## 8. Ağ mimarisi (`rl/models/feature_extractor.py`, `configs/rl_config.py:247-268`)

```
scan (4, 90) ─ Conv1d(4→32, k5, s2, p2) ─ReLU─ (32, 45)
             ─ Conv1d(32→32, k3, s2, p1) ─ReLU─ (32, 23) ─ Flatten 736 ─ Linear(736→128) ─ReLU─┐
                                                                                              ├─ concat 160
state (5,)  ─ Linear(5→32) ─ReLU──────────────────────────────────────────────────────────────┘
                 │
                 ├─ Actor : 160 → 256 → 256 → (μ[2], log σ[2]) → tanh-Gaussian
                 └─ Critic: Q1, Q2 : (160 + 2 action) → 256 → 256 → 1   (+ target kopyaları)
```

- Uzunluklar hesaplandı: `(90+2·2−5)/2+1 = 45`, `(45+2·1−3)/2+1 = 23`; `32×23 = 736`.
  Kodda `flat_dim` dummy forward ile bulunuyor (`feature_extractor.py:61-62`).
- Frame'ler (ve fresh mask) Conv1d'de **kanal** olarak giriyor (`:46-57`).
- `features_dim = 128 + 32 = 160` (`:44`); test: `test_barn_env.py:677-688`.
- `net_arch = [256, 256]` (`rl_config.py:268`, `train_sac.py:426-429`).
- Parametre sayısı (elle hesap): extractor ≈ 98.3k; actor ≈ 206k; critic tarafı (2 Q + kendi
  extractor'ı) ≈ 314k. SB3 SAC'ta actor ve critic **ayrı** feature extractor kullanır
  (`share_features_extractor=False`, SB3 varsayılanı).

---

## 9. Eğitim (`train_sac.py`)

### 9.1 Hiperparametreler

| Parametre | Değer | Kaynak |
|---|---|---|
| Policy | `MultiInputPolicy` + `ScanStateExtractor` | `train_sac.py:426-429, 467-480` |
| learning_rate | 3e-4 | `:328` |
| batch_size | 256 | `:329` |
| buffer_size | 300 000 | `--buffer-size` |
| learning_starts | 1 000 | `:331` |
| gamma | **0.99** (ilerleme ödülü gerçek olduğu için hedefin ufukta olması gerekmiyor) | `--gamma` |
| tau | 0.005 | `:341` |
| ent_coef | `auto` (target entropy = −dim(A) = −2, SB3 varsayılanı) | `:342-343` |
| train_freq / gradient_steps | 1 / 1 (SB3 varsayılanı → her env adımında 1 update) | — |
| total_timesteps | 200 000 | `:296-299` |
| n_envs | 1 (tek Gazebo; SB3 `DummyVecEnv`'e sarar) | `:388-395, 421` |

**Neden artık γ = 0.99?** γ=0.999'da hedef `r + 0.999·Q` neredeyse hiç daralmıyor; buffer'da terminal kalmayınca Q seviyesi serbest kaldı ve +716'ya kaydı. İlerleme ödülü gerçek olunca (terminal potansiyel kapalı) 0.99 yeterli — Temmuz/Ağustos koşuları bu ayarla %47–63 başarı aldı.

*Eski gerekçe (tarihçe):* 10 Hz'de γ=0.99 → ufuk 100 adım = 10 s = 0.2 m/s'de **2 m**. ~500 adım
ötedeki hedef ödülü ~+0.7'ye iner; politika sadece dense shaping'e bakıp donar. 0.999 → 1000
adım. (`train_sac.py:332-340`, `rl_config.py:238-241`, test `test_barn_env.py:585-603`)

Replay buffer bellek tahmini: obs = 4×90 + 5 = 365 float32 = 1460 B; obs + next_obs ≈ 2.9 KB/geçiş
→ 300k geçiş ≈ **~0.9 GB**.

### 9.2 Dünya seçimi — `WorldCyclingEnv` (`train_sac.py:62-96`)

- Bir dünya listesi üzerinde, **her dünyada 5 episode**, sonra sıradakine (liste sonunda başa
  döner). `:79-86`
- Gazebo yalnızca dünya değişince yeniden başlar (her 5 episode'da bir, ~10+ s).
- Liste önceliği (`:360-378`): `--curriculum` > `--worlds` > `WorldConfig.eval_worlds()`.
- **Default liste = benchmark'ın eval listesi** (moborobo: 0,3,…,147 — `config.py:136`).
  Yani varsayılan kullanımda **eğitim ve test dünyaları aynı**; genelleme ölçülmüş olmaz.
- Robot, `SimConfig.ROBOT` üzerinden dataset'i seçer; bu yüzden `--robot` config import'undan
  önce parse ediliyor (`train_sac.py:43-48`).

### 9.3 Curriculum (`--curriculum`, `rl/curriculum.py`)

- Her dünya için `metrics_N.npy`'deki 5 BARN metriği (en yakın engel mesafesi, ortalama
  görünürlük, dispersion, karakteristik boyut, tortuosity) dataset genelinde min-max normalize,
  "kolay" yönlüler ters çevrilip ortalanır → [0,1] zorluk. `curriculum.py:38-97`
- Sıralanıp 3 eşit kademe: easy/medium/hard. `:100-129`
- Fazlar kümülatif: `[0, T/3)` easy, `[T/3, 2T/3)` easy+medium, `[2T/3, T]` hepsi. `:132-153`
- `CurriculumCallback` faz değişince `set_world_list()` çağırır; **aktif dünya bir sonraki
  cycle'a kadar değişmez**, pointer yeni listenin uzunluğuna mod alınır (sıfırlanmaz,
  yorumdaki "pointer is reset" ifadesine rağmen). `train_sac.py:88-96, 134-187`
- Curriculum açıkken havuz, metrics dosyası olan **tüm** dünyalardır (moborobo: 153'e kadar) —
  eval dünyaları da dahil.

### 9.4 Checkpoint / resume

- `CheckpointCallback(save_freq=20000, name_prefix="model", save_replay_buffer=True)` →
  `model_20000_steps.zip`, `model_replay_buffer_20000_steps.pkl`, … (`train_sac.py:491-497`)
- Çıkışta (normal veya hata) `model_final.zip` (`:512-516`).
- `--resume PATH`: modeli yükler, yanında replay buffer varsa onu da yükler; `num_timesteps`
  sıfırlanmaz (`reset_num_timesteps=False`). `:434-465, 509`
  Dikkat: resume'da `learning_starts` ve `policy_kwargs` tekrar verilmiyor (checkpoint'teki
  kullanılır).

### 9.5 Yardımcı modlar

- `--smoke-test`: Gazebo'yu açar, reset + 5 rastgele adım, çıkar. `:398-412`
- `--debug-obs`: sabit action script'i (ileri, ileri, sol, sağ, dur) ile obs ve reward
  bileşenlerini basar. `:415-418, 192-279`

---

## 10. PPO (`train_ppo.py`)

Aynı env, aynı extractor, aynı world cycling/curriculum. Farklar:

| Parametre | Değer | Kaynak |
|---|---|---|
| total_steps | 500 000 | `train_ppo.py:186-189` |
| n_steps (rollout) | 2048 | `:216-220` |
| batch_size | 64 | `:221-225` |
| n_epochs | 10 | `:226-227` |
| gamma | 0.99 | `--gamma` |
| gae_lambda | 0.95 | `:235-237` |
| clip_range | 0.2 | `:238-239` |
| vf_coef / ent_coef | 0.5 / 0.01 | `:240-244` |
| run_id prefix | `ppo_` | `:279` |

`--debug-obs` PPO'da yok. Replay buffer olmadığı için checkpoint sadece policy zip.

---

## 11. Loglama ve çıktılar

Run dizini: `results/rl/runs/{run_id}/` (`rl_config.py:279-280`, `train_sac.py:381-384`),
`run_id` varsayılan `YYYYmmdd_HHMMSS`.

| Dosya | Kim yazar | İçerik |
|---|---|---|
| `episodes.csv` | `BarnEnv._log_episode` (`barn_env.py:1032-1047`) | `episode,world_idx,outcome,steps,total_reward,final_dist_to_goal` |
| `tb/` | SB3 | `rollout/ep_rew_mean`, `rollout/ep_len_mean`, `train/actor_loss`, `train/critic_loss`, `train/ent_coef` … |
| `model_*_steps.zip`, `model_replay_buffer_*_steps.pkl`, `model_final.zip` | SB3 callback | §9.4 |
| `gazebo_worldN.log`, `navstack_worldN.log`, `roscore.log` | `BarnEnv._restart_gazebo` (`barn_env.py:620-625, 709-711`) | ROS/Gazebo çıktıları |
| Konsol satırı | `TrainProgressCallback` (`train_sac.py:101-129`) | her 2000 adımda son 100 episode ort. reward/len |

- `episodes.csv`'de `steps` terminated episode'larda 1 eksik (bkz. §3.4).
- `RLLogger` (`rl/logging/rl_logger.py`) `run_config.json`/`training.log` üretir ve
  `UN %` (unsafe oranı) hesaplar, ama **`train_sac.py` / `train_ppo.py` onu import etmiyor**.
  Dolayısıyla eğitim run'larında `run_config.json` ve `training.log` oluşmaz; hiperparametreler
  sadece SB3 zip'inde ve konsol çıktısında kalır. Ayrıca `_outcome_counts` başlangıçta `unsafe`
  anahtarını içermiyor ama `.get()` ile okunduğu için sorun çıkmıyor (`rl_logger.py:72-74, 151`).

---

## 12. Deployment / Benchmark

### 12.1 `SACController` (`rl/controllers/rl_sac_controller.py`)

- Çağrı: `python3 -m benchmark.run_benchmark --robot moborobo --planners rl_sac --rl-model PATH`
  (`run_benchmark.py:90-94, 160-164`). `PATH` zip, zip'siz yol veya run dizini olabilir
  (dizinde `model_final.zip`, yoksa alfabetik son `model_*.zip`). `:85-108`
- Algoritma tespiti: zip içindeki `data`'da `n_steps` var ve `tau` yoksa PPO, değilse SAC. `:63-82`
- Obs formatı modelden okunur: Dict ise shape'ten stack ve sektör sayısı; satır sayısı
  `SCAN_STACK + 1` ise memory açık kabul edilir. `:161-196`
- `wants_raw_scan=True` (memory açıksa) → benchmark NaN'lı ham scan'i verir. `:217-219`,
  `benchmark_env.py:254-260`
- `reset()` last action, frame history ve hafızayı sıfırlar. `:221-225`
- Varsayılan **deterministic** (policy mean). `:138, 244`
- Benchmark her episode için controller'ı **yeniden oluşturuyor** → model her episode yeniden
  yükleniyor. `run_benchmark.py:241-245`

### 12.2 Eğitim vs benchmark farkları (önemli!)

| Konu | Eğitim (`BarnEnv`) | Benchmark (`BenchmarkEnv`) | Kanıt |
|---|---|---|---|
| Episode limiti | 1000 **adım** (≈100 s) | **120 s sim zamanı** (`rospy.get_time`) | `barn_env.py:515`, `run_benchmark.py:153`, `benchmark_env.py:228` |
| Fizik | Her adım pause/unpause (inference sırasında dünya durur) | **Sürekli akar**; 0.1 s duvar saati periyodu | `barn_env.py:434-437`, `benchmark_env.py:281-284` |
| Unsafe (0.42 m) | Episode'u bitirir | **Yok** | `barn_env.py:502`, `benchmark_env.py:218-230` |
| Collision | Bitirir | Bitirir (dosya docstring'i "bitirmez" diyor ama kod `break` ediyor) | `benchmark_env.py:6, 149, 219-222` |
| Başarı | dist ≤ 1.0 m | dist ≤ 1.0 m | aynı `NavConfig.GOAL_TOLERANCE` |
| Pose | Gazebo ground truth | Gazebo ground truth | `benchmark_env.py:525-529` |
| Obs | `_build_obs` | Controller'da birebir kopya | `rl_sac_controller.py:286-324` |

Sonuç: Eğitimde "unsafe" ile biten durumlarda benchmark'ta robot devam eder (ve muhtemelen
çarpar ya da kurtarır). Ayrıca benchmark'ta inference gecikmesi sırasında robot hareket etmeye
devam eder; eğitimde etmez.

### 12.3 Benchmark metrikleri (`benchmark_env.py:291-326`)

`success, collision_count, path_length, opt_path_length (A* uzunluğu), time_elapsed, SPL =
success × opt/max(actual, opt), nav_metric = success × OT / clip(t, 2·OT, 8·OT)` (OT = opt_len /
2.0 m/s), `mean/max_path_deviation, min_clearance, final_dist_to_goal`.

---

## 13. Testler (`rl/tests/test_barn_env.py`)

Çalıştırma: `pytest rl/tests/ -m "not gazebo" -v` (Gazebo'suz).

Kapsananlar: obs/action space şekil ve sınırları, config doğrulaması, goal geometrisi, scan
normalize ve min-pool, ScanMemory davranışları (kör halkaya takip, raycast temizleme, kör
halkada temizlememe, arka engelleri yok sayma, yaş/yarıçap budama, voxel dedup, hız),
safety/reward config ilişkileri, γ ufku, feature extractor boyutları, RLLogger.

Sınırlar (bilinmesi gereken):
- `test_total_shaping_is_independent_of_where_the_episode_ends` (`:544-564`) formülü **testin
  içinde yeniden yazıp** kontrol ediyor; `BarnEnv.step()`'i çağırmıyor. Yani env'deki işareti
  birisi bozarsa bu test **yakalamaz**.
- `step()`/`reset()` reward ve termination mantığı yalnızca `@pytest.mark.gazebo` testlerinde
  (`:794-879`) koşuyor; bunlar dışarıda çalışan Gazebo ister.
- `test_world_idx_too_high_raises` / `test_boundary_world_299_ok` (`:139-153`)
  `SimConfig.ROBOT="jackal"` (varsayılan, `config.py:53`) varsayar; moborobo'da sınır 152'dir.

---

## 14. Kısa soru–cevap (sınav modu)

**S: Episode length ne kadar?** — moborobo 1000 adım, jackal 500 adım; 0.1 s/adım → 100 s /
50 s. `rl_config.py:89-98`, `barn_env.py:209-213`.

**S: Nasıl sayılıyor?** — `_step_count` sadece terminal olmayan adımlarda artar;
`≥ max_episode_steps` olunca `truncated=True`, outcome `timeout`. `barn_env.py:513-519`.

**S: Episode başka nasıl biter?** — Sırayla: collision (−50), unsafe (ileri ark < 0.42 m, −40),
success (≤1 m, +100). Hepsi `terminated`. `barn_env.py:497-512`.

**S: Timeout neden terminated değil truncated?** — Zaman sınırı MDP'nin gerçek sonu değil;
critic `V(s')` ile bootstrap etmeli. Bu yüzden Φ düzeltmesi de uygulanmaz. `barn_env.py:538-541`.

**S: Obs nedir?** — Dict: `scan (4,90)` = 3 füzyonlu/normalize LiDAR frame'i + fresh mask;
`state (5,)` = [dist/10, sin, cos, last_lin/vmax, last_ang/wmax]. `barn_env.py:898-941`.

**S: Neden 90 ışın?** — Ham ~722 ışın 4°'lik 90 sektöre min-pool edilir. `rl_config.py:27-31`.

**S: Neden 5 m ile normalize?** — BARN boşlukları <1 m; 10 m ile normalize edilse önemli
geometri [0, 0.1]'e sıkışır. `rl_config.py:38-44`.

**S: Scan memory ne işe yarar?** — 0.45 m kör halkadaki engelleri hatırlayıp "boş" görünmesini
engeller. §5.

**S: Reward nedir?** — §7.1 tablosu + terminal ödüller + Φ düzeltmesi.

**S: Progress ölçeği neden 12.5?** — 0.2 m/s × 0.1 s = 0.02 m/adım; ×12.5 = +0.25/adım
(jackal ile eşit). `rl_config.py:110-120`.

**S: A* yolu policy'ye veriliyor mu?** — Hayır, sadece reward'da (path deviation). `rl_config.py:182-186`.

**S: Hangi algoritma, hangi hiperparametreler?** — SB3 SAC; §9.1.

**S: Kaç dünyada eğitiliyor?** — Varsayılan 50 (moborobo 0,3,…,147), her birinde 5 episode,
döngüsel. Curriculum ile tüm dataset (3 fazda). §9.2–9.3.

**S: Eğitim ne kadar sürer?** — 200k adım × ≥0.1 s ≈ ≥5.6 saat + Gazebo restart'ları.

**S: Robot konumunu nereden biliyor?** — Gazebo ground truth, odometri değil. `barn_env.py:440-443`.

**S: Eğitilen model nasıl değerlendirilir?** — `run_benchmark --planners rl_sac --rl-model PATH`;
120 s sim zamanı, unsafe yok. §12.

---

## 15. Tutarsızlıklar / dikkat edilmesi gerekenler (kanıtlı)

1. **`docs/rl_structure.md` eskimiş değerler:**
   - `:270` "gamma=0.99" → kodda **0.999** (`train_sac.py:333`).
   - `:370` tabloda `MAX_EPISODE_STEPS 500` → moborobo için **1000** (`rl_config.py:90`);
     `:34` de robotsuz yazılmış.
   - `:160-162` "PATH_MAX_DEV = 0.5 m" → kodda **1.0** (`rl_config.py:210`; aynı dokümanın
     `:385` satırı doğru).
   - `:114-116` diyagramda scan `(3, 90)` ve `Conv1d(3→32)` → memory açıkken **(4, 90)**,
     `Conv1d(4→32)`.
   - `:242-253` reward bütçesi (crash ≈ −85, unsafe ≈ −65, "terminal potential −12.5×d")
     **ters işaretli eski sürüme** göre; güncel kodda Φ düzeltmesi **+12.5×d** (§7.4).
   - `:15-16` RLLogger'ın run_config.json/training.log ürettiği yazıyor; eğitim scriptleri
     RLLogger kullanmıyor (§11).
2. **`BarnEnv` docstring'i** `terminal_potential_zero` için "Subtract" diyor (`barn_env.py:158-160`);
   kod **ekliyor** (`:542-543`).
3. **`episodes.csv` `steps`** terminated episode'larda 1 eksik; Monitor/TensorBoard ile 1 fark (§3.4).
4. **Eğitim = test dünyaları** varsayılan ayarda (§9.2).
5. **Eğitim/benchmark farkı:** unsafe yok, fizik sürekli, limit adım değil saniye (§12.2).
6. **`benchmark_env.py:6, 149`** "collision episode'u bitirmez" diyor; kod bitiriyor (`:219-222`).
7. **`RLEvalConfig`** (`EVAL_WORLDS`, `MAX_EVAL_STEPS=600`, `EPISODES_PER_WORLD`) okuduğum
   dosyalarda kullanılmıyor; `EVAL_WORLDS` jackal listesi, moborobo'nunki `WorldConfig`'de.
8. **`SACController` memory tespiti** `rows == RLEnvConfig.SCAN_STACK + 1` karşılaştırmasına
   dayanıyor (`rl_sac_controller.py:173`). Örn. memory'siz ama 4 frame ile eğitilmiş bir
   checkpoint yanlışlıkla "memory açık, stack 3" sanılır. Mevcut default'larda sorun yok.
9. **Jackal + memory:** ScanMemory sektörleri 360°'ye yayar; jackal LiDAR'ı 270° olduğundan arka
   sektörler hep "bilinmiyor → 10 m" olur. Memory'siz yolda ise 90 sektör 270°'ye yayılır — iki
   modda sektörlerin açısal anlamı farklı.
10. **`train_sac.py:26-31`** docstring "ROS master restart" diyor; artık roscore kalıcı
    (`barn_env.py:704-728`).
11. **Curriculum** `set_world_list` yorumu "pointer is reset" diyor; kod mod alıyor
    (`train_sac.py:88-96`).

---

## 16. Sık komutlar

```bash
# Gazebo/ROS zincirini doğrula
python3 train_sac.py --robot moborobo --smoke-test
# Obs ve reward bileşenlerini adım adım gör
python3 train_sac.py --robot moborobo --debug-obs
# Varsayılan eğitim (50 dünya, 200k adım)
python3 train_sac.py --robot moborobo
# Curriculum
python3 train_sac.py --robot moborobo --curriculum --total-steps 300000
# Devam ettir (replay buffer yanında varsa o da yüklenir)
python3 train_sac.py --robot moborobo --resume results/rl/runs/RUN_ID/model_100000_steps
# TensorBoard
tensorboard --logdir results/rl/runs
# Benchmark
python3 -m benchmark.run_benchmark --robot moborobo --planners rl_sac mppi \
    --rl-model results/rl/runs/RUN_ID/model_final
# Birim testleri
pytest rl/tests/ -m "not gazebo" -v
```

---

## Bölüm 2'de yapılabilecekler

- `results/rl/runs/*` altındaki gerçek run'ların `episodes.csv`'lerinden outcome oranları,
  ortalama episode uzunluğu, dünya bazlı başarı (gerçek sayılarla kanıt).
- `git diff` ile HEAD'e göre ne değiştiğinin özeti (reward/safety/memory değişikliklerinin tarihçesi).
- `robots/moborobo` launch dosyalarından LiDAR frekansı, `pointcloud_to_laserscan`
  parametreleri, collision publisher'ın nasıl çalıştığı.
- Klasik planner'larla (MPPI/DWA) benchmark sonuçlarının karşılaştırması.
