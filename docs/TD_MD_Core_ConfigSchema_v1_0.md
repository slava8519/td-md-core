# СХЕМА КОНФИГУРАЦИИ `config.yaml`
## TD-MD Core · v1.0 · 2026-06-05 · ревизия 2026-06-11 (B-ревью) · ревизия 2026-07-02 (синхронизация схема↔код, метки INERT, §5 — scope CLI)

Единая точка задания всех параметров запуска. Движок обязан валидировать конфиг до старта и печатать предстартовый чек-лист. Все величины — в системе `metal` (см. `TD_MD_Core_Units_v1_0.md`).

> **Контракт синхронизации (2026-07-02, гигиена `_meta/AUDIT_W_PHASE_NEIGHBORS` §7.3; владелец —
> этот документ + `src/io/config.cpp`, меняются ВМЕСТЕ):** каждый ключ этого документа парсер
> ПРИНИМАЕТ без typo-warning. Ключи трёх сортов: **действующие**, **[INERT]** — документирован,
> легален, но в текущей сборке не действует (парсер печатает явную `[config] note`, не молчит и
> не ругается; исключение — `verify.*`: нота только при `enabled: true`, когда пользователь
> реально ждёт поведения — шипованные конфиги несут инертный `enabled: false` без шума),
> **[FATAL-в-CLI]** — валиден на уровне конфига, но CLI-путь отвергает с пояснением
> (см. §5). Неизвестный ключ = warning c именем (защита от опечаток), плохой enum/диапазон = fatal.

---

## 1. Аннотированный пример (полный)

```yaml
# === TD-MD Core run configuration ===
run:
  steps: 50000              # число шагов интегрирования
  ensemble: nve             # v1: только nve (NVT/NPT — backlog)
  seed: 20070101            # глобальный seed для всех ГПСЧ (повторяемость)
  init_temperature: 300.0   # K; >0 — максвелловская инициализация скоростей от seed
                            # (импульс ЦМ обнуляется); 0 — скорости из Velocities/нулевые (M2.6)

units: metal                # фиксировано; смена единиц запрещена

precision:
  mode: production_mixed     # production_mixed | deterministic_fp64
  real_type: fp32            # [INERT] контракт mixed фиксирует FP32-парную математику при
                             #   FP64-геометрии (B5/M4) — ключ типы не переключает
  # Договорное определение production_mixed (= индустриальный "mixed" LAMMPS/HOOMD,
  # см. _meta/MIXED_PRECISION_BESTPRACTICES_2026-06-11.md): координаты — FP64;
  # расчёт парной силы — real_type (FP32); накопление сил/энергии/вириала —
  # фикс-точечное int64 (B1/INV-9: целое ассоциативно => порядок потоков не влияет);
  # интегрирование — FP64. Формат: силы Q24.40 (scale 2^40 на эВ/Å), энергия/вириал
  # Q34.30-аналог (scale 2^30). FP64-atomicAdd — только референс сверки (наш B1 ≡
  # AMBER SPFP). Дефолт движка до появления mixed-пути (M4) — deterministic_fp64
  # (временно: на consumer-Blackwell FP64 1:66, в production он не для горячего пути).

geometry:
  file: reference_data/al_fcc_72.data
  format: lammps_data        # read_data (LAMMPS)

boundary:                    # ось декомпозиции задаётся в decomposition.axis
  x: periodic                # periodic | free
  y: periodic
  z: periodic                # PBC по оси декомпозиции => замыкание кольца (Гл.2.4)

decomposition:
  axis: z                    # ось разбиения на зоны
  mode: by_zone_width        # by_zone_width | by_n_zones
  zone_width: 4.0            # Å (>= R_cut); тюнится под occupancy (A2)
  n_zones: null              # альтернатива zone_width
  cell_size: 2.33            # Å; сетка для пространственной сортировки (A9)
  ring:
    backend: streams         # streams (1 GPU, эмуляция) | multi_gpu
    n_nodes: 4               # узлов кольца = одновременно считаемых шагов
    steps_per_node: 1        # k шагов на узел (Гл. 3.4, ур. 49–51): трафик ↓ в k раз,
                             # до k зон в полёте на узле — рычаг occupancy multi-GPU (B4);
                             # k>1 реализован в MPI-кольце (M5a, тестовый путь), CLI: k=1
    transport: auto          # [INERT] auto | memcpy | p2p | mpi — CLI-кольцо = CPU-эталон;
                             #   GPU/MPI-транспорты тест-ведомые до упаковки M5b

neighbor:
  mode: direct               # direct (эталонный O(N²)) | cluster (M3: Z-order +
                             # 32-атомные кластеры + список пар; сортирует атомы in-place,
                             # идентичность — в id)
  skin: 1.0                  # Å; слой валидности списка пар кластеров (аналог таблиц
                             # Варлета, R_k−R_max); перестройка при смещении > skin/2 (M3)
  verlet:                    # M4-B: PersistentVerlet-лёвер (opt-in). GPU-ring only —
                             # CPU-эталонный ринг (CLI) игнорирует (w-tiling). default=cells.
    enable: false            # bool; включить переиспользование persistent-списка (D авто-
                             # включит при K_pred >= K_on). Дефолт OFF (memory-gated, Bench M4-B)
    K_on: 3.0                # float; порог включения verlet (K_pred = skin/(2·R_buf))
    K_off: 1.5              # float > 0, <= K_on; откат на cells (гистерезис, C_buf-пол)
    default: false           # bool; verlet активен с холодного старта

potential:
  type: morse                # enum ПАРСЕРА: morse | lj | eam. morse/lj — действуют (CLI);
                             #   eam — [FATAL-в-CLI]: конфиг валиден, CLI отвергает с
                             #   пояснением (§5 — many-body через tests/tools).
                             #   sw/tersoff/meam РЕАЛИЗОВАНЫ (M6), но в CLI-enum не входят;
                             #   fs/ml — backlog. Тройное расхождение enum устранено 2026-07-02.
  r_cut: 4.0                 # Å
  truncation: shift          # cut | shift | force_shift (M3, cutoff.hpp):
                             #   cut — простое усечение (NIST SRSW «+LRC»-схемы);
                             #   shift — энергетический сдвиг U−U(rc) (дефолт, golden);
                             #   force_shift — U и F непрерывны на rc (NIST LFS;
                             #   выбор M3 для NVE-прогонов M3.5+, см. Rationale)
  shift: true                # УСТАРЕЛО: bool ⇒ truncation shift/cut; truncation
                             # при одновременном задании выигрывает (warning)
  morse: { D: 0.29614, alpha: 1.11892, r0: 3.29692 }   # эВ, Å⁻¹, Å
  # lj: { epsilon: 1.0, sigma: 1.0 }                   # для type: lj (эВ, Å;
  #                                                    # приведённые единицы — 1.0/1.0)
  # eam:  { file: potentials/Al.eam.alloy }            # [INERT] блок принят, но CLI-путь
  #                                                    #   many-body не поднимает (§5)
  # table: { file: ..., smoothing: poly5 }             # [INERT] табличная форма — backlog (Гл.1.3)

timestep:
  mode: auto                 # auto (C1/K2/C3) | fixed
  dt_initial: 0.005          # ps (= 5 фс); стартовое значение
  dt_max: 0.02               # ps, верхний предел
  C1: 0.1                    # доля буфера для самого быстрого атома (приоритет)
  K2: 50.0                   # K, макс. прирост температуры атома за шаг
  C3: 0.5                    # порог смены шага (0=не менять,1=каждый шаг)
  C_buf: 1.5                 # коэффициент ширины буфера (C>=1)

integrator: velocity_verlet  # фиксировано (Гл.1.2)

io:
  trajectory: { file: traj.lammpstrj, every: 1000, format: lammpstrj }
                             # format — [INERT]: формат фиксирован .lammpstrj
  telemetry:  { every: 100 }                  # [INERT] каденс телеметрии — дело дашборда
                                              #   (флаг CLI --dashboard, M7)
  rescue:     { enabled: true, file: rescue.xyz }   # расширенный XYZ: x y z vx vy vz
                                                    # + step/dt в комментарии (рестарт-пригодный, B9);
                                                    # rescue.format — [INERT]: формат фиксирован
  async: true                                 # [INERT] async-I/O — отложенный M7 sub-PR

verify:                      # [INERT] секция целиком: VerifyLab/liblammps в движок НЕ встроен и
                             #   не планируется как зависимость — эталоны заморожены в
                             #   reference_data/ (+gen_*.in регенерация офлайн-LAMMPS'ом);
                             #   флага -DWITH_LAMMPS НЕ существует (фантом ранних доков, снят
                             #   2026-07-02). enabled: true печатает явную [config] note.
  enabled: false
  lammps_lib: /opt/lammps/liblammps.so
  golden: reference_data/    # самодостаточные тесты без LAMMPS
  tests: [test_0_step, test_nve_invariant, test_determinism]

logging:                     # [INERT] секция целиком: уровень лога/дашборд — CLI-дела
                             #   (live-dashboard включается флагом --dashboard, M7)
  level: info                # debug | info | warn | error
  dashboard: true            # Live CLI Dashboard (ANSI)
```

---

## 2. Справочник полей (ключевые инварианты валидации)

| Поле | Тип / допустимое | Проверка при старте |
|------|------------------|---------------------|
| `precision.mode` | enum | `deterministic_fp64` ⇒ форсить fp64 везде |
| `decomposition.zone_width` | float ≥ `potential.r_cut` | иначе **fatal** (нарушение причинности) |
| `decomposition.ring.n_nodes` | int ≥ 1 | 1 = одноузловой эталонный прогон; минимум памяти INV-7 |
| `decomposition.ring.backend` | enum | `streams` ⇒ 1 GPU; `multi_gpu` ⇒ проверить число GPU |
| `boundary.<axis>` | periodic/free | PBC по оси декомпозиции ⇒ включить замыкание кольца (A7) |
| `potential.r_cut` | float ≤ ½·мин. ребро бокса | для min-image при PBC |
| `timestep.C1` | 0<C1≤1 | приоритетный лимит (Гл.3.3) |
| `timestep.C_buf` | ≥ 1.0 | иначе буфер не гарантирует причинность |
| `timestep.mode` | enum: auto/fixed | неизвестное значение ⇒ **fatal** (не молчаливый fixed) |
| `integrator` | enum: velocity_verlet | иначе **fatal** (валидация добавлена 2026-07-02) |
| `decomposition.ring.steps_per_node` | int; **в CLI-сборке = 1** | k>1 — MPI-кольцо (M5a, тестовый путь); в CLI ≠1 ⇒ **fatal** с подсказкой |
| `decomposition.ring.n_nodes` vs $P_{op}$ | n_nodes ≤ s/s_min | предупреждение, если узлов больше полезного максимума (ур. 44–45) |
| `neighbor.skin` | float > 0 | слой списка пар; кадрность перестройки из C1-оценки |
| `neighbor.verlet.K_off` | float > 0 | гистерезис-низ; `K_on >= K_off` ⇒ **fatal** иначе (M4-B) |
| `neighbor.verlet.{enable,K_on,default}` | bool/float/bool | PersistentVerlet-лёвер (opt-in, GPU-ring); default=cells (memory-gated, Bench M4-B) |
| `potential.type` | enum: morse/lj/eam | неизвестный тип ⇒ **fatal**; eam ⇒ [FATAL-в-CLI] с пояснением (§5) |
| `potential.truncation` | enum: cut/shift/force_shift | неизвестная схема ⇒ **fatal** |
| `potential.lj.{epsilon,sigma}` | float > 0 | иначе **fatal** |
| неизвестные ключи | — | warning с именем ключа (защита от опечаток) |
| [INERT]-ключи (см. §1) | — | явная `[config] note` — не warning и не молчание |
| `verify.enabled` | bool | true ⇒ `[config] note` (VerifyLab не встроен; эталоны — reference_data/) |

---

## 3. Предстартовый чек-лист (движок печатает)

Доступные GPU и их память · выбранный backend кольца и число узлов/стримов · система единиц и константы · топология зон (число, ширина, ось) · режим точности · ГУ и признак замыкания кольца · потенциал и $R_{cut}$ · стартовый $\Delta t$ · предупреждение, если `backend: streams` (режим эмуляции — метрики производительности недостоверны, A3).

---

## 4. Минимальный конфиг для walking skeleton (M0)

```yaml
run: { steps: 100, ensemble: nve, seed: 1 }
units: metal
precision: { mode: deterministic_fp64 }
geometry: { file: reference_data/al_fcc_72.data, format: lammps_data }
boundary: { x: periodic, y: periodic, z: periodic }
decomposition: { axis: z, mode: by_n_zones, n_zones: 1, ring: { backend: streams, n_nodes: 1 } }
potential: { type: morse, r_cut: 4.0, shift: true, morse: { D: 0.29614, alpha: 1.11892, r0: 3.29692 } }
timestep: { mode: fixed, dt_initial: 0.005 }
integrator: velocity_verlet
io: { trajectory: { file: traj.lammpstrj, every: 50 } }
verify: { enabled: false, golden: reference_data/, tests: [test_0_step] }
```

---

## 5. Scope CLI-бинаря `tdmd` (запись решения, 2026-07-02)

Находка `_meta/AUDIT_W_PHASE_NEIGHBORS_2026-07-02.md` §7.1: физика M6 (EAM/SW/Tersoff/MEAM,
GPU/MPI-кольца) полностью реализована и валидирована, но живёт в tests/tools; из `config.yaml`
доступны только Morse/LJ. Аудит требовал явного решения владельца. **Решение:**

- **CLI v1 = парное демо**: Morse/LJ × (direct | cluster | CPU-эталонное кольцо) + `--dashboard`.
  Это витрина метода TD (детерминизм, авто-шаг, HALT/rescue), а не продуктовый MD-раннер.
- **Many-body и GPU/MPI-кольца** запускаются через тестовые/инструментальные харнессы
  (`Test_EAM*/Test_SW*/Test_Tersoff*/Test_MEAM*`, `bench_*`, `eam_drift`/`eam_coexist`/
  `eam_rdf_stat` и т.д.) — там же живут их гейты корректности.
- `potential.type: eam` в конфиге **валиден** (парсится и валидируется), но CLI отвергает его
  fatal-сообщением со ссылкой сюда — чтобы он никогда молча не провалился в Морзе-ветку.
- **CLI-интеграция many-body/GPU-колец — отдельный трек** (не гигиена): требует ключей файла
  setfl/параметров SW/Tersoff/MEAM, выбора кольца и упаковки прогонов; берётся осознанным PR
  с собственным дизайном, если/когда понадобится продуктовый запуск из конфига.

---
*Связь: единицы — `Units`; параметры метода — `ZoneFSM`/`TZ`; коэффициенты $C_1/K_2/C_3$ — диссертация Гл. 3.3, 3.5.*
