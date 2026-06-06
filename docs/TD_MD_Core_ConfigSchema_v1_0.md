# СХЕМА КОНФИГУРАЦИИ `config.yaml`
## TD-MD Core · v1.0 · 2026-06-05

Единая точка задания всех параметров запуска. Движок обязан валидировать конфиг до старта и печатать предстартовый чек-лист. Все величины — в системе `metal` (см. `TD_MD_Core_Units_v1_0.md`).

---

## 1. Аннотированный пример (полный)

```yaml
# === TD-MD Core run configuration ===
run:
  steps: 50000              # число шагов интегрирования
  ensemble: nve             # v1: только nve (NVT/NPT — backlog)
  seed: 20070101            # глобальный seed для всех ГПСЧ (повторяемость)

units: metal                # фиксировано; смена единиц запрещена

precision:
  mode: production_mixed     # production_mixed | deterministic_fp64
  real_type: fp32            # тип расчёта сил при production_mixed
  # accumulation всегда fp64 с фиксированным порядком (INV-9)

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
    transport: auto          # auto | memcpy | p2p | mpi

potential:
  type: morse                # morse | eam | fs | meam | ml
  r_cut: 4.0                 # Å
  shift: true                # энергетический сдвиг на R_cut (pair_modify shift yes)
  morse: { D: 0.29614, alpha: 1.11892, r0: 3.29692 }   # эВ, Å⁻¹, Å
  # eam:  { file: potentials/Al.eam.alloy }            # для type: eam
  # table: { file: ..., smoothing: poly5 }             # табличная форма (Гл.1.3)

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
  telemetry:  { every: 100 }                  # дашборд/лог
  rescue:     { enabled: true, file: rescue.xyz }
  async: true                                 # I/O в фоновом CPU-потоке

verify:
  enabled: false             # true только если собран -DWITH_LAMMPS=ON
  lammps_lib: /opt/lammps/liblammps.so
  golden: reference_data/    # самодостаточные тесты без LAMMPS
  tests: [test_0_step, test_nve_invariant, test_determinism]

logging:
  level: info                # debug | info | warn | error
  dashboard: true            # Live CLI Dashboard (ANSI)
```

---

## 2. Справочник полей (ключевые инварианты валидации)

| Поле | Тип / допустимое | Проверка при старте |
|------|------------------|---------------------|
| `precision.mode` | enum | `deterministic_fp64` ⇒ форсить fp64 везде |
| `decomposition.zone_width` | float ≥ `potential.r_cut` | иначе **fatal** (нарушение причинности) |
| `decomposition.ring.n_nodes` | int ≥ 2 | минимум памяти INV-7 |
| `decomposition.ring.backend` | enum | `streams` ⇒ 1 GPU; `multi_gpu` ⇒ проверить число GPU |
| `boundary.<axis>` | periodic/free | PBC по оси декомпозиции ⇒ включить замыкание кольца (A7) |
| `potential.r_cut` | float ≤ ½·мин. ребро бокса | для min-image при PBC |
| `timestep.C1` | 0<C1≤1 | приоритетный лимит (Гл.3.3) |
| `timestep.C_buf` | ≥ 1.0 | иначе буфер не гарантирует причинность |
| `verify.enabled` | bool | true требует `-DWITH_LAMMPS=ON` |

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
*Связь: единицы — `Units`; параметры метода — `ZoneFSM`/`TZ`; коэффициенты $C_1/K_2/C_3$ — диссертация Гл. 3.3, 3.5.*
