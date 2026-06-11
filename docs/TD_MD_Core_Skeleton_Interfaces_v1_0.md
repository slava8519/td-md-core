# КАРТА МОДУЛЕЙ И ИНТЕРФЕЙСЫ
## TD-MD Core · v1.0 · 2026-06-05 · контракты до начала кода

Назначение — зафиксировать раскладку репозитория и ключевые C++-интерфейсы, чтобы модули не разъехались между шагами разработки. Сигнатуры — ориентир (C++20); агент уточняет детали, но **границы модулей и контракты держит**.

---

## 1. Раскладка репозитория

```
td-md-core/
├── CMakeLists.txt              # C++20, опции -DWITH_LAMMPS, -DPRECISION
├── Apptainer.def · spack.yaml  # HPC-деплой
├── .github/workflows/ci.yml    # линт, сборка GCC/Clang, юнит-тесты
├── config/                     # примеры config.yaml
├── reference_data/             # золотые данные (готово)
├── docs/                       # эти md-инструкции (источник истины)
├── include/tdmd/
│   ├── hal/         hal.hpp, transport.hpp, event.hpp      # абстракция железа
│   ├── core/        soa.hpp, zone.hpp, fsm.hpp,
│   │                conveyor.hpp, integrator.hpp, buffer.hpp
│   ├── potentials/  ipotential.hpp, morse.hpp, eam.hpp, fs.hpp, ml.hpp
│   ├── io/          config.hpp, reader_lammps.hpp,
│   │                writer.hpp, dashboard.hpp, rescue.hpp
│   └── verify/      verifylab.hpp, lammps_bridge.hpp
├── src/                        # реализации (.cpp / .cu)
└── tests/                      # Google Test: unit/ + integration/
```

**Принцип:** физика потенциалов (`potentials/`) не знает о CUDA напрямую — только через `hal/`. Логика конвейера (`core/conveyor`) не знает о вендоре транспорта — только через `ITransport`.

---

## 2. HAL — слой аппаратной абстракции (`hal/`)

```cpp
namespace tdmd::hal {
  using Stream = /* cudaStream_t | заглушка CPU */;
  using Event  = /* cudaEvent_t  | ... */;

  Stream make_stream();
  void   record(Event, Stream);              // продюсер после CALC_END
  void   wait  (Stream, Event);              // консьюмер перед CALC_START (INV-2)
  void   memcpy_async(void* dst, const void* src, size_t n, Stream);

  // детерминированное накопление силы (INV-9, реализация B1 ≡ AMBER SPFP —
  // см. ZoneFSM §8 и _meta/MIXED_PRECISION_BESTPRACTICES_2026-06-11.md):
  // фикс-точечное int64 Q24.40 — целочисленное сложение ассоциативно, результат
  // не зависит от порядка потоков/блоков. Масштаб 2^40 отсчётов на эВ/Å: диапазон
  // целой части ±2^23 ≈ ±8.4e6 эВ/Å (запас ~1e4-1e5x против реальных сил Al/Морзе;
  // ≥80x в adversarial-пределе до HALT-по-перекрытию). Аппаратный atomicAdd(double) —
  // НЕфиксированный порядок (на consumer-Blackwell ~40x медленнее int64) — допустим
  // только как референс сверки, не основной путь.
  // ОПТИМИЗАЦИЯ (из LAMMPS): копить парные вклады в локальном FP-регистре в inner-loop
  // по соседям, конвертировать (cvt.rni.s64.f32) и atomicAdd(int64) ОДИН раз на атом
  // при сбросе — не на каждый парный вклад (порядок атомов фиксирован Z-order).
  template<typename Real>
  __device__ void accumulate(long long* dst, Real val, Real scale);

  // макросы переносимости: HAL_SHARED_MEMORY, HAL_SYNC_THREADS, HAL_ATOMIC
}
```

### 2.1. Транспорт зон (`hal/transport.hpp`) — закрывает A4

```cpp
struct ITransport {                          // одна логика для эмуляции и кластера
  virtual void post_send(ZoneView, int to_node, Event done) = 0;
  virtual void post_recv(ZoneView, int from_node, Event ready) = 0;
  virtual void progress() = 0;               // неблокирующий прогон очередей
  virtual ~ITransport() = default;
};
// Реализации: StreamTransport (memcpy_async), P2PTransport (memcpyPeer/NVLink), MpiTransport
```

---

## 3. Данные атомов и зоны (`core/`)

```cpp
template<typename Real>
struct AtomSoA {                             // строго SoA (ТЗ §4)
  double *x, *y, *z;                         // глобальные координаты FP64
  Real   *xo, *yo, *zo;                      // локальные offsets FP32 от границы зоны
  Real   *vx, *vy, *vz;
  long long *fx_acc, *fy_acc, *fz_acc;       // накопление сил: int64 фикс-точка (B1/INV-9)
  int     n;                                 // (конвертация в Real — на выходе редукции)
};

enum class ZoneType { o, d, w, c, p };       // см. ZoneFSM §2 (буквы по диссертации)

struct Zone {
  int id, step_h, n_atoms;
  ZoneType type = ZoneType::o;   // свободна
  double left_bound_glob;                    // граница зоны (для offsets)
  bool   force_complete = false;            // INV-3
  uint32_t contrib_mask = 0;                // вклады соседей
  double v_max_local = 0, R_buf_local = 0;
  // срез AtomSoA для атомов зоны
};
```

---

## 4. Конечный автомат зон (`core/fsm.hpp`)

```cpp
enum class ZoneEvent { RECV, SPHERE, START, END, SEND };

struct ZoneFSM {
  // таблица переходов ZoneFSM §4; недопустимый переход -> error/rescue
  static ZoneType next(ZoneType from, ZoneEvent e);     // чистая функция
  static bool     allowed(ZoneType from, ZoneEvent e);
  // применяет переход к зоне + выполняет action (накопление сил/интеграция вызываются снаружи)
  static void apply(Zone&, ZoneEvent);
};
```

> Реализуется **строго** по таблице переходов `TD_MD_Core_ZoneFSM_v1_0.md §4` и инвариантам §6. Это первый юнит-тестируемый артефакт (M1).

---

## 5. Потенциалы (`potentials/ipotential.hpp`)

```cpp
template<typename Real>
struct IPotential {
  // силы внутри зоны + парное взаимодействие со следующей зоной (Ньютон-3 -> partial в neighbor)
  virtual void compute(const Zone& self, Zone& neighbor,
                       AtomSoA<Real>& a, hal::Stream) const = 0;
  virtual Real r_cut() const = 0;
  virtual bool is_many_body() const = 0;     // EAM/FS/MEAM -> true (отдельные ядра)
  virtual ~IPotential() = default;
};
// Morse: парный, эталон. EAM/FS/MEAM: is_many_body()=true, собственные CUDA-ядра (ТЗ §5).
```

---

## 6. Интегратор и буфер (`core/`)

```cpp
struct IIntegrator {                         // скоростная форма Варлета (Гл.1.2)
  virtual void first_half (Zone&, AtomSoA<>&, double dt) = 0;   // r, v(½) по f(t)
  virtual void second_half(Zone&, AtomSoA<>&, double dt) = 0;   // v по f(t+dt)
};

namespace buffer {
  double compute_R_buf(double v_max, double dt, double C);      // ур.33; C>=1
  bool   causality_ok (double v_max, double dt, double R_buf);  // INV-4 -> иначе rescue
  double auto_dt(double v_max, double R_buf, const TimeStepCfg&); // C1/K2/C3 (Гл.3.3)
}
```

---

## 7. Конвейер (`core/conveyor.hpp`) — оркестратор

```cpp
class TimeConveyor {
public:
  TimeConveyor(const Config&, ITransport&, IPotential<>&, IIntegrator&);
  void run(int steps);                       // главный цикл, GPU-resident
private:
  void step_node(int node);                  // порядок send/recv по чётности (ZoneFSM §7.4)
  // держит зоны узла (>=2, INV-7), применяет ZoneFSM, транспорт, события
};
```

**Поток одного подинтервала** (псевдокод — `ZoneFSM §8.2`): try_send/try_recv по чётности → `causality_ok` → START → compute (potential) → END (reduce fp64 + integrate + auto_dt + buffer check) → пометить SEND.

---

## 8. Правила сборки точности (закрывает A6)

- Один набор ядер, шаблон `template<typename Real>`; инстанциации `float` и `double`.
- `precision.mode = deterministic_fp64` → инстанциация `double` + фиксированный порядок редукции.
- CMake-опция `-DTDMD_PRECISION=mixed|fp64` управляет дефолтом; режим также переключается из конфига.

---

## 9. Что тестируется на каждом уровне (вход для tests/)

| Модуль | Юнит-тест |
|--------|-----------|
| `fsm` | все переходы/запреты (таблица §4) |
| `buffer` | `R_buf`, детект нарушения причинности |
| `potentials/morse` | силы vs `reference_data` (Test_0_Step) |
| `integrator` | сохранение энергии NVE на коротком прогоне |
| `conveyor` | детерминизм 1 vs N стримов (INV-9), anti-deadlock |

---
*Интерфейсы — инженерное решение `[ENG]`, согласованное с методом. Контракты автомата и буфера — строго по диссертации (Гл. 2.1, ур. 33) и `ZoneFSM`.*
