All facts verified against the tree. Producing the design doc.

---

# PR-0c — W-контракт, шаг «каркас соседей + rebuild_epoch»: дизайн-док (готов к реализации)

Ветка `opus`, HEAD `b15083d`. Дом-стиль: состязательный дизайн ДО, приёмка ПОСЛЕ; F-NOOP; SSoT-примитив; цитаты file:line.

## 0. ВЕРДИКТ (решающий фактор — structurally-dead урок PR-0a D2)

**Device-каркас `cuda/zone_nbr_store.cuh` (Tier-0/1/2 буферы + ядра + пер-ранговый `local_epoch[]`) — ОТЛОЖЕН к своим потребителям (PR-1/PR-3a/PR-4). СЕЙЧАС поставляем ровно три вещи:**

1. инертное поле `uint32_t rebuild_epoch = 0` в `core::ZoneHeader` (ADD-alongside, wire-safe, sizeof-нейтрально);
2. ЧИСТЫЙ host-примитив эпохи `core/rebuild_epoch.hpp` (SSoT, который потребит PR-4) + CPU-юнит-тест `Test_Rebuild_Epoch` с kill-мутацией на зуб;
3. `cuda/zone_nbr_store.cuh` как **device-FREE контрактный дом** (§5 требует файл-«каркас» дословно) — re-export примитива + §3.5-контракт Tier-0/1/2 с MUST-FIX-клаузами verbatim + явные DEFER-записи «потребитель приходит с каркасом». **Ноль device-буферов, ноль ядер, ноль `cudaMalloc`.**

**Обоснование (буква аудита + 5 прецедентов рецидива structurally-dead):** §5 говорит «**каркас**» (scaffold), НЕ «реализовать/загейтить»; реальные потребители ярусов расписаны по PR-1 (Tier-0 биннинг+G-B, `AUDIT:342-346`), PR-3a (Tier-1 window-view, `:350-352`), PR-4 (Tier-2 CSR + активация эпохи, `:355`). Инертный device-буфер, который ни одно ядро не трогает, — это ровно правило D2 (`PR0A_WCONTRACT_DESIGN §11`: обязательный-но-мёртвый хук у 7+ политик = structurally-dead, входит В концепт ВМЕСТЕ с потребителем через осознанный compile-break) и формула анти-мёртвости PR-0a (`:451`: «гейт по вкомпилированной константе никогда не бросает ⇒ мёртв и нетестируем — 5-раз-наступленные грабли Te1 fc_d, Me1 partial-S, Me5 taper, Me5b OOB»). Прецедент прямой: PR-0a поставил `donation_layout` как ЧИСТУЮ юнит-тестируемую функцию, device-состояние (ρ-аккумуляторы) пришло с PR-1; PR-0b поставил `core/skin_budget.hpp` host-примитив + CPU-тест и в шапке (`skin_budget.hpp:6-8`) уже назвал «PR-4, ZoneNeighborStore» будущим потребителем. PR-0c — прямое продолжение того же паттерна: следующий SSoT-примитив (эпоха), который потребит тот же PR-4. **Wire-safety (см. §2 ниже) снимает единственный аргумент «за» ранний device-каркас** — резервировать место в ABI заранее НЕ нужно, поле садится в любой момент без wire-разрыва.

---

## 1. Файлы (новые / изменённые)

| Файл | Тип | Что |
|---|---|---|
| `include/tdmd/core/transport.hpp` | ИЗМ | +1 инертное поле `rebuild_epoch` в `ZoneHeader` (после `verlet_active`, стр. 49) + комментарий |
| `include/tdmd/core/rebuild_epoch.hpp` | **НОВ** | ЧИСТЫЙ host-примитив эпохи (SSoT); зеркало `core/skin_budget.hpp` |
| `include/tdmd/cuda/zone_nbr_store.cuh` | **НОВ** | device-FREE контрактный дом ZoneNeighborStore: re-export примитива + §3.5-контракт + DEFER-записи |
| `tests/test_rebuild_epoch.cpp` | **НОВ** | `Test_Rebuild_Epoch`, kill-мутация на зуб; включает `zone_nbr_store.cuh` (g++-компиляция каркаса) |
| `CMakeLists.txt` | ИЗМ | регистрация `test_rebuild_epoch` рядом с `test_skin_budget` (стр. 383-385) |

**Байт-нетронуты (F-NOOP):** `conveyor_gpu.cuh`, `mpi_ring_edge.hpp`, `skin_budget.hpp`, `fsm.hpp`, все кольца, все `zone_*`/`cuda/*` силовые пути. Никакой правки боевого send/arrival — примитив unit-тестируется автономно (как `donation_layout` в PR-0a, которого «в PR-0a не потребляет ни одно кольцо»).

---

## 2. (a) Поле `rebuild_epoch` на `ZoneHeader` — размещение, комментарий, инертность, wire-parity

**Размещение — сразу после `verlet_active` (`transport.hpp:49`), в семантическом блоке `rebuild_now`:**

```cpp
  double  skin_consumed = 0.0;   // accumulated 2*R_buf budget since last rebuild
  uint8_t rebuild_now   = 1;     // 1 => materialize the list this pass
  uint8_t verlet_active = 0;     // 1 => Verlet reuse path; 0 => cell-raster (Базис A)
  // PR-4 lazy MPI materialization (INERT until PR-4; rebuild_now stays the ACTIVE
  // decision). Monotone per-head rebuild counter: a rank materializes zone z's list
  // iff local_epoch[z] < rebuild_epoch (core/rebuild_epoch.hpp). Default 0 => no rank
  // ever sees a stale local epoch => bitwise no-op, exactly like d_full/drift_full ride
  // zero when verlet_hybrid=false. z-independent broadcast scalar (NL-INV-4), same
  // induction as skin_consumed. Consumers (PR-4): conveyor_gpu.cuh:719-720/750-752/
  // 786-788 (read arrival-0), :1136-1138/1219-1221 (write head sent==0).
  uint32_t rebuild_epoch = 0;
  // PR-3 hybrid criterion ... (d_full/drift_full — без изменений)
  double d_full = 0.0;
  double drift_full[3] = {0, 0, 0};
```

**Sizeof-нейтральность (target ABI, сильнейшая форма wire-parity).** Текущая раскладка хвоста: `double skin_consumed` (8-выровнен), `uint8 rebuild_now` + `uint8 verlet_active`, затем **6 байт padding** до 8-выровненного `double d_full`. `uint32_t` требует 4-выравнивания: после двух uint8 (offset 8k+10) ближайший 4-aligned = 8k+12 ⇒ `rebuild_epoch` ложится в 8k+12..15, `d_full` остаётся на 8k+16. **`sizeof(ZoneHeader)` и `sizeof(GpuHeader)` НЕ меняются** — поле занимает существующую дыру. То есть 4 из 6 ранее-неопределённых padding-байт теперь держат определённый 0, длина wire-образа не растёт.

**Wire-parity подтверждён end-to-end (даже если бы sizeof вырос — safe by construction):**
- `ZoneHeader` пересекает провод ТОЛЬКО как первый член `GpuHeader` (`conveyor_gpu.cuh:188-192`).
- **MPI:** `std::memcpy(p.buf.data(), &h, sizeof h)` на send (`mpi_ring_edge.hpp:111`), `std::memcpy(&h, rbuf_.data(), sizeof h)` на recv (`:122`); пул `p.buf.resize(sizeof(cuda::GpuHeader)+bytes_)` (`:56`), `rbuf_.resize(...)` (`:57`); poison — `sizeof fwd`/`sizeof p` (`:129,144`). Всё sizeof-производно, хардкод-offset'ов в поля hdr НЕТ. Оба конца перекомпилируются с одним хедером ⇒ согласованы. Проверка `if (bytes != bytes_)` (`:94,119`) — на PAYLOAD, не заголовок; поле её не трогает.
- **Внутрикольцо:** `SpscChannelT<GpuHeader>` (`:208`) — by-value (`out.chan_.send(std::move(gh))` :1225, `in.chan_.recv(gh)` :774), размер транспарентен.
- **Агрегатная инициализация — БЕЗОПАСНО:** grep подтвердил ноль позиционных `ZoneHeader{...}`/`GpuHeader{...}` в дереве; все использования — default/value/copy/пофилдовое присваивание (`GpuHeader p{}` :141, `GpuHeader gh;` :728). Интерьерная вставка с default-инициализатором ничего не ломает.

**Инертность (written-to-default-0-and-never-read):** ни один send-сайт (`:1128-1138`, `:1211-1221` пишут только `dt_next/v_full/a_full/k2cap_full/d_full/drift_full/skin_consumed/rebuild_now/verlet_active`) и ни один arrival-0 сайт (`:716-721`, `:748-753`, `:786-788` читают только `dt_next/skin_consumed/rebuild_now/verlet_active` + Λ-цепочку) в PR-0c поле не касается. `rebuild_now` остаётся ACTIVE-решением. Прецедент — `d_full`/`drift_full` (`transport.hpp:53` «Both inert unless their flag is set»), едут нулями пока `verlet_hybrid=false`.

**Оговорка (честно, в doc):** `uint32` эпоха не оборачивается в реализуемом прогоне (нужно >4·10⁹ перестроек; при перестройке каждый шаг и 10⁶ шагах — 10⁶ ≪ 2³²) ⇒ предикат `local < hdr` корректен без wrap-логики; G-EPOCH (PR-4) wrap не тестирует.

---

## 3. (b) SSoT-примитив эпохи — `include/tdmd/core/rebuild_epoch.hpp` (ЧИСТЫЙ host, БЕЗ CUDA)

**Почему `core/*.hpp`, а не device:** побитовость ленивой эпохи держится на ДВУХ чисто-скалярных host-вычислимых свойствах (Reader 2, L-SUP `AUDIT:204-206`): (i) предикат `local_epoch < rebuild_epoch` и (ii) монотонность пары (эпоха, skin_consumed). CSR-буферы к побитовости отношения не имеют. Прямой сиблинг `core/skin_budget.hpp` (PR-0b извлёк Λ-скин-рекуррентность как pure host, «no CUDA ⇒ independently unit-testable»). Эпоха — следующий SSoT, который потребит ТОТ ЖЕ PR-4.

```cpp
#pragma once
#include <cstdint>

// PR-0c (W-contract ladder) — the lazy-materialization EPOCH primitive, the SINGLE
// SOURCE OF TRUTH the many-body ring port (PR-4, ZoneNeighborStore) consumes instead
// of re-deriving it. Pure host C++ (no CUDA) ⇒ independently unit-testable
// (tests/test_rebuild_epoch.cpp). INERT in PR-0c: no ring site calls these yet — the
// primitive is exercised ONLY by its unit test, exactly as donation_layout was in PR-0a.
//
// Design of record: docs/_meta/AUDIT_W_PHASE_NEIGHBORS_2026-07-02.md §3.5 (MPI lazy epoch).
//
// SEMANTICS (for PR-4): rebuild_now stays the pass-local ACTIVE decision (decide_pass,
// core/skin_budget.hpp). The epoch is a MONOTONE per-head counter that ticks EXACTLY on a
// rebuild pass and rides EVERY header (rebuild_epoch, NL-INV-4). A rank keeps
// local_epoch[zone_id] (PR-4 per-rank state, NOT shipped) and materializes zone z's list
// from ITS OWN current positions iff stale(local_epoch[z], hdr.rebuild_epoch).
//
// z-INDEPENDENCE: rebuild_epoch is a z-independent broadcast scalar (head decision in every
// header), same induction as dt_next/skin_consumed ⇒ the epoch path inherits 1-vs-z bitwise.
//
// L-SUP BITWISENESS (AUDIT §3.2, "ленивые эпохи"): between two ticks a rank REUSES its list.
// Bitwise identity to a per-pass rebuild holds because the charged list (skin >= 2*R_buf) is
// a SUPERSET of the true neighbours AND the force kernel's exact r^2<rc^2 retest zeroes any
// extra candidate (W-1 "same double contributions" + L-CLOSE). The load-bearing bridge to
// L-SUP is the EPOCH-vs-REBUILD CONSISTENCY below: advance() strictly increases iff a rebuild
// occurred ⇒ a lagging rank rebuilds on EXACTLY the passes the head did — no rank lazily skips
// a mandatory rebuild (conservativeness), none rebuilds on a reuse pass (laziness).
namespace tdmd::core::rebuild_epoch {

// Advance the head's rebuild epoch. `prev` = epoch shipped on the last header; `rebuild` =
// decide_pass's rebuild flag for THIS pass. Returns the epoch to broadcast this pass.
//   epoch(h) = epoch(h-1) + (rebuild ? 1 : 0)
// Monotone non-decreasing; STRICTLY increases iff a rebuild occurred (the L-SUP bridge).
constexpr std::uint32_t advance(std::uint32_t prev, bool rebuild) {
  return prev + (rebuild ? 1u : 0u);
}

// Lazy-materialization predicate. A rank whose local list is at generation `local` must
// (re)materialize the zone's list iff the head has advanced strictly past it. STRICT less:
// equal epochs REUSE (list current, laziness), lagging epochs REBUILD (conservativeness).
// Total/robust for local > hdr (unreachable ahead-of-head) => reuse.
constexpr bool stale(std::uint32_t local, std::uint32_t hdr) {
  return local < hdr;
}

}  // namespace tdmd::core::rebuild_epoch
```

**Что PR-4 сделает с этим (НЕ в PR-0c, для контекста реализатора):** WRITE — в блоке `sent==0` рядом с `gh.hdr.rebuild_now = rebuild_next?1:0` (`conveyor_gpu.cuh:1137,1220`) добавит `gh.hdr.rebuild_epoch = head_epoch = advance(head_epoch, rebuild_next)`; фактическая материализация списка — под `if (rebuild_now_pass)` в `launch_pairs` (`:902`), где эпоха головы становится «текущей». READ — на arrival-0 (`:748-753/786-788/716-721`) заменит/дополнит булев `rebuild_now` сравнением `stale(local_epoch_[want_id], gh.hdr.rebuild_epoch)`. Пер-ранговый `std::vector<uint32_t> local_epoch_` размера `n_` селится РЯДОМ с `vl_off_`/`vl_idx_`/`vl_xref_` (`:570-582`, индекс `A.fsm.id`, статическое членство `:896`) — **это device/host-состояние заводит PR-4, НЕ PR-0c** (иначе ровно тот инертный буфер, что проект DEFER-ит).

---

## 4. План теста — `tests/test_rebuild_epoch.cpp` → `Test_Rebuild_Epoch` (kill-мутация на зуб)

Зеркало `tests/test_skin_budget.cpp`. Каждый тест несёт свою мутацию.

| Тест | Что пинит | KILL-мутация (обязана покраснеть) |
|---|---|---|
| **T1 `AdvanceTicksIffRebuild`** | `advance(e,false)==e`; `advance(e,true)==e+1` | `+ (rebuild?1:0)` → безусловное `+1` (тик без rebuild) ⇒ false-ветка RED; либо `(rebuild?0:1)` ⇒ обе ветки RED |
| **T2 `MonotoneNonDecreasing`** | прогон канонической bool-последовательности перестроек: эпоха НИКОГДА не убывает, тикает ровно на `true` | любой decrement / reset-в-0 на non-rebuild ⇒ RED |
| **T3 `EpochRebuildConsistency`** (несущий L-SUP-мост) | `advance(prev,r) > prev  ⟺  r` по последовательности | тик на неверном условии (напр. `advance` игнорит `rebuild`) ⇒ эквивалентность RED |
| **T4 `StalePredicateStrictLess`** | `stale(k,k)==false` (reuse), `stale(k-1,k)==true` (rebuild), `stale(k+1,k)==false` (total) | `<`→`<=` ⇒ `stale(k,k)` RED (rebuild-шторм, убита лень); `<`→`!=` ⇒ `stale(k+1,k)` RED |
| **T5 `LazyReuseConservativeness`** (интеграция обоих + инвариант монотонности) | симуляция: голова advance-ит эпохи по последовательности rebuild-решений, отстающий ранг вызывает `stale` + догоняет (`local=hdr`). Ассерт: ранг материализует РОВНО на passes, где голова перестроила; НИКОГДА не пропускает rebuild (консервативность); НИКОГДА не строит на reuse-проходе (лень) | предикат `<=` ⇒ rebuild-шторм; `advance` без тика ⇒ ранг живёт со stale-списком ⇒ ассерт «материализовал на всех rebuild-passes» RED |
| **T6 `DeterministicPure`** | одинаковые входы ⇒ одинаковые выходы (proxy z-независимости; header-эпоха — z-независимый broadcast-скаляр, та же индукция что `skin_consumed`) | зеркало `SkinBudget.DeterministicPure` |

**Анти-vacuity:** T3 и T5 — исполнимый мост L-SUP (эпоха строго растёт ⟺ голова перестроила ⇒ отстающий ранг перестраивает ровно тогда же), прямая аналогия «эпоха инкрементируется ⟺ rebuild_now взведён» из PR-0b. Это делает примитив EXERCISABLE-NOW, а не мёртвым.

**CMake (рядом с `test_skin_budget`, `CMakeLists.txt:383-385`):**
```cmake
add_executable(test_rebuild_epoch tests/test_rebuild_epoch.cpp)  # PR-0c: SSoT epoch primitive
target_link_libraries(test_rebuild_epoch PRIVATE tdmd_core GTest::gtest_main)
add_test(NAME Test_Rebuild_Epoch COMMAND test_rebuild_epoch)
```

---

## 5. `cuda/zone_nbr_store.cuh` — «каркас» как device-FREE контрактный дом (решение + содержимое)

**Решение:** файл СОЗДАЁТСЯ (честит букву §5 «каркас cuda/zone_nbr_store.cuh»), но в PR-0c это **device-FREE контрактная поверхность**: ноль `__global__`, ноль `cudaMalloc`, ноль `#include <cuda_runtime.h>` для примитива — чистый C++, компилируемый g++. Содержит РОВНО:
1. `#include "tdmd/core/rebuild_epoch.hpp"` + `namespace tdmd::cuda { namespace nbr_store { using core::rebuild_epoch::advance; using core::rebuild_epoch::stale; } }` — единственный ЖИВОЙ (уже оттестированный) элемент, чтобы будущие device-потребители писали `cuda::nbr_store::stale(...)`;
2. большой doc-comment: **§3.5 таблица Tier-0/1/2 verbatim + ВСЕ MUST-FIX-клаузы** (Tier-0 биннинг дрейфанувших атомов «слэб + drift-padding ≥ g + sticky-страж выхода ЛИБО CSR над глобальным индексным пространством; L-SUP страхует лишний ЗАПРОШЕННЫЙ слэб, но НЕ донора, выпавшего из всех CSR; фикстура донор-дрейф-через-границу-слэба в G-B; клампинг за экстентом + wrap стенсила на шве»; Tier-1 «default ТОЛЬКО EAM+SW, Tersoff/MEAM держат host gather+canonical-sort»; Tier-2 «opt-in, mem-gate `neighbor.mem_budget_gib` default 3.5 + VRAM-probe; EAM ≈2.1 ГБ@1e6 проходит, ≈14–21@1e7 блокирует ⇒ фолбэк cells; angular NO-GO до ncu-пробы»);
3. явные **DEFER-записи**: `// DEFERRED — consumer arrives with the scaffold (PR-0a D2): Tier-0 CSR binning + G-B straddle fixture => PR-1/PR-3a; Tier-1 window-view => PR-3a; Tier-2 per-(zone,role) Verlet-CSR + per-rank local_epoch[] device state + mem-gate + G-EPOCH => PR-4.` Никакого `NeighborTier` enum — vocabulary-enum, на который никто не свитчит, сам по себе мягко-мёртв; ярусный тип заводит потребитель Tier-0 (PR-1/3a) вместе с логикой.

Это точный приём PR-0a: контракт `donation_layout` жил как 8-клаузный SPEC doc-comment, поставленный ДО потребителей PR-1/2. Здесь Tier-контракт живёт doc-comment'ом, поставленным ДО потребителей PR-1/3a/4; единственный исполняемый элемент файла — re-export уже-оттестированного примитива (ничего мёртвого).

**DEFERRED к PR-4 и ПОЧЕМУ (правило consumer-arrives):**
- **пер-ранговый `local_epoch_` (device/host-состояние)** — активируется вместе с epoch-READ на arrival и epoch-WRITE на send (PR-4). Заводить `std::vector<uint32_t>` сейчас = инертный буфер, который ни один сайт не читает = ровно D2-класс.
- **Tier-0/1/2 CSR-буферы + ядра** — у ярусов нет потребителя и нет гейта в лестнице до PR-1 (Tier-0 биннинг+G-B), PR-3a (Tier-1 window-view), PR-4 (Tier-2). Строить device-store с G-B фикстурой (донор-дрейф-через-слэб) сейчас = 6-й рецидив structurally-dead.
- **активация эпохи (epoch WRITE/READ в боевом кольце) + G-EPOCH** — PR-4 (`AUDIT:355`). **MUST-FIX эпохи (в PR-чеклист PR-4, не PR-0c):** G-EPOCH включает `verlet_hybrid=ON` (валидность `d_lagged/x_ref` при ПЕР-РАНГОВЫХ моментах постройки — `x_ref` снимается в момент локальной материализации); `Test_MPI_Conveyor` не гоняется ни в одном CI-контуре ⇒ гейт исполняется ЛОКАЛЬНО (`AUDIT:275-278`).

---

## 6. (d) F-NOOP гейт (существующий suite байт-идентичен)

Тот же класс, что PR-0a/0b.

1. **Поле default-0-never-read:** §2 показал — ни send, ни arrival его не касаются; `rebuild_now` остаётся ACTIVE-решением; ни `decide_pass` (`skin_budget.hpp:41-65`), ни `auto_dt` его не принимают. ⇒ ни одно решение не меняется.
2. **Wire-parity:** §2 — sizeof-нейтрально на target ABI (поле в padding-дыре); безусловно safe by construction (memcpy обоих концов, пул под `sizeof(GpuHeader)`, `SpscChannel<GpuHeader>` by-value, poison bare-`sizeof`). Оба конца перекомпилируются с одним хедером.
3. **Verlet/MPI bitwise:** GPU verlet 1-vs-z (`Test_CUDA_Conveyor`) и `Test_MPI_Conveyor` остаются побитовыми — поле не читается, тот же код-путь. NL-INV-4 не тронут.
4. **A/B-hex пояс (пояс PR-0a):** прогнать существующий детерминированный инструмент HEAD vs PR-0c (напр. `eam_drift` ~500 шагов ИЛИ ring-NVE) → **финальное состояние (atoms) байт-в-байт**. ВАЖНО: пояс хеширует ФИНАЛЬНОЕ СОСТОЯНИЕ СИМУЛЯЦИИ, НЕ raw-заголовки провода — padding-байты никогда не стабильный comparand (4 из них меняются indeterminate→определённый-0, но это невидимо решениям и payload-проверке). Финальное состояние зависит ТОЛЬКО от решений, решения неизменны ⇒ hex идентичен.
5. **Чистый ребилд:** CPU-suite N/N (+`Test_Rebuild_Epoch`) байт-идентичен + CUDA-suite M/M байт-идентичен + memcheck/racecheck чисто (поле не вводит device-работы).
6. **Diff-зуб приёмки:** `git diff` по 5 сайтам (send `:1128-1138/1211-1221`, arrival `:716-721/748-753/786-788`) = 0 — `rebuild_epoch` там не появляется. Мутация «вписать `gh.hdr.rebuild_epoch=...` в send» без чтения ничего не ломает (докажет, что READ отсутствует) — но такую правку PR-0c НЕ делает (это PR-4).

---

## 7. (e) CI compile-TU — нужна ли облачная nvcc-джоба?

**НЕТ — PR-0c вводит ноль device-кода.** Раскладка покрытия:
- `core/rebuild_epoch.hpp` — pure host, компилируется g++ в CPU-тесте `Test_Rebuild_Epoch` (джоба build-test, `ctest -LE cuda`, критерий M7, зелёный БЕЗ GPU/LAMMPS). Никакого nvcc.
- `cuda/zone_nbr_store.cuh` — в PR-0c device-FREE ⇒ компилируется g++. **CI-покрытие: CPU-тест `#include "tdmd/cuda/zone_nbr_store.cuh"`** (файл чисто-C++). Это (i) держит каркас живым-компилируемым без nvcc, (ii) — **фича, не смелл: включение из g++-TU МЕХАНИЧЕСКИ форсит device-freeness**. Когда PR-1/3a добавит `__global__`/`cudaMalloc` в этот хедер, CPU-тест сломается ⇒ автор ОБЯЗАН перенести include в nvcc-TU — миграция compile-coverage co-located с приходом device-кода = правило D2 сделано механическим. Комментарий в хедере это фиксирует.
- **Поле `ZoneHeader.rebuild_epoch`** — попадает в device-сторону через `GpuHeader` (`conveyor_gpu.cuh:189`), покрыто СУЩЕСТВУЮЩЕЙ cuda-compile джобой (nvidia/cuda-контейнер, compile-only, mpi{OFF,ON}) через sizeof + локальным gpu_gate. **Новый TU не нужен** — в отличие от PR-0a, где угловые GPU-политики инстанцировали device-шаблоны и потребовали `tools/cuda_compile_check{,_tersoff}.cu`. PR-0c device-шаблонов не инстанцирует.

**Итог CI:** PR-0c — zero cloud-compile burden. nvcc-покрытие `zone_nbr_store.cuh` приходит с его первым device-потребителем (PR-1/3a), добавляющим его в cuda-compile TU (compile-break с потребителем).

---

## 8. Порядок реализации + чеклист приёмки

1. `core/transport.hpp` — вставить поле + комментарий (§2). Собрать → CUDA-suite байт-идентичен (поле инертно).
2. `core/rebuild_epoch.hpp` — примитив (§3).
3. `tests/test_rebuild_epoch.cpp` — T1-T6 (§4), включить `zone_nbr_store.cuh`.
4. `cuda/zone_nbr_store.cuh` — device-free контракт + re-export + DEFER (§5).
5. `CMakeLists.txt` — регистрация теста.
6. **Гейты:** чистый ребилд CPU N/N (+Test_Rebuild_Epoch) + CUDA M/M; 6/6 kill-мутаций краснеют свои зубы; A/B-hex пояс (eam_drift 500 vs HEAD) = 0; `git diff` 5 сайтов send/arrival = 0; memcheck/racecheck чисто.
7. **Carry-forward в PR-4-чеклист (не PR-0c):** G-EPOCH с `verlet_hybrid=ON` + `Test_MPI_Conveyor` локально; `local_epoch_` per-rank state + Tier-0/1/2 device-store строятся ВМЕСТЕ со своими потребителями (PR-1/3a/4, осознанный compile-break).

**Смета:** ~3 новых файла + 2 правки, ~120-160 строк с тестами, ноль device-кода, ноль cloud-compile. Строго аддитивный F-NOOP.

**Load-bearing источники (verbatim-сверено):** `transport.hpp:31-56` (ZoneHeader + прецедент d_full/drift_full :53-55); `skin_budget.hpp:5-9,41-65` (прецедент SSoT host-примитива + именование PR-4/ZoneNeighborStore); `conveyor_gpu.cuh:188-192` (GpuHeader), `:716-721/748-753/786-788` (arrival READ), `:1128-1138/1211-1221` (send WRITE), `:570-582/896-902` (будущий local_epoch_ дом, PR-4); `mpi_ring_edge.hpp:56-57,94,111-135` (wire memcpy/pool/poison, sizeof-производно); `AUDIT §3.5:258-286` (контракт Tier-0/1/2 + MPI lazy epoch + MUST-FIX), `§3.2:204-206` (L-SUP), `§5:341,355-356` (PR-0c/PR-4 лестница); `PR0A_WCONTRACT_DESIGN §11` (правило D2); `tests/test_skin_budget.cpp` (шаблон kill-мутаций); `CMakeLists.txt:383-385` (регистрация теста).