# PR-2 W-контракта — живое донационное кольцо EamRing: дизайн (состязательный, финальный)

**2026-07-03 · ветка `opus` · дизайн-workflow `wf_3aad7a4b` (4 читателя → 2 дизайнера → судья → 4 верификатора; все ACCEPT-WITH-FIXES — MUST-FIX вшиты ниже).**

## Вшитые MUST-FIX верификации
1. **compile-blocker (gate-mechanism):** `ZoneBlockView.key` — `const long*`, а `s.msg.id.data()` — `int*` ⇒ агрегатная инициализация не компилится. Кольцо-донации используют NullPairHook (key НЕ разыменовывается) ⇒ передавать `nullptr` для key в block_view.
2. **doc-таблица леджера (bitwise-orch):** §5 строка «pbc zone 0» — ячейки seam/interior переставлены; КОД §3.1 верен (close_cross_batch(la=lower→CrossHi, lb=upper→CrossLo); шов (n-1,0): n-1→CrossHi=4, 0→CrossLo=2, интерьер (0,1): 0→CrossHi=4 ⇒ zone0=self1+CrossLo2+CrossHi4=7). Финальный ledger 7 неизменен.
3. **G8 D7 drift-band (teeth-vacuity):** зуб недоспецифицирован — на живом кольце membership_ok бьёт ПОСЛЕ compose, а класс-3 расхождение на КРАЮ полосы суб-tol ⇒ kill-мутация вакуумна. Фикстура ОБЯЗАНА гнать read-halo атом ХОРОШО за g (не на краю) ⇒ supra-tol расхождение, ИЗМЕРЕННОЕ на живом кольце при снятом страже.
4. **fork-баннеры (fork-fnoop):** sw/tersoff/meam баннеры несут теперь-ЛОЖНОЕ «diff shows only 5 swaps» доказательство. Siblings БАЙТ-НЕТРОНУТЫ (byte-freeze выигрывает), но EAM-баннер (eam_ring.hpp:24-47) ОБЯЗАН задокументировать расхождение + пометить sibling-заявки как superseded-historical.

---

All line numbers and the decisive technical facts are confirmed against HEAD. Critically: `ZoneBlockView` (eam_donation.hpp:91) and `EamDonationState` (eam_donation.hpp:257) live in `eam_donation.hpp` — which forces the refinement-concept placement (Design B's `many_body.hpp` location cannot compile). Here is the merged final design.

---

# PR-2 W-КОНТРАКТА — ЖИВОЕ ДОНАЦИОННОЕ КОЛЬЦО EamRing (ФИНАЛЬНЫЙ ДИЗАЙН)

Слияние Designer A (minimal-surgery) и Designer B (forward-consumable). Каждое расхождение разрешено в K-таблице §1 с явным «pick + graft». Все ссылки — `file:line`, сверены с HEAD 8b2c189. Самодостаточно для имплементера.

## 0. Тезис

Взять ДОКАЗАННЫЙ сериальный донационный оракул PR-1 (`eam_donation.hpp`: `eam_donate_self/cross`, `eam_window_force_from_rho`, `EamDonationState`, леджер) и вшить его в ЖИВОЕ кольцо `eam_ring.hpp`, растворив плоский цикл `zone_eam_pass_donated_impl` (`eam_donation.hpp:488-495`) в scan-цикле кольца (`eam_ring.hpp:392-408`). Плотность ρ собирается донационными int64-батчами в scan-шаге на границе `405→406` в пер-проходный `EamDonationState<DA>` (ключ zone_id), `finalize_owned` КОМПОНУЕТ силу из персистентной ρ вместо ×3-пересчёта. Это снимает ×3-избыточность density на CPU-кольце — реальный w-выигрыш. Оракул `zone_eam_pass`/`eam_window_force` (`eam_zone.hpp:197`) и sibling-кольца (sw/tersoff/meam) + `fsm.hpp` — БАЙТ-НЕТРОНУТЫ.

## 1. K-ТАБЛИЦА (разрешение конфликтов A vs B)

| # | Вопрос | Designer A | Designer B | РЕШЕНИЕ | Обоснование |
|---|---|---|---|---|---|
| **K1** | Дом ρ-состояния | ring-local `EamDonationState<DA>` в `run_pass_impl<DA>`, ключ zone_id | policy-owned associated type `WinForce::DonationStore` + `std::variant` | **A** | Переиспользует PR-1 ДОСЛОВНО (лемма W-1), HALT-discard даром (RAII локала), без variant/associated-type-машинерии. B — спекулятивная общность ради PR-3b (gated R_W≥1.15, может NULL-откатиться как E5c-concurrency); measure-first запрещает строить под отложенный рычаг. Стоимость PR-3b названа честно (§11) |
| **K2** | fb-диспатч (44/40) | `run_pass`→`run_pass_impl<DensAccum>` на ринге | спрятан в policy-variant | **A** | Зеркалит СУЩЕСТВУЮЩИЙ паттерн `zone_eam_pass_impl<DA>`/`zone_eam_pass_donated_impl<DA>` (fb-диспатч на драйвере, `eam_donation.hpp:506-513`). Идиоматично; fb-TRAP соблюдён — диспатч от `fb_=pot.math.density_fracbits()` (`eam_ring.hpp:125`), НЕ от `accum_fracbits` |
| **K3** | Место концепта | `eam_donation.hpp` (конец namespace) | `many_body.hpp` (аддитивно) | **A — ИЗМЕРЕНО: B НЕ КОМПИЛИТСЯ** | Концепт ссылается на `ZoneBlockView` (`eam_donation.hpp:91`) и `EamDonationState` (`:257`) — их НЕТ в `many_body.hpp`, а инклуд идёт eam_donation→many_body (не обратно). B's placement технически невозможен без forward-decl. A корректен |
| **K4** | Расколлапс SPHERE | НЕ двигать `apply(SPHERE)`; ГЕЙТ на ledger-полноту | физически вынести в `try_sphere_closed` на scan-шаге | **A** | `apply(SPHERE)` (`fsm.hpp:44-48`) мутирует только `fsm.type`/`force_complete` — трасса нейтральна к месту, но перестановка рискует INV-порядком за НОЛЬ функционала (Reader 4 §E). «Реальность» SPHERE = w несёт МАТЕРИАЛЬНУЮ донированную ρ (гейт), а не место транзиции. Grafts B: FSM-аддендум фиксирует материальность |
| **K5** | ГЕЙТ «≡ старому кольцу» §12.5 | транзитивный frozen-oracle; dual-path ОМИТНУТ | temporary dual-path (runtime `bool donate_` + scan-loop `if`) | **B-ВЫБОР / A-РЕАЛИЗАЦИЯ** (см. §6) | §12.5 pick = temporary dual-path (буквальная опция, теснейшая атрибуция), но реализован POLICY-SWAP (recompute-reference политика), НЕ runtime-флагом со scan-loop-ветками (grafts A's minimal-surgery). Durable backbone = транзитив + FP64 (grafts A + оба ридера) |
| **K6** | GPU-хуки в PR-2 | inert + `compose` recompute-ит | inert + `finalize_window` делегирует `compute` | **совпадают** | `EamGpuRing` bitwise-неизменен; device-донация — PR-3b |
| **K7** | 3-й член концепта (finalize-из-ρ) | `compose` | `finalize_window` | **A-имя `compose`**, обязателен в рефайнменте | Чтобы GPU-путь PR-3b не мог обойти recompute-заглушку молча |
| **K8** | Владелец леджера | кольцо (ring) | кольцо (ring) | **совпадают** | Свободные `eam_donate_self/cross` (`:110-149`) ρ-ONLY, НЕ трогают леджер; кольцо ставит биты в `donate_position` (`closure_bit`/`close_cross_batch`) |

## 2. Где живёт ρ (ring-local `EamDonationState<DA>`, ключ zone_id)

`run_pass` (`eam_ring.hpp:197`) расщепляется на диспетчер + шаблон — ЕДИНСТВЕННЫЙ структурный рефактор, зеркалит fb-диспатч драйверов:
```cpp
bool run_pass(int k, long h, std::vector<ZoneMsg>* preload) {
  return (fb_ == 44)
      ? run_pass_impl<core::fixed::FixedAccum<44>>(k, h, preload)
      : run_pass_impl<core::fixed::FixedAccum<40>>(k, h, preload);
}
template <class DensAccum>
bool run_pass_impl(int k, long h, std::vector<ZoneMsg>* preload) { /* всё тело 197-426 */ }
```
Все лямбды (`ensure_arrival`/`ensure_drift`/`end_eam`/`finalize_owned`/`send_slot`) уже локальны и захватывают по ссылке — переезжают внутрь без изменений. Сразу после `std::vector<Slot> slot(n_)` (`:206`):
```cpp
EamDonationState<DensAccum> dstate;         // ЛОКАЛ run_pass_impl (pass-scoped)
dstate.rho.assign(std::size_t(n_), {});     // внешний размер [n_zones]; внутренний — на RECV
dstate.ledger.assign(std::size_t(n_), 0u);
```

**Reset на RECV** (SPEC(6) `eam_zone.hpp:114-116`): в `ensure_arrival`, сразу после `s.present = true` (`:237`):
```cpp
dstate.reset_zone(want_id, std::size_t(s.msg.n()));   // rho[label].assign(n_members,{}); ledger[label]=0
```
`reset_zone` (`eam_donation.hpp:267-270`) спроектирован ПОД ЭТО (комментарий «PR-2: on RECV»). `want_id` — label. **ВНИМАНИЕ (SPEC(2), `eam_zone.hpp:96-98`):** reset ρ = на RECV, но ИСПОЛНЕНИЕ self-донации = на первом дрейфе (drift ленивый, dt приезжает с arrival-0). Не путать.

**HALT-discard — автоматически:** `dstate` — локал `run_pass_impl`; любой `fail(...)`/`return false` (`:213,280,308-313,316,402,423`) или исключение (перехват `node_main:192-194`) уничтожает стек-фрейм. Частичный реплей невозможен по построению (нет API).

**Ключевание zone_id, НЕ slot (несущее для зуба G-ROT).** Под PBC `slot ≠ label`: `r=(h-1)%n` (`:201`), `label=want_id=(r+arrived)%n` (`:223`). Донации пишут `dstate.rho[label]` (`label = slot[sl].fsm.id`), finalize читает `dstate.rho[slot[wslot].fsm.id]`. **Честная оговорка measure-first (8-й рецидив паттерна SW-T3b/Te3b/Me5b-G-SORT):** для ПАСС-ЛОКАЛЬНОГО store, где каждая зона reset+donate+finalize в одном проходе, чисто-slot-keying ПОБИТОВО ИЗОМОРФНО label-keying. Дискриминирующая сила G-ROT — против MIXED-space бага (донируем по label, читаем по slot), а не против чистого slot. Production ключует по label потому что (а) SPEC(1) `eam_zone.hpp:93-95` заморозил это, (б) forward-compat с node-persistent `zone_nbr_store` (PR-0c) и device-ρ (PR-3b), (в) делает finalize-read-by-slot ДЕТЕКТИРУЕМЫМ багом. Фиксируется в комментарии зуба.

## 3. Scan-хирургия (точечно)

### 3.1 Донационный шаг — splice на `405→406`

Между `ensure_drift(j-1); ensure_drift(j); ensure_drift(j+1)` (`:405`) и `finalize_owned(j)` (`:406-407`). Тело зеркалит `zone_eam_pass_donated_impl:489-491`. **Кольцо владеет леджером** (K8); хуки пишут только ρ:
```cpp
auto block_view = [&](const Slot& s) -> potentials::ZoneBlockView {
  return { s.msg.x.data(), s.msg.y.data(), s.msg.z.data(), s.msg.id.data(), s.msg.n() };
};                                         // POST-drift координаты (донации после :405)
auto donate_position = [&](int jj) -> bool {
  const DonationBatches b = donation_layout(jj, n_, box_.periodic[2]);   // eam_zone.hpp:129
  for (int s = 0; s < b.n_self; ++s) {
    const int sl = b.self[s];
    const int label = slot[std::size_t(sl)].fsm.id;                      // LABEL keying
    const uint32_t bit = closure_bit(0, DonorRole::kSelf);
    if (dstate.ledger[std::size_t(label)] & bit)                         // exactly-once (INV-8, defensive)
      return fail(Halt::Internal, "donation: self batch twice zone " + std::to_string(label));
    winforce_.on_zone_arrival(dstate, label, block_view(slot[std::size_t(sl)]), geom);   // → eam_donate_self
    dstate.ledger[std::size_t(label)] |= bit;                            // РИНГ ставит бит (вакуумно тоже)
  }
  for (int c = 0; c < b.n_cross; ++c) {
    const int sa = b.cross[c][0], sb = b.cross[c][1];
    const int la = slot[std::size_t(sa)].fsm.id, lb = slot[std::size_t(sb)].fsm.id;
    const uint32_t hi = closure_bit(0, DonorRole::kCrossHi);
    if (dstate.ledger[std::size_t(la)] & hi)
      return fail(Halt::Internal, "donation: cross batch twice edge " + std::to_string(la));
    winforce_.on_edge(dstate, la, lb, block_view(slot[std::size_t(sa)]),
                      block_view(slot[std::size_t(sb)]), geom);          // → eam_donate_cross
    close_cross_batch(dstate.ledger[std::size_t(la)], dstate.ledger[std::size_t(lb)], 0);
  }
  return true;
};
```
Вставка (`:405-407`):
```cpp
ensure_drift(j - 1); ensure_drift(j); ensure_drift(j + 1);
if (!donate_position(j)) return false;                    // <<< НОВЫЙ scan-шаг (self→cross)
if (!(defer_head && j == 0))
  if (!finalize_owned(j)) return false;
```

**READY/DEADLINE (SPEC(3-4), WContract §9(ii-iii)).** `donation_layout(j)` эмитит ровно READY-на-скане-j батчи: interior `self(j+1)`+`cross(j,j+1)`; j=0→`self0,self1,cross(0,1)`; шов j=n−1 (pbc)→`cross(n-1,0)` (`eam_zone.hpp:136-147`). Слоты {j,j+1} present+drifted на `:404-405`; под defer_head шов на скане n−1 читает slot 0 (резидентен — не отослан до хвоста, `:409-415`) + slot n−1 (дрейфнут на `:405`). DEADLINE структурен: `donate_position(j)` ПЕРЕД `finalize_owned(j)`; шов перед `finalize_owned(n-1)` in-scan.

**EDGE-ORDER ЛОВУШКА** (`many_body.hpp:134-139`, `eam_donation.hpp:344-350` — «спот, который PR-2 сделает руками неверно»): `close_cross_batch(ledger[la], ledger[lb], 0)` — CrossHi в НИЖНЮЮ зону ребра (`la`), CrossLo в верхнюю (`lb`); шов `(n-1,0)` → CrossHi в n−1, CrossLo в 0. Численная сортировка (`lo=0`) инвертировала бы роли — само-ловится want-mask-assert'ом в §3.3, но конвенция названа намеренно.

### 3.2 `finalize_owned` — композиция из персистентной ρ (замена `:356-363`)

Gather координат окна (`:341-355`) — БЕЗ изменений. Добавить ПАРАЛЛЕЛЬНЫЙ gather ρ в ТОМ ЖЕ block-порядке `[pred][center][succ]`, затем `compose` вместо `compute`:
```cpp
const int m = int(wx.size());
std::vector<DensAccum> rho_w(m);
{ int at = 0;
  for (int t = 0; t < nw; ++t) {
    const int sl = wslots[t];
    const int label = slot[std::size_t(sl)].fsm.id;       // LABEL (зуб G-ROT: slot = баг)
    const auto& rz = dstate.rho[std::size_t(label)];
    for (int i = 0; i < slot[std::size_t(sl)].msg.n(); ++i)
      rho_w[std::size_t(at++)] = rz[std::size_t(i)];      // raw copy — bit-trivial
  } }
std::vector<core::fixed::ForceAccum> wFx(m), wFy(m), wFz(m);
winforce_.compose(wx.data(), wy.data(), wz.data(), key.data(), m,
                  ownedloc.data(), int(ownedloc.size()), geom, rho_cap, rho_w.data(),
                  wFx, wFy, wFz, pe, min_r2, j);           // было winforce_.compute(...)
return end_eam(j, ownedloc, wFx, wFy, wFz);
```
CPU `compose` = `eam_window_force_from_rho` (`eam_donation.hpp:205-252`) ДОСЛОВНО — pass-2/3 C-фазы с ρ на входе, cap ТОЛЬКО на owned-полной ρ в pass-3 (К3, `:227-230`), min_r2/PE/φ-once идентичны оракулу. **Оракул `eam_window_force` НЕ трогается** — остаётся для `serial_vv` (тесты), GPU-recompute и как frozen-референс. **Это исчезновение «старого пути»** (K5/§6): CPU-кольцо больше НЕ вызывает `eam_window_force` для density; ×3-пересчёт снят.

### 3.3 Ledger-гейт + материальный SPHERE (замена `end_eam:272-277`)

**НЕ перетасовывать FSM apply-порядок (K4).** Одна проверка в НАЧАЛЕ `end_eam` (перед `:276`) служит ОБОИМ: (а) SPHERE-materiality (расколлапс — w несёт материальную ρ), (б) ledger-before-END (O4, WContract §5/§12.4):
```cpp
auto end_eam = [&](int j, const std::vector<int>& ownedloc, ...) -> bool {
  Slot& s = slot[std::size_t(j)];
  // PR-2: density-леджер ЭТОЙ зоны закрыт (self+cross-донации исполнены) — SPHERE(d→w)
  // теперь МАТЕРИАЛЕН; и это completeness-проверка СТРОГО перед END.
  const uint32_t want = want_closure_mask(pot_.passes(), j, n_, box_.periodic[2]);  // slot j
  if (dstate.ledger[std::size_t(s.fsm.id)] != want)                                 // label
    return fail(Halt::StaleZone, "eam_ring: donation ledger " +
        std::to_string(dstate.ledger[std::size_t(s.fsm.id)]) + " != want " +
        std::to_string(want) + " (missing/late batch) step " + std::to_string(h) +
        " zone " + std::to_string(s.fsm.id));
  core::ZoneFSM::apply(s.fsm, core::ZoneEvent::SPHERE);  // d -> w (МАТЕРИАЛЬНО: ledger closed)
  core::ZoneFSM::apply(s.fsm, core::ZoneEvent::START);   // w -> c
  ... /* :278-317 БЕЗ изменений */
  core::ZoneFSM::apply(s.fsm, core::ZoneEvent::END);     // :318 c -> p
```
**Разграничение (не смешать, `eam_zone.hpp:160-164`):** `want_closure_mask(j)` — бухгалтерия САМОЙ зоны j (её self/lo/hi), НЕ планировочный want-set окна. `want` использует **slot j** (free-z edge-drop: `has_lo=cyc||j>0`, `has_hi=cyc||j+1<n`, `:171-172`; под PBC `cyc=true` ⇒ роли не дропаются), `ledger` — **label** (`s.fsm.id`) ⇒ отображение slot→label под ротационным зубом. Под defer_head `finalize_owned(0)` в хвосте (`:412`) идёт через `end_eam(0)` ⇒ проверка покрывает голову. Комментарий `:273-275` («EVERY zone's d→w SPHERE is artificial») переписать: теперь материален. **DEFER:** физический перенос `apply(SPHERE)` на донационный edge — косметика, риск INV-порядка, ноль функционала.

## 4. Концепт-хуки (O1) — рефайнмент в `eam_donation.hpp`

**Базовый `WindowForcePolicy` (`many_body.hpp:153-163`) — БАЙТ-НЕТРОНУТ.** Sibling-кольца static_assert'ят ИМЕННО его (`sw_ring.hpp:146`, `tersoff_ring.hpp:183`, `meam_ring.hpp:185`) ⇒ `SwWinForce`/`TersoffWinForce`/`MeamWinForce` целы. Это КРУКС совместимости O1 с «siblings byte-untouched»: **рефайнмент, не расширение базы.**

Новый концепт — в `eam_donation.hpp` (конец namespace; K3 — только там видны `ZoneBlockView`/`EamDonationState`; base-концепт в `many_body.hpp` их НЕ видит, инклуд-цикл):
```cpp
template <typename WF>
concept DonatingWindowForcePolicy =
    WindowForcePolicy<WF> &&
    requires(const WF wf, EamDonationState<core::fixed::FixedAccum<44>>& st, int label,
             const ZoneBlockView& blk, const core::PairGeom& geom,
             const double* d, const long* k, int i, const int* ip, double rc,
             const core::fixed::FixedAccum<44>* rho_w,
             std::vector<core::fixed::ForceAccum>& f,
             core::fixed::EnergyAccum& pe, double& mr) {
      { wf.on_zone_arrival(st, label, blk, geom) } -> std::same_as<void>;   // self
      { wf.on_edge(st, label, label, blk, blk, geom) } -> std::same_as<void>;  // cross
      { wf.compose(d,d,d,k,i,ip,i, geom, rc, rho_w, f,f,f, pe, mr, i) } -> std::same_as<void>;
    };
```
Концепт проверяет `FixedAccum<44>`-инстанциацию; методы — template на `DensAccum`, инстанциация 40 компилится на call-сайте. `eam_ring.hpp:100` апгрейдится: `static_assert(potentials::DonatingWindowForcePolicy<WinForce>, "...PR-2 donation hooks...")`. HARD-требование (compile-break, НЕ `if constexpr requires`) ⇒ три свойства §12.1 (mandatory/compile-enforced/не-opt-in). **Честное отклонение от буквы «в WindowForcePolicy»:** хуки — в РЕФАЙНМЕНТЕ, а не базе (audit-D2 «хуки входят вместе с потребителем»); вынуждено constraint-b + инклуд-циклом. `compose` — 3-й обязательный член (K7): GPU-путь PR-3b не обойдёт recompute-заглушку.

`CpuEamWindowForce` (`eam_ring.hpp:66-93`) получает три ЖИВЫХ template-метода (тонкие обёртки над PR-1):
```cpp
template <class DA> void on_zone_arrival(EamDonationState<DA>& st, int label,
    const ZoneBlockView& blk, const core::PairGeom& geom) const {
  eam_donate_self<Math, DA>(blk, *math, geom, st.rho[std::size_t(label)]);          // :110
}
template <class DA> void on_edge(EamDonationState<DA>& st, int la, int lb,
    const ZoneBlockView& a, const ZoneBlockView& b, const core::PairGeom& geom) const {
  eam_donate_cross<Math, DA>(a, b, *math, geom, st.rho[std::size_t(la)], st.rho[std::size_t(lb)]);  // :130
}
template <class DA> void compose(const double* wx,...,const DA* rho_w,...,int zone_j) const {
  eam_window_force_from_rho<Math, DA>(wx,...,rho_w,...,NullDonationTrace{}, zone_j);  // :205
}
```
Хуки ЖИВЫЕ (вызываются рингом), НЕ структурно-мёртвые ⇒ НЕ 5-й рецидив. `assert_supported` (`:90-92`), `compute` (`:71-83`) — без изменений (компьютер остаётся для базы; после PR-2 CPU-кольцо его не зовёт для density). `#include "tdmd/potentials/eam_donation.hpp"` в `eam_ring.hpp` (циклов нет).

### 4.1 GPU-политика — инертные хуки, `EamGpuRing` побитово (K6)

`GpuEamWindowForce` (`include/tdmd/cuda/eam_window_force_gpu.cuh:231`) ОБЯЗАНА моделировать рефайнмент (`cuda_compile_check.cu:42,49` инстанцирует `EamRing<double,EamSetfl<double>,GpuEamWindowForce>::run()`). Добавить (+`#include eam_donation.hpp`; CUDA-хедер, НЕ sibling-кольцо ⇒ разрешено):
```cpp
template <class DA> void on_zone_arrival(EamDonationState<DA>&, int, const ZoneBlockView&,
                                         const core::PairGeom&) const {}
template <class DA> void on_edge(EamDonationState<DA>&, int, int, const ZoneBlockView&,
                                 const ZoneBlockView&, const core::PairGeom&) const {}
// PR-2: GPU-кольцо ПЕРЕСЧИТЫВАЕТ density per-window (bitwise ≡ сегодня). Device-резидентная
// донация — PR-3b (гейт R_W≥1.15, NULL-откат). rho_w ИГНОРИРУЕТСЯ.
template <class DA> void compose(const double* wx,...,const DA* /*rho_w*/,...,int /*zone_j*/) const {
  compute(wx, ..., pe, min_r2);   // существующий E5 device-recompute
}
```
`cuda_compile_check.cu:42` апгрейдить: `static_assert(tp::DonatingWindowForcePolicy<tc::GpuEamWindowForce>)`. **Несущее следствие:** ринг для GPU-политики (а) аллоцирует `dstate` + зовёт `reset_zone` (безвредно, host-вектора), (б) зовёт no-op хуки, (в) РИНГ ставит ledger-биты независимо от политики ⇒ END не голодает, (г) `compose` ИГНОРИРУЕТ `rho_w` (весь ноль) и recompute-ит на device ⇒ `EamGpuRing` output == сегодняшний. `Test_CUDA_EAM_Ring` остаётся побитово-зелёным; PR-2 обязан прогнать `scripts/gpu_gate.sh build-cuda` (memcheck+racecheck). Одна кодовая ветка (`run_pass_impl`), без `if constexpr`. Хуки не мертвы (ledger живой — им управляет ринг; ρ отложена).

## 5. Сверка бит-леджера (проверено вручную)

`closure_bit(0,kSelf)=1, kCrossLo=2, kCrossHi=4`. `close_cross_batch(lo,hi)`: lower-зона ребра |=CrossHi(4), upper |=CrossLo(2).

| зона | self | cross(j−1,j)→j | cross(j,j+1)→j | ledger | want_closure_mask |
|---|---|---|---|---|---|
| interior (free) | 1 | CrossLo=2 | CrossHi=4 | 7 | self\|lo\|hi=7 ✓ |
| j=0 (free) | 1 | — | 4 | 5 | self\|hi=5 (lo dropped) ✓ |
| j=n−1 (free) | 1 | 2 | — | 3 | self\|lo=3 (hi dropped) ✓ |
| pbc zone n−1 | 1 | 2 (ребро n−2,n−1) | 4 (шов n−1,0) | 7 | 7 ✓ |
| pbc zone 0 | 1 | 4 (шов, zone 0 = upper) | 2 (ребро 0,1) | 7 | 7 ✓ |

Шов `cross(n-1,0)`: `close_cross_batch(ledger[n-1], ledger[0], 0)` ⇒ zone n−1 (lower)|=CrossHi(4), zone 0 (upper)|=CrossLo(2). Роль-инверсия само-ловится want-mask-assert'ом §3.3.

## 6. ГЕЙТ «≡ старому кольцу побитово» (§12.5) — ГЛАВНОЕ РЕШЕНИЕ

**ВЫБОР (K5): §12.5-механизм = ВРЕМЕННЫЙ DUAL-PATH (опция b), реализованный POLICY-SWAP; durable backbone = транзитив + FP64.** Явно отвергаю: (i) закоммиченный FP64-блоб — оба ридера и оба дизайнера отвергли (кросс-платформенно ХРУПОК: CLAUDE.md фиксирует квант-флипы exp ⇒ ляпуновская дивергенция аналит-EAM/Морзе между компиляторами/арками/`--fmad`; чужероден репо — `reference_data/` держит только ТОЛЕРАНТНЫЕ LAMMPS-golden); (ii) B's runtime `bool donate_` со scan-loop `if(donate_)`-ветками — критика Designer A: хирургия в горячем цикле, второй diff на удаление; (iii) A's «омитнуть dual-path» — критика Reader 3: транзитив НЕ изолирует density-swap от orchestration-drift.

### DURABLE (в CI навсегда, house-rule standing gate)
**L1 — транзитивная эквивалентность через байт-замороженный оракул.** 14 существующих `Test_EAM_Ring`/`Test_EAM_RingPBC` (`test_eam_ring.cpp`, тела НЕ править) сравнивают `run_eam_ring` (после PR-2 — донационный путь дефолтной `CpuEamWindowForce`) с `serial_vv[zone_eam_pass]` (`:57-75`) побитово (`state_bitwise_equal:77`, free+PBC, fixed+auto, 150-шаг реплики). Цепочка: `donated_ring ≡ serial_vv[zone_eam_pass]` (kept-green) ∧ `zone_eam_pass` байт-заморожен НАВСЕГДА (`eam_zone.hpp:197`) ∧ `old_ring ≡ serial_vv[zone_eam_pass]` (истинно на HEAD) ⇒ **`donated_ring ≡ old_ring` транзитивно, кросс-платформенно, БЕЗ артефакта в репо.** Это §12.5-«frozen reference trajectories» ЧЕСТНО реализовано: замороженный артефакт = ИСХОДНИК оракула, вычисляемый вживую (строго сильнее коммит-блоба; идиоматично — все ring-тесты пересчитывают оракул вживую).

**L2 (НЕПЕРЕУСТУПАЕМО) — независимый FP64-оракул выпавшего донора.** L1/1-vs-z/run-to-run/dual-path СЛЕПЫ к паре, молча выпавшей ВНУТРИ батча (WContract §3, честная граница леджера). Новый ring-зуб G-ORACLE-RING: донационная траектория within-tol от `eam_direct_fp64`-driven VV (`eam.hpp:41`; не делит window/zone/donation-логику). Обязателен (дом-правило MB2).

### §12.5-ЛИТЕРАЛ — temporary dual-path через POLICY-SWAP (исчезает тем же PR)
Test-only `CpuEamRecomputeWinForce` — структурный CPU-АНАЛОГ GPU-политики PR-2 (§4.1): хуки no-op, `compose` ИГНОРИРУЕТ `rho_w` и зовёт замороженный `eam_window_force` (recompute). Гейт (в тесте): та же фикстура, ОДНА оркестровка, две политики через СУЩЕСТВУЮЩИЙ policy-injection overload `run_eam_ring(...,winforce)` (`eam_ring.hpp:524`): donation-ring (дефолт, compose-из-ρ) vs recompute-ring (`CpuEamRecomputeWinForce`, recompute), `state_bitwise_equal` + пер-проходный `PassStats`.

**Почему это ИЗОЛИРУЕТ ровно нужное:** `donate_position`/ledger/reset_zone/SPHERE-gate — в РИНГЕ (policy-agnostic) ⇒ гоняются в ОБЕИХ политиках идентично; ЕДИНСТВЕННАЯ разница — compose-из-ρ vs recompute. Баг в `donate_position` (коррупция ρ) НЕ маскируется: donation-ring юзает корруптнутую ρ, recompute-ring её игнорирует и пересчитывает верную ⇒ дивергенция ⇒ dual-path ЛОВИТ. Это ровно сердце PR-2 (G-A на живом кольце), на ИДЕНТИЧНОМ drift/arrival/Λ-timing. **«Старый путь исчезает тем же PR»:** production-finalize кольца больше НЕ recompute-ит density (`compose`-из-ρ). Recompute остаётся ТОЛЬКО как (а) frozen `eam_window_force`-оракул, (б) test-only recompute-reference политика (= полезный CPU-аналог GPU-политики + durable A/B).

**Graft:** беру B's ВЫБОР (dual-path — буквальная §12.5-опция, теснейшая атрибуция), A's РЕАЛИЗАЦИЮ (policy-swap, не scan-loop-флаг — minimal-surgery), A's транзитив как durable-слой, обоих ридеров FP64-непереуступаемость.

## 7. Тест-план (каждый зуб — kill-мутация; слияние A×B)

14 существующих — тела НЕ править, остаются зелёными = L1 backbone. Новые зубы в `test_eam_ring.cpp` (переиспользуя `make_fcc:28`/`make_fcc_pbc:241`/`serial_vv:57`/`ring_opts:86`/`state_bitwise_equal:77`/`total_momentum:246`/`VacuumGapEmptyZones:206`):

| # | Зуб | Фикстура | Проверка | Kill-мутация |
|---|---|---|---|---|
| **G1** | dual-path bitwise (§6) | free n=6 + PBC n=6, fixed+auto, 20 шагов | donation-ring ≡ recompute-ring (`state_bitwise_equal`+PassStats) | off-by-one в потреблении `donation_layout` → RED |
| **G2** | 14 существующих зелены | как есть | L1 транзитив; тела не тронуты | правка тела = сигнал регрессии оркестровки |
| **G3** | 1-vs-z донационный | free{2,3,5}+PBC{2,3,6}, fixed+auto | z ≡ z=1 побитово | ключ store по slot → RED (пересекается G5) |
| **G4** ⭐ | G-ORACLE-RING (L2, непереуступаемо) | free n=6 + PBC n=5 (seam), 8 шагов | траектория within-tol от `eam_direct_fp64` (`eam.hpp:41`) | poison-политика (G10) → >1e-3 |
| **G5** ⭐ | G-ROT (≥2 оборота, O2) | PBC n=5, атомы в rcut ШВА обоих концов, ≥10 шагов, fixed+auto | донац. ≡ z=1 ≡ serial cyclic + `eam_direct_fp64` на шве | в finalize читать `dstate.rho[wslot]` (slot) вместо `[label]` → под ротацией r≠0 читается ρ ЧУЖОЙ зоны → RED. Комментарий: slot-keying изоморфен для pass-local; зуб бьёт MIXED-space |
| **G6** | G-VACUUM (D8, O3) | `VacuumGapEmptyZones:206` (зоны 1..4 пусты) | донац. ≡ оракул, `halt==None`; пустая зона: `eam_donate_self` на n=0 (0 пар) + РИНГ ставит self-бит вакуумно | не ставить бит на пустом батче → ledger-before-END HALT (StaleZone) |
| **G7** ⭐ | G-HALT-ATTR (rho_cap, O4) | `from_analytic(m,tight=rho_max*0.5)` | донац. HALT `Halt::Internal` (`compose`→`:227` throw→`node_main:193`) с «zone» в msg; `loose=1.5` → None+≡оракул | отключить owned-cap `:227` → tight не HALT'ит |
| **G8** ⭐ | G-D7-DRIFT-BAND (O4) | атом за `g=0.5(w−2rc)` (`membership_ok:462`) за N шагов | кольцо HALT'ит `Halt::StaleZone` (`:315`); in-band ≡ оракул. `g_override` НЕ прокидывается (O6) — полоса задаётся геометрией z_d | снять `membership_ok` `:315` → band-атом не HALT'ит, G-A краснеет vs FP64 |
| **G9** | G-POISON-HALFBUG | test-only `CpuEamPoisonWinForce` (`one_sided=true` в `eam_donate_cross`, `:134`) через policy-overload | G-A red vs serial И vs `eam_direct_fp64` (ledger ЧИСТ — ринг ставит биты — но физика красная) | это и есть мутация; позитив = G4 зелёный на дефолте |
| **G10** | G-LEDGER-DROP / LATE-SEAM | hand-verified: пропустить `donate_position` / сдвинуть шов после finalize(n−1) | `ledger != want` на `end_eam` → HALT (зеркало Т-5/Т-6) | ревьюер ре-мутирует |
| **G11** | fb40 (Q23.40) fb-TRAP (Т-12) | `test_eam(beta=3.3)`⇒fb=40 | донац. z=3≡z=1 (`DualFormatQ23OneVsZ:159`-стиль) | диспатч от статич. `accum_fracbits` → RED |
| **G12** | anti-deadlock донационный | `AntiDeadlock:375` | free z=1..6/PBC z=1..5, `steps==2z+3`, `Halt::None` | liveness-регрессия = RED сама |

**Инвариантные (зелены без правки):** `MomentumConservation:290`, `NveEnergyConservation:339`, `LongRunReplicaBitwise:175/305`. **GPU:** `Test_CUDA_EAM_Ring` 12 гейтов побитово-неизменны (recompute-compose); `gpu_gate.sh` чист.

## 8. Инварианты, которые хирургия НЕ ломает

1. **Λ-цепочка dt-handoff (B2):** `dt` из arrival-0 (`:238`), `R_buf` из `v_pred`/lag=n−1 (`:239-244`), Λ в `send_slot` (`:376-380`). Донации density-only, не трогают dt/R_buf/Λ/скорости.
2. **INV-4:** `causality_ok` (`:308-314`) — дословно.
3. **StaleZone/membership_ok** (`:315-317,460-482`, `g=0.5(w−2rc)`): L-CLOSE-предпосылка САМОЙ эквивалентности (класс-3, `eam_donation.hpp:50-54`); `g_override` в кольцо НЕ прокидывать (O6, §12.6) — кольцо берёт g inline из `:462`.
4. **Anti-deadlock:** ёмкость `n_+3` (`:154`), задержанный send `j-2` (`:397`) — донации НЕ вводят блокирующих recv/send, НЕ переносят send раньше (иначе донация прочла бы moved-from msg, `:372`).
5. **defer_head PBC-хвост** (`:409-415`): порядок finalize(0)-first + sends не меняется; SPHERE(0) на скане шва n−1.
6. **§7.4 parity** `node_io_order(k+1)` (`:200,393`) — не менять.
7. **B1/INV-9:** ρ int64 order-free (W-1); fb-диспатч ТОЛЬКО от `density_fracbits()`.
8. **PE/min_r2/φ-once** — C-фаза дословно (`eam_window_force_from_rho`=`eam_window_force` symbol-for-symbol).
9. **HALT-семантика** (`set_halt`/`halt_on_`, `:484-491`) — не менять; `dstate` pass-scoped ⇒ mid-pass HALT отбрасывает проход.
10. **Дескриптор:** `validate_pass_decls`+`assert_supported` (`:137-138`) UNCONDITIONAL; добавить `assert_eam_donation_descriptor(pot_.passes())` (`eam_donation.hpp:70`) в `run()` рядом (проверяет D5-флип `eam.hpp:114-117`).

## 9. Fork-дисциплина (comment/doc, код siblings НЕ трогать)

`run_pass_impl` расходится с sw/tersoff/meam-кольцами НЕОБРАТИМО (аудит §5/§7.7 предвидел — унификация форков до ReaxFF, форк №5). Правки ТОЛЬКО в баннере `eam_ring.hpp:24-47` (append): с PR-2 EAM scan-loop несёт донационный W-путь и расходится с siblings; siblings — форки PRE-PR-2 baseline (git-восстановимый коммит), байт-нетронуты; будущие orchestration-фиксы портируются вручную; протухшую sibling-заявку «diff shows only 5 swaps» (`sw_ring.hpp:30-35`) НЕ править (byte-freeze) — она остаётся ИСТОРИЧЕСКОЙ провенанс-заявкой. Рефайнмент-концепт АДДИТИВЕН ⇒ `SwWinForce`/`TersoffWinForce`/`MeamWinForce` моделируют БАЗУ, байт-целы. `fsm.hpp` — байт-нетронут.

## 10. FSM-аддендум (append-only, `fsm.hpp` нетронут)

Существующий §12 (`ZoneFSM_v1_0.md:240+`) УЖЕ покрывает ARRIVAL-семантику (PR-1 написал forward-looking). Добавить п.6 (append-only, таблица §4 + `fsm.hpp` неизменны):
> **6. Реализация PR-2 (кольцо, `eam_ring.hpp`):** (i) `apply(SPHERE)` d→w сохранён на месте (`end_eam`), но ГЕЙТИТСЯ завершённостью density-леджера зоны (`ledger[label]==want_closure_mask`) — материальная w = донационные батчи, наполнившие ρ зоны в её d/w; артифициальность снята. (ii) Completeness-ledger проверяется непосредственно перед `apply(SPHERE)`/`apply(END)`. (iii) Донор-в-d/w теперь ИСПОЛНЯЕТСЯ (`donate_position` на scan-шаге 405→406). Λ-цепочка §6, defer_head §7.2, INV-3/4, чёт/нечёт §7.4 — нетронуты.

Плюс пометка «§12 приземлено PR-2» в шапке `TD_MD_Core_WContract_v1_0.md`; летопись в CLAUDE.md.

## 11. DEFER (честный scope)

- **GPU device-resident донация** (PR-3b, R_W≥1.15, NULL-откат): GPU-хуки инертны в PR-2. **ЧЕСТНАЯ СТОИМОСТЬ (graft B's callout):** т.к. store — ring-local `EamDonationState<DA>` (host, K1), PR-3b — ЕСЛИ материализует device-резидентную ρ (после измерения R_W≥1.15) — ПЕРЕ-ТРОНЕТ сигнатуры хуков (`EamDonationState<DA>&`→device-store) + `CpuEamWindowForce` + концепт. Это ЯВНАЯ, принятая вторая fork-дивергенция, НЕ скрытая. Measure-first: платим (малую) цену PR-3b ТОГДА, а не спекулятивной associated-type-машинерией (variant/`DonationStore`) СЕЙЧАС — под рычаг, который на single-GPU host-orchestrated кольце может NULL-нуть (прецедент E5c-concurrency; device-D2D — ценность M5b/multi-GPU, WContract §11).
- **Физический перенос `apply(SPHERE)`** на донационный edge — косметика, риск INV-порядка, ноль функционала.
- **Node-persistent / cross-pass ρ** — не нужно (pass-scoped); label-keying оставлен forward-compat с PR-0c `zone_nbr_store`.
- **Fork-унификация** siblings — ReaxFF (форк №5).
- **Хуки в БАЗОВЫЙ `WindowForcePolicy`** (буква §12.1) — неисполнимо (ломает siblings + инклуд-цикл K3); рефайнмент — санкционированное состязательным дизайном отклонение.

## 12. Сводка точек правки (для имплементера)

- `eam_ring.hpp`: +`#include eam_donation.hpp`; `:100` static_assert→`DonatingWindowForcePolicy`; `CpuEamWindowForce`(`:66`) +3 template-хука (`on_zone_arrival`/`on_edge`/`compose`); `:197` split `run_pass`→`run_pass_impl<DA>` (fb-диспатч); top `run_pass_impl` +`dstate` sizing; `ensure_arrival:237` +`reset_zone`; scan `:405→406` +`donate_position`; `finalize_owned:356-363` +rho_w gather + `compose`; `end_eam` top (перед `:276`) +ledger-check/SPHERE-gate; `run():137` +`assert_eam_donation_descriptor`; баннер `:24-47`.
- `eam_donation.hpp`: +концепт `DonatingWindowForcePolicy` (аддитивно, конец namespace).
- `include/tdmd/cuda/eam_window_force_gpu.cuh`: `GpuEamWindowForce`(`:231`) +3 инертных хука (+`#include eam_donation.hpp`).
- `tools/cuda_compile_check.cu:42`: static_assert→`DonatingWindowForcePolicy`.
- `tests/test_eam_ring.cpp`: +12 зубов (§7) + `CpuEamRecomputeWinForce`/`CpuEamPoisonWinForce` (test-only); 14 существующих — тела неизменны.
- doc: `ZoneFSM_v1_0.md` §12 п.6 (append), `WContract_v1_0.md` шапка, `CLAUDE.md` летопись.

**Гейт приёмки:** чистый ребилд → CPU 43/43 (`Test_EAM_Ring` расширен, `Test_EAM_Donation` 15/15 неизменны, siblings байт-целы) + CUDA 17/17 (`Test_CUDA_EAM_Ring` побитово) + `gpu_gate.sh` чист + все kill-мутации красят свои зубы + G4/G5/G7/G8 подтверждены FP64-свидетелем.

**Абсолютные пути:** `/home/slava8519/td-md-core/include/tdmd/potentials/eam_ring.hpp`, `/home/slava8519/td-md-core/include/tdmd/potentials/eam_donation.hpp`, `/home/slava8519/td-md-core/include/tdmd/potentials/eam_zone.hpp`, `/home/slava8519/td-md-core/include/tdmd/potentials/many_body.hpp`, `/home/slava8519/td-md-core/include/tdmd/cuda/eam_window_force_gpu.cuh`, `/home/slava8519/td-md-core/tools/cuda_compile_check.cu`, `/home/slava8519/td-md-core/tests/test_eam_ring.cpp`, `/home/slava8519/td-md-core/docs/TD_MD_Core_ZoneFSM_v1_0.md`, `/home/slava8519/td-md-core/docs/TD_MD_Core_WContract_v1_0.md`.