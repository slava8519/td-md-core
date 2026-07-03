# PR-1 — Serial Donation Oracle для EAM: дизайн (состязательный, финальный)

**2026-07-03 · ветка `opus` · дизайн-workflow `wf_57f16ac5-a90` (4 читателя → 2 дизайнера → судья → 4 верификатора; все ACCEPT-WITH-FIXES — MUST-FIX вшиты в текст ниже).**

## Вшитые MUST-FIX верификации

1. **Полоса дрейфа ПЕРЕВЁРНУТА ИЗМЕРЕНИЕМ (bitwise-math):** деривация слияния упустила класс-3
   расхождение (уже-донированное вне-оконное ребро читаемого halo — открывается ровно при D>g_eam)
   ⇒ на полосе (g_eam, g_pair] G-A КРАСНЫЙ (пробник: |ΔF|=1.24 эВ/Å), контроль no-drift — ПОБИТОВО
   зелёный. D7b-зуб дизайнера A реинстейтирован; g_eam несущий для самой эквивалентности. §8/Т-11/§12
   переписаны по измерению.
2. **Т-13 (teeth-vacuity + implementability):** EamPotential `final` + захардвайренный passes() ⇒
   донационное требование вынесено в свободную `assert_eam_donation_descriptor(span)` — ctor зовёт,
   зуб зовёт напрямую с kCOnly-копией.
3. **§4 (pr2-consumability):** «тестов, триггерящих cap, НЕТ» — ложь (EamSpline.DensityBeyondGridHalts
   кроет eam_direct_fp64); Т-10а пере-специфицирован на пришпиливание именно zone_eam_pass.

SHOULD применены: cap-фикстура rho_max=5.0 (прецедент), fb40 через beta=3.3, PE-не-дрейф-зуб,
якоря цитат (:104-108 и др. — сверять по HEAD при реализации), замок T-GAP переживает флип (проверено).

---

Anchors verified (donation_layout SPEC :86-118, `close_cross_batch` edge-order semantics at many_body.hpp:134-139, T-INERT :92-108, T9 :303-314, pass bodies :202-246, g-formula eam_ring.hpp:462, guard call :315). One critical merge-relevant discovery confirmed: `close_cross_batch` marks CrossHi in the **lower zone of the edge** (not numeric-lower), and eam_zone.hpp:116-118 confirms pbc n=1 degenerates to the free path while `eam_window_layout` silently duplicates slots — both designs missed the dedup obligation. Producing the merged design now.

# ДИЗАЙН PR-1 — SERIAL DONATION ORACLE для EAM (W-лестница, шаг 2) — ОБЪЕДИНЁННЫЙ ФИНАЛ

**Файл назначения:** `docs/_meta/PR1_DONATION_ORACLE_DESIGN_2026-07-02.md`. Слияние двух состязательных дизайнов (A: minimal-diff; B: PR-2-consumability). База: HEAD `738893f` (PR-0a). Самодостаточен — аудит (`docs/_meta/AUDIT_W_PHASE_NEIGHBORS_2026-07-02.md`) и PR-0a-дизайн перечитывать не требуется. Все номера строк сверены по HEAD.

**Оракул навсегда:** `zone_eam_pass` / `zone_eam_pass_impl` / `eam_window_force` (include/tdmd/potentials/eam_zone.hpp:185-320) — кодовые строки байт-нетронуты (единственное исключение — COMMENT-ONLY правка §5.3, К2). Все кольца (`eam_ring.hpp`, sw/tersoff/meam), `core/fsm.hpp`, весь GPU-код — байт-нетронуты. GPU-кода в PR-1 нет.

---

## §К. Таблица конфликтов A↔B (trade-off → выбор → почему)

| # | Вопрос | A (minimal-diff) | B (PR-2-consumability) | РЕШЕНИЕ |
|---|---|---|---|---|
| **К1** | Флип шипнутого `kEamPassDecls[0]` → kAccumB1+roles (D5) | DEFER в PR-2: драйвер требует W-дескриптор, тест подаёт локальную флипнутую копию; замки T-INERT/T9 не трогаются | Флип в PR-1 + осознанные ретаргеты замков | **B.** (i) Первый потребитель дескриптора — донационный драйвер и его ledger-assert — приезжает в PR-1; правило «флип вместе с потребителем» (тот же мотив, что D2 для хуков). (ii) Тест-локальная копия A создаёт риск дрейфа «копия vs будущий шипнутый флип» и оставляет EAM-маски `want_closure_mask` непришпиленными к ШИПНУТОМУ дескриптору. (iii) Легальность вшита в сам код PR-0a: «flipping EAM Density → kAccumB1 in PR-1 breaks no consumer» (many_body.hpp:44-48); право ретаргета замков зарезервировано PR-0a D5. (iv) PR-2 (тредовый, рискованный) остаётся чисто-оркестровочным. **Грефт из A:** зуб «kCOnly-копия дескриптора ⇒ ctor-throw» = страж ОТКАТА флипа (§9 Т-13). |
| **К2** | Правка `eam_zone.hpp` | COMMENT-ONLY :105-107 («PR-1 picks it» → «РЕШЕНО: arrival») | zero-touch целиком | **A.** После PR-1 комментарий «PR-1's adversarial design picks it» (eam_zone.hpp:105-107) — ЛОЖНЫЙ; дом чинит ложные комментарии COMMENT-ONLY (прецеденты SW T3b Gap-A, Te3 MUST-FIX). Приёмка: diff `eam_zone.hpp` содержит ТОЛЬКО замену этих комментарий-строк, ни одной кодовой. |
| **К3** | Место rho_cap-проверки в донированном finalize | pass-3, owned-цикл, перед `eval_F(rho[ii])` (:227) | pass-2 с параметром `is_owned[m]` | **A.** Trigger-set идентичен (owned-полная ρ), но без лишнего параметра — pass-3 уже итерирует ровно owned; копия ближе к оракулу. Из B берём Trace-параметр (нужен G-LCLOSE, §8) — он `is_owned` не требует: тест классифицирует читаемые атомы по `zd` сам. |
| **К4** | D7-зуб membership/g — **решающий пункт слияния** | D7b: `skip_residence_guard` + дрейф δ∈(g, g_pair] ⇒ «G-A побитово краснеет» | Геометрическая деривация: на полосе (g_eam, g_pair] донированный ≡ окну ПОБИТОВО (оба слепы одинаково) ⇒ D7b структурно мёртв; несущее свойство g_eam — бит-финальность ЧИТАЕМЫХ значений ⇒ G-LCLOSE-trace; G-A краснеет только при D>g_pair | **B.** A-шный D7b — кандидат в 6-й рецидив structurally-dead: окно {j−1,j,j+1} и донации self(j±0/1)+cross видят ОДНИ И ТЕ ЖЕ три зоны ⇒ owned-ρ совпадает при любом дрейфе; читаемые halo расходятся лишь через пары (j−1,j+1), возможные только при D>g_pair=0.5(w−rcut). Полный вывод — §8. Аудит-буква «ослабленный g ⇒ гейт обязан упасть» выполняется ДВУМЯ зубами на ДВУХ полосах дрейфа: G-LCLOSE красный на (g_eam, g_pair] (контракт бит-финальности чтений — то, что реально несёт g_eam для W-1-свободы PR-2/M5b), G-A сам красный при D>g_pair (мутант g=∞). Приёмка ОБЯЗАНА ИЗМЕРИТЬ предсказание «G-A зелёный на полосе» (measure-first; прецеденты пере-вердиктов Te3 ζ-order / SW-T3b глоб-key / Me5b G-SORT). **Грефт из A:** D7c «δ≤0.9·g ⇒ страж молчит И G-A побитово держится» (страж не пере-затянут). |
| **К5** | Форма API | free-функции + `DonationPoison{drop_self, drop_cross, late_seam, …}` в драйвере | пошаговый класс `EamDonationPass` (donate_self/donate_cross/finalize/finish) + канонический драйвер | **B + грефт A.** Пошаговый класс = точки врезки PR-2 один-в-один (self на drift-фронте eam_ring.hpp:250/:405, cross на CoRes :215-248, finalize внутри finalize_owned :324-365); POISON drop/late-seam выражаются ЕСТЕСТВЕННО (не вызвать батч / вызвать шов поздно) — без прод-фильтров. Из A остаётся МАЛЫЙ тест-only `DonationPoison{skip_first_pair, one_sided, ledger_off}` для интра-батчевых зубов, к которым леджер слеп (честная граница) — прецедент poison-параметров в оракулах (tersoff drop_class, MEAM poison). `g_override` — из B. |
| **К6** | Гейтить ли атрибуцию rho_cap-HALT | нет (обе стороны — исключение из того же прохода; тест слеп) | да, «зона-владелец» у нового пути | **Слияние.** Различие атрибуции старый-vs-новый — легитимно, НЕ гейтится (A). Но атрибуция НОВОГО пути детерминирована по построению ⇒ гейтится его сообщение «owner-зона» (B) — это то, что дословно наследует PR-2 (throw → `set_halt(Halt::Internal)`, eam_ring.hpp:192-194). Грефт из A: кейс near-cap-below (НИ ОДИН не бросает). |
| **К7** | Порядок окна в донированном finalize | view в духе сортированного serial-окна | блочный [pred][center][succ] (порядок кольца, БЕЗ сортировки) | **B.** G-A тогда ЗАОДНО сертифицирует блочный порядок против сортированного оракула (int64-order-freedom B1, явно) — PR-2 не пере-доказывает. **НОВОЕ (найдено слиянием, ни A ни B не покрыли):** обязателен дедуп слотов по zone_id — при pbc n=1 `eam_window_layout` эмитит дубликаты (гард-комментарий eam_zone.hpp:116-118: «eam_window_layout silently duplicates slots»); у оракула дедуп делает `sort+unique` (eam_zone.hpp:55-56, несущий ТОЛЬКО при pbc n=1). G-A включает pbc n=1. |

**Прочие грефты из A** (в B отсутствовали): G-ORDER — исполнимый W-1-свидетель (§9 Т-9); Q23.40 анти-вакуумность `EXPECT_NE` квантов (Т-12); порядок «пришпилить оракул rho_cap ДО гейта эквивалентности» (Т-10); условие n≥3 у residence-стража (зеркало eam_ring.hpp:315); тексты ошибок. **Из B** (в A отсутствовали): `reset_zone` пер-зонный; `ZoneBlockView`-спаны; ручная сверка скан-vs-want; порядок работ; G-LCLOSE. **Новое от слияния:** дедуп pbc n=1 (К7); явное предупреждение о edge-порядке аргументов `close_cross_batch` (§3.5).

---

## §1. Резюме и скоуп

PR-1 строит **сериальный донационный драйвер плотности EAM**: ρ собирается донационными батчами по `donation_layout(j,n,pbc)` (eam_zone.hpp:126-146) — self(k) intra-zone пары + cross(k,k+1) граничные — в **персистентные пер-зонные int64 `DensAccum`-массивы, ключёванные по zone_id**, вместо пооконного 3-зонного пересчёта; и доказывает **ПОБИТОВОЕ ≡ `zone_eam_pass`** на домене G-A. Force/PE/min_r2/φ-once — C-фаза дословно: донации двигают ТОЛЬКО density-пасс (SPEC п.(6), eam_zone.hpp:110-112; точки: `pe.add(Fi)` :228, `min_r2` :233, φ-once по key :243). Пере-специфицировать при будущем раздвоении force-пасса. Virial не считается (0.0, как оракул).

Жёстко (нарушение = MUST-FIX): существующая сюита зелёная байт-идентично (F-NOOP); осознанно правятся ТОЛЬКО T-INERT/T9 в `test_w_contract.cpp` (право D5) и комментарий eam_zone.hpp:105-107 (К2).

## §2. Файлы

| Файл | Действие |
|---|---|
| `include/tdmd/potentials/eam_donation.hpp` | **NEW** (~400 строк) — §3 целиком: state, экзекьюторы, residence-страж, `eam_window_force_from_rho`, `EamDonationPass`, `zone_eam_pass_donated`, SPEC-шапки (контракт §5.2, лемма W-1, запрет full-ρ-memcmp, fb-правило, L-CLOSE-вывод §8) |
| `include/tdmd/potentials/eam.hpp` | флип `kEamPassDecls[0]` (§7, К1) — единственная правка (:106-107) |
| `include/tdmd/potentials/eam_spline.hpp` | D3-страж в `build_` (§6; :107-116, после `rhoaspl = interpolate(...)` :112) |
| `include/tdmd/potentials/eam_analytic.hpp` | D3-страж в `finalize()` (§6; :76-79) |
| `include/tdmd/potentials/eam_zone.hpp` | **COMMENT-ONLY** :105-107 (К2) — ни одной кодовой строки |
| `tests/test_eam_donation.cpp` | **NEW** (~750 строк) — §9 |
| `tests/test_w_contract.cpp` | осознанные ретаргеты T-INERT (:92-108) и T9 (:303-314) + новый T9-EAM (§7) — каждый с D5-ссылкой в диффе |
| `CMakeLists.txt` | `Test_EAM_Donation` по образцу соседних Test_EAM_* целей |
| `docs/TD_MD_Core_WContract_v1_0.md` + аддендум [ENG] к `docs/TD_MD_Core_ZoneFSM_v1_0.md` | D4 (§10); `fsm.hpp` байт-нетронут |
| CLAUDE.md / Roadmap | статус-абзац по факту приёмки |

**Байт-нетронуты (git-diff-чек приёмки):** `eam_zone.hpp` кодовые строки, `eam_ring.hpp`, `sw_ring.hpp`, `tersoff_ring.hpp`, `meam_ring.hpp`, `many_body.hpp`, `zones.hpp`, `fixed_accum.hpp`, `fsm.hpp`, `conveyor.hpp`, `cuda/*`, все SW/Tersoff/MEAM-хедеры, все существующие тесты кроме `test_w_contract.cpp`.

## §3. `eam_donation.hpp` — спецификация

Включает `eam_zone.hpp` (donation_layout/eam_window_layout/want_closure_mask — байт-нетронутые источники истины), `eam.hpp`, `many_body.hpp`, `core/{fixed_accum,zones,soa}.hpp`.

### 3.1 Персистентное состояние (то, что PR-2 потребляет как есть)

```cpp
template <typename DensAccum>            // FixedAccum<44> | FixedAccum<40>
struct EamDonationState {
  std::vector<std::vector<DensAccum>> rho;  // [zone_id][member_idx] — сырой int64; ключ zone_id, НЕ слот
  std::vector<uint32_t> ledger;             // [zone_id] closure-биты (many_body.hpp:129-139)
  void reset_pass(const core::ZoneDecomposition& zd);   // serial: старт прохода
  void reset_zone(int zid, std::size_t n_members);      // PR-2: на RECV (pass-scoped, SPEC п.(6) eam_zone.hpp:110-112)
};
```
- Индекс атома стабилен в обоих мирах: (zone_id, member_idx) — порядок `zd.members[zone]` в serial ≡ порядок msg-payload кольца (`make_preload`, eam_ring.hpp:432-439).
- «Персистентный» = переживает finalize-вызовы ВНУТРИ прохода, НЕ проходы: pass-scoped, сброс на RECV (PR-2) / `reset_pass` (serial). Mid-pass HALT отбрасывает проход целиком; частичный реплей батчей запрещён — специального API не нужно (следующий RECV делает `reset_zone`); задокументировать в шапке.
- **fb-ловушка (MUST, SPEC-комментарий):** формат ключуется ТОЛЬКО от `pot.math.density_fracbits()` (eam_analytic.hpp:93-100 / eam_spline.hpp:68-73), НИКОГДА от статической `kEamPassDecls[0].accum_fracbits==44` (eam.hpp:106-107) — иначе Q23.40-потенциал молча ломается (класс P0-ловушки `eam_run_fixed`, eam.hpp:191-194). Диспатч зеркалит eam_zone.hpp:312-319. Зуб — §9 Т-12.

### 3.2 Батч-экзекьюторы (спаны, не AtomSoA — у кольца нет глобального SoA мид-пасс, eam_zone.hpp:186-189)

```cpp
struct ZoneBlockView { const double *x, *y, *z; int n; };   // блок зоны в порядке members/msg

template <typename Math, typename DensAccum, typename Hook = NullPairHook>
void eam_donate_self(const ZoneBlockView& k, const Math& math, const core::PairGeom& geom,
                     std::vector<DensAccum>& rho_k, Hook&& on_pair = {},
                     const DonationPoison* poison = nullptr);       // пары t<u по порядку спана

template <typename Math, typename DensAccum, typename Hook = NullPairHook>
void eam_donate_cross(const ZoneBlockView& a, const ZoneBlockView& b, const Math& math,
                      const core::PairGeom& geom, std::vector<DensAccum>& rho_a,
                      std::vector<DensAccum>& rho_b, Hook&& on_pair = {},
                      const DonationPoison* poison = nullptr);      // все (i∈a, j∈b); внешний цикл — a
```
Перечисление **НЕупорядоченное, один раз на пару**; ориентация фиксирована `dx = x_a[i] − x_b[j]`; тело пары (единственная математика): `geom.reduce(dx,dy,dz,r2)` (zones.hpp:93-100; accept iff `1e-18 ≤ r2 < rc2`, ОТКРЫТЫЙ cutoff :99) → **один** `math.eval_rhoa(sqrt(r2), v, dv)` → **тот же v** в `rho_a[i].add(v)` и `rho_b[j].add(v)` (квантизация fixed_accum.hpp:38-45). `dv` вычисляется и выбрасывается — как в pass-1 оракула (:210), чтобы FP-выражения совпадали посимвольно. `on_pair(g_i, g_j)` — раз на ПРИНЯТУЮ пару (прецедент PairHook zones.hpp:117-131; дефолт no-op ⇒ ноль стоимости, прецедент :180-186).

**Лемма W-1 (в шапку экзекьюторов, дословно):** оконный pass-1 (eam_zone.hpp:202-212) обходит каждую неупорядоченную пару ДВАЖДЫ (направленно); `wx[a]−wx[b]` и `wx[b]−wx[a]` — точные IEEE-негации; `std::round` в min-image нечётен ⇒ редукция даёт точную негацию; квадраты и сумма `dx²+dy²+dz²` в том же порядке слагаемых ⇒ r2 побитово равен с обеих сторон ⇒ sqrt/eval_rhoa/rint идентичны, предикат приёмки идентичен ⇒ донация «один v в оба конца» воспроизводит мультисет квантов направленного обхода на каждый поатомный аккумулятор, а int64-сложение ассоциативно (B1) ⇒ ρ побитово ≡ оконному пересчёту при ЛЮБОМ разбиении на батчи и любом их порядке.

### 3.3 Residence-страж (серийный эквивалент membership_ok — несущая посылка L-CLOSE)

`membership_ok` живёт ТОЛЬКО в кольцах (eam_ring.hpp:460-482); serial-путь обязан привезти эквивалент, иначе смещённые от stale-`zd` координаты молча дают пару zone-k↔zone-k+2, которой нет НИ В ОДНОМ батче (детерминированно-неверно, невидимо 1-vs-z).

`check_donation_residence(a, box, zd, g)` — формула-зеркало eam_ring.hpp:462-478 ДОСЛОВНО: `g = 0.5*(width − 2.0*rcut)` (:462 — НЕ парная `(w−rcut)/2`, conveyor.hpp:76), wrap по z при PBC (:468), excess по cyclic-min (:470-478), free-края exempt (first/last, :465,:473,:477). Вызывается в ctor класса **при n≥3** (зеркало условия вызова eam_ring.hpp:315; при n<3 донации покрывают все пары тривиально); координаты статичны на протяжении сборки — в кольце соответствие: проверка по-проходно на END (задокументировать). Throw: `"donation residence: atom beyond g=0.5*(width-2*rcut) of its zone slab — donation completeness premise broken"`. Ctor-параметр `g_override = -1.0` (−1 ⇒ формула) — **тест-only** рычаг D7-мутаций; PR-2 его в кольцо НЕ прокидывает (задокументировать).

Вывод из g: два атома зон k и k+2 разнесены ≥ width−2g = 2·rcut > rcut ⇒ density-пары нет; не-донированное ребро вносит литеральный ноль (открытый cutoff zones.hpp:99) — никакого квантованного +0.

### 3.4 Донированный finalize — pass-2/3 C-фазы дословно

```cpp
struct NullDonationTrace { void on_neighbor_read(int, long, long long) {} };

template <typename Math, typename DensAccum, typename Trace = NullDonationTrace>
void eam_window_force_from_rho(const double* wx, const double* wy, const double* wz,
    const long* key, int m, const int* owned, int n_owned, const Math& math,
    const core::PairGeom& geom, double rho_cap,
    const DensAccum* rho_w,                    // ВХОД: raw-копии донированной ρ в gather-порядке окна
    std::vector<core::fixed::ForceAccum>& wFx, /*wFy, wFz,*/
    core::fixed::EnergyAccum& pe, double& min_r2, Trace&& trace = {}, int zone_j = -1);
```
Тело = eam_zone.hpp:213-246 посимвольно, с ТРЕМЯ контролируемыми отличиями:
1. pass-1 заменён входом `rho_w` (значения читаются `rho_w[aa].value()` — тот же путь int64→double; raw-копирование бит-тривиально);
2. **rho_cap — только owned (К3):** в pass-2 (:214-222) `eval_F` считается для ВСЕХ m как сегодня (kWrapup; throw :217-218 УДАЛЁН из pass-2), а проверка `if (rho > rho_cap) throw` стоит в pass-3 owned-цикле перед `eval_F(rho_w[ii])` (:227), текст: `"donated finalize: full ρ of OWNED atom exceeds the F(ρ) grid (zone <j>)"`;
3. `trace.on_neighbor_read(zone_j, key[bb], rho_w[bb].raw)` — на каждую ПРИНЯТУЮ пару pass-3 (после accept :234); тест классифицирует halo/owned по `zd` сам (не нужен `is_owned`). Дефолт NullDonationTrace — no-op, ноль стоимости.

Каналы БЕЗ изменений: `pe.add(Fi)` раз на owned из полной ρ (:227-228); `min_r2` ДО cutoff (:233); брекет `-(dphi+(fp[ii]+fp[bb])*dra)/r` (:239); квантизация Q24.40 (:240-242); φ-once `key[ii]<key[bb]` (:243). **Честная оговорка parity (в шапку):** это копия ~30 строк, не вызов (`eam_window_force` фьюзит pass-1 и заморожен); анти-дрейф-страж копии = сам гейт G-A.

### 3.5 Пошаговый класс + канонический драйвер

```cpp
struct DonationPoison {              // тест-only, дефолт ИНЕРТЕН (прецедент MEAM/Tersoff drop_class)
  bool skip_first_pair = false;      // потерять 1 пару ВНУТРИ первого непустого батча (леджер слеп)
  bool one_sided       = false;      // add(v) только в нижний конец (классический полу-баг)
  bool ledger_off      = false;      // отключить ledger-assert в finalize (для FP64/G-A-свидетелей)
};

template <typename Real, typename Math, typename DensAccum, typename Trace = NullDonationTrace>
class EamDonationPass {
 public:
  EamDonationPass(core::AtomSoA<Real>& a, const core::Box& box,
                  const core::ZoneDecomposition& zd, const EamPotential<Real, Math>& pot,
                  Trace* trace = nullptr, double g_override = -1.0,
                  const DonationPoison& poison = {});
  void donate_self(int k);           // ledger-бит уже стоит ⇒ THROW (exactly-once, INV-8 runtime)
  void donate_cross(int a, int b);   // b ОБЯЗАН быть циклическим преемником a; см. edge-order ниже
  void finalize(int j);              // (а) ledger[j]==want_closure_mask ⇒ иначе THROW (кроме ledger_off);
                                     // (б) gather окна по eam_window_layout(j,n,pbc) в БЛОЧНОМ порядке
                                     //     [pred][center][succ] С ДЕДУПОМ слотов по zone_id (К7);
                                     //     key = глобальный индекс (eam_zone.hpp:267); raw-gather ρ;
                                     // (в) eam_window_force_from_rho; (г) scatter owned в a.f* (+=, :279-284)
  EamAccum finish();                 // fold: acc.pe = pe_.value() (один EnergyAccum на проход, :254/:289);
                                     // acc.min_r2; acc.virial = 0.0
  const EamDonationState<DensAccum>& state() const;   // инспекция (тесты + PR-2)
};

template <typename Real, typename Math>
EamAccum zone_eam_pass_donated(core::AtomSoA<Real>& a, const core::Box& box,
                               const core::ZoneDecomposition& zd,
                               const EamPotential<Real, Math>& pot,
                               const DonationPoison& poison = {});
// внутри: fb-диспатч 44/40 по density_fracbits() (зеркало eam_zone.hpp:312-319) + КАНОНИЧЕСКИЙ СКАН
// (модель T-SIM, test_w_contract.cpp:224-253):
//   for j=0..n-1 { батчи donation_layout(j,n,pbc): self ПЕРВЫМИ, затем cross (eam_zone.hpp:120-125);
//                  if (!(pbc && n>1 && j==0)) finalize(j); }
//   if (pbc && n>1) finalize(0);                        // defer_head-зеркало (eam_ring.hpp:202,409-415)
// + Trace/Hook-перегрузки для G-LCLOSE/G-EDGE.
```
Ctor: `validate_pass_decls(pot.passes())` + `assert_eam_symmetric_passes` (eam.hpp:120-136) + **донационное требование** вынесено в span-принимающую свободную функцию `assert_eam_donation_descriptor(std::span<const PassDecl>)` (MUST-FIX верификации: EamPotential — `final` с захардвайренным passes() ⇒ kCOnly-копию через ctor подать НЕВОЗМОЖНО; ctor зовёт функцию, зуб Т-13 зовёт её НАПРЯМУЮ с kCOnly-копией — иначе зуб вакуумен): `passes()[0].w_class==kAccumB1 && donor_roles==(self|lo|hi)` — иначе throw; width-страж — тот же текст/условие, что eam_zone.hpp:307-311; residence-страж §3.3 (n≥3); копия пер-зонных координатных блоков by-value (бит-сохранно, прецедент gather eam_zone.hpp:265-268 / eam_ring.hpp:347-351; = ZoneMsg-payload PR-2).

**Edge-порядок `close_cross_batch` (место, где PR-2 ошибётся руками — предупреждение прямо из many_body.hpp:134-139):** аргументы — по КОНЦАМ РЕБРА, не по числовому порядку: `close_cross_batch(ledger[a], ledger[b], 0)`, где a — зона, чья ВЕРХНЯЯ граница есть ребро (CrossHi → a, CrossLo → b). Для шва: `donate_cross(n−1, 0)` ⇒ `close_cross_batch(ledger[n−1], ledger[0], 0)` — зона 0 получает CrossLo. Числовая сортировка (lo=0) инвертировала бы роли; want-mask-assert на finalize это само-ловит (маска не сойдётся), но спецификация обязана назвать конвенцию явно.

Пустые зоны: экзекьюторы no-op, но ledger-биты СТАВЯТСЯ (конвенция «бит = батч ИСПОЛНЕН, возможно вакуумно», eam_zone.hpp:153-156) — иначе END голодает. Ledger-проверка в finalize — **первый runtime-потребитель `want_closure_mask`** (eam_zone.hpp:163-183; будущая точка PR-2 — непосредственно перед `ZoneFSM::apply(END)`, комментарий :157-161). Ручная сверка скан-vs-want (проверено): j=0 free → self(0),self(1),cross(0,1) ⇒ ledger[0]=Self|CrossHi==want ✓; PBC: finalize(0) в хвосте — Self@поз.0, CrossLo от шва@n−1, CrossHi от cross(0,1)@поз.0 ✓; n=1 (free И pbc — donation_layout:132 «free path; no seam») → Self-only, want(0,1,pbc)=Self (замок T10, test_w_contract.cpp:338) ✓.

### 3.6 Доказательство G-A (в шапку хедера; домен — ЯВНО)

Обозначим g=0.5(w−2rc). Под стражем (дрейф ≤ g):
- **ρ owned(j) на finalize(j):** self(j)@поз.max(0,j−1), cross(j−1,j)@j−1, cross(j,j+1)@j, при pbc шов@n−1 — всё исполнено к дедлайну ⇒ мультисет квантов = полный = оконный ⇒ raw побитово (лемма W-1 §3.2).
- **Читаемые halo (принятые pass-3 соседи bb, |bb−ii|<rcut):** bb∈j+1 ⇒ дистанция до любого d∈j+2 ≥ (w−g)−(g+rcut) = w−2g−rcut = rcut ⇒ открытый cutoff отвергает (строгое `r2<rc2`, zones.hpp:99) ⇒ ребро (j+1,j+2) вносит bb ЛИТЕРАЛЬНЫЙ НОЛЬ; пары (bb, c∈j−1) невозможны (≥ w−2g = 2rcut); симметрично для нижнего halo ⇒ читаемые ρ/F'(ρ) бит-финальны И бит-равны оконным.
- **Deep-halo:** донированный партиал ≠ оконный партиал (персистент держит рёбра, которых окно {j−1,j,j+1} не видело, напр. (j−2,j−1)) — **ЛЕГИТИМНЫЙ дивергент; полный memcmp ρ-массивов — ЛОЖНО-КРАСНЫЙ и ЗАПРЕЩЁН как гейт** (обязательная фраза SPEC). fp[deep-halo] вычисляется (kWrapup, все m), но не читается; на G-A-домене его аргумент ≤ полной ≤ cap (D3-монотонность) ⇒ в сетке, throw невозможен.
- **Силы/PE/min_r2:** входы pass-3 бит-идентичны на читаемом множестве, тела дословны ⇒ побитово. Силы пишутся ТОЛЬКО owned, ровно раз ⇒ **memcmp сил по ВСЕМ атомам легален** (в отличие от ρ). Блочный порядок окна vs сортированный оракула невидим: ρ raw — пер-атомные копии, ForceAccum/EnergyAccum int64 (B1), min_r2 — min, φ-once гейтится key-сравнением, не позицией (READER-факт по eam_zone.hpp:243; окно оракула сортировано :55-56, дедуп несущий только pbc n=1 — закрыт дедупом К7).

## §4. rho_cap / G-HALT-ATTR — принятая семантика

**Сегодня (оракул):** cap на КАЖДОМ атоме окна, включая deep-halo с ЧАСТИЧНОЙ оконной ρ (eam_zone.hpp:215-218); в кольце throw → `Halt::Internal` (eam_ring.hpp:192-194); `rho_cap = density_grid_max()`: setfl `(Nrho−1)*drho` (eam_spline.hpp:78), аналитика +inf (eam_analytic.hpp:105). Cap-триггер УЖЕ покрыт для `eam_direct_fp64` (EamSpline.DensityBeyondGridHalts, test_eam_spline.cpp:205-224, from_analytic rho_max=5.0 + FCC — MUST-FIX верификации: прежняя формулировка «тестов НЕТ» ложна); НЕ покрыт именно `zone_eam_pass` ⇒ **Т-10а пришпиливает cap-throw ОРАКУЛА G-A (zone_eam_pass) на той же фикстуре rho_max=5.0, потом гейтится эквивалентность**.

**Принято:** донированный путь проверяет cap ТОЛЬКО на owned-атомах зоны j, на их ПОЛНОЙ ρ, в finalize(j) (К3). **Аргумент эквивалентности trigger-set (опирается на D3 ρ_a≥0 — потому D3 в этом PR):** квант `rint(v·2^fb) ≥ 0` при v≥0 ⇒ частичные суммы персистента монотонно неубывают от 0 ⇒ partial ≤ full. (а) *нет ложных HALT:* deep-halo партиалы (вкл. рёбра, которых окно не видело) не проверяются вовсе; на без-HALT домене partial ≤ full ≤ cap ⇒ все eval_F в сетке; (б) *нет пропущенных:* триггер старого пути ⇒ ∃ атом с полной ρ > cap (оконный партиал ≤ полной) ⇒ owner-finalize ТОГО ЖЕ прохода бросает; (в) *обратно:* наш триггер ⇒ полная > cap ⇒ старый путь ловит того же атома как owned его окна. ⇒ **Trigger-set идентичен на гранулярности ПРОХОДА:** `{∃ атом: полная ρ > cap}`. **Атрибуция:** различие старый-vs-новый легитимно и НЕ гейтится; атрибуция нового пути детерминированно-пер-зонная (владелец) — гейтится сообщением (К6). Mid-pass throw оставляет a.f частично записанным — как оракул; caller отбрасывает (HALT-discard). **Нормативная формулировка для PR-2 (в WContract §7):** HALT атрибутируется зоне-владельцу; mid-pass HALT отбрасывает проход целиком; частичный реплей запрещён; в кольце — throw → `set_halt(Halt::Internal)` как сегодня.

## §5. §8.1 — РЕШЕНИЕ: ARRIVAL-вариант

### 5.1 Рационале
M5b — главный потребитель: донации = полезная работа при ожидании апстрима (opportunistic try_recv-планировщик); зашивка в тело finalize(k−1) эту свободу теряет, выигрыша не имеет — по W-1 оба кандидата побитово эквивалентны. Serial-исполнение кандидаты склеивает ⇒ тест PR-1 к выбору слеп; связывает PR-2 только ФОРМУЛИРОВКА. Зуб G-ORDER (§9 Т-9) демонстрирует W-1-инвариантность выбора исполнимо. Цена — doc-churn аддендума ZoneFSM — оплачивается здесь один раз (D4 всё равно в PR-1).

### 5.2 Нормативная формулировка (в WContract §9 дословно; наследует PR-2)
> **W-РАСПИСАНИЕ.** (i) Батчи позиции j — `donation_layout(j,n,pbc)` (eam_zone.hpp:126-146); внутри позиции self, затем cross (соглашение; наблюдательно невидимо по W-1). (ii) **READY (событийная привязка):** self(k) — фронт false→true `ensure_drift(k)` прохода h (eam_ring.hpp:250; НЕ «на RECV» — dt приезжает с arrival 0, drift ленивый); cross(a,b) — CoRes(a,b): оба слота present ∧ drifted. Донации идут по ПРОДРЕЙФОВАННЫМ координатам («слот k продрейфован в позиции скана max(0,k−1)» — модель T-SIM, test_w_contract.cpp:224-253). Донор НЕ обязан быть в состоянии `c`: материальные вклады легальны из `d`/`w` — [ENG]-перепланировка диссертационного донора (аддендум ZoneFSM). (iii) **DEADLINE:** все батчи позиции j — СТРОГО ДО `finalize_owned(j)` (fin_pos(j)=j, кроме pbc j=0 → хвост); шов cross(n−1,0): ready=n−2, deadline — СТРОГО ДО finalize(n−1), который идёт in-scan; вариант «в хвосте прохода» опровергнут верификатором аудита — не воскрешать (POISON-зуб late-seam держит). (iv) **СВОБОДА:** в [READY, DEADLINE] исполнитель волен выбирать момент и порядок; побитовость от выбора не зависит (W-1) ПРИ предпосылке L-CLOSE (страж g; §8). (v) **НОРМАТИВНАЯ ТОЧКА — arrival-событие:** PR-2 планирует донации самостоятельным шагом scan-цикла между arrival/drift и finalize, НЕ зашивает в finalize. Альтернатива «внутри finalize(k−1) после START» (диссертационная буква) лежит в том же окне и W-1-эквивалентна — зафиксирована как отвергнутая по мотиву M5b.

### 5.3 COMMENT-ONLY правка (К2)
eam_zone.hpp:105-107: «…is NOT decided here … PR-1's adversarial design picks it» → «РЕШЕНО (PR-1): arrival-событие — см. TD_MD_Core_WContract_v1_0.md §9; альтернатива finalize(k−1)-после-START отвергнута (M5b)». Ни одной кодовой строки.

## §6. D3 — страж ρ_a ≥ 0: **ИЗМЕРЕНО ЧИСТО ⇒ САДИТСЯ (не DEFER)**

Измерено (два независимых метода: плотный сэмплинг + ТОЧНЫЙ минимум каждого кубика через корни производной-квадратики; харнесы — scratchpad `d3_scan.cpp`/`d3_exact.cpp`, репо не тронут): `Al_zhou.eam.alloy` — точный min сплайна **+2.92e-13** (узел r=10.10149), интервалов <0: **0**; хвост r∈[10.10149, rcut=10.1025) клампится последним узлом ≥0 (eval p→1, eam_spline.hpp:43-44); `Al_small` — min **+0.01**; `from_analytic(make_analytic_al,…)` — min **РОВНО 0** в узле r=rcut (force-shift зануляет) ⇒ **страж обязан принимать 0.0: «throw iff min < 0.0», НЕ `<=`**.

- **`EamSetfl::build_`** (eam_spline.hpp:107-116), сразу после `rhoaspl = interpolate(...)` (:112): для m=1..Nr−1 точный минимум кубика `v(p)=c₃p³+c₄p²+c₅p+c₆` (cᵢ = `rhoaspl[7m+3..6]`): min(v(0), v(1), v(p*)) по вещественным корням p*∈(0,1) уравнения `3c₃p²+2c₄p+c₅=0`; узлы m=1..Nr покрывают clamp-хвост. O(Nr) на load. Одна воронка закрывает ОБА загрузчика (`from_setfl` :146ff, `from_analytic` :165ff). Проверка одних узлов НЕДОСТАТОЧНА (межузловой овершут — то, что D3 велел мерить); сэмплинг — не доказательство. Текст ошибки называет причину: «donation-контракт требует ρ_a≥0: монотонность частичных ρ (partial ≤ final) и эквивалентность rho_cap-триггер-сета держатся только на неотрицательных квантах».
- **`AnalyticEam::finalize()`** (eam_analytic.hpp:76-79): `if (!(beta>0 && rho_amp>0)) throw` — при них force-shifted ρ_a (eam_analytic.hpp:59-65) строго убывает к v(rc)=0 ⇒ v>0 на [0,rc) (замкнуто: v'(r)=β(g(rc)−g(r))<0); строже скана.
- Горячий `eval_spline` НЕ трогать (HOST_DEVICE, побитовый CPU↔GPU контракт) — гарды только load-time. Побочно закрывается и нижняя дыра F-домена: `eval_spline` при x<0 молча экстраполирует первый интервал (eam_spline.hpp:41-42) — при ρ_a≥0 отрицательная ρ недостижима.

## §7. D5 — флип дескриптора + осознанные замки (К1)

`eam.hpp:106-107`: `{PassKind::Density, true, false, false, 44}` → `{…, 44, WClass::kAccumB1, donor_bit(kSelf)|donor_bit(kCrossLo)|donor_bit(kCrossHi)}`. Легально по всем консюмерам: w_class — capability (many_body.hpp:44-48 — прямая санкция «flipping … in PR-1 breaks no consumer»); кольца зовут только `validate_pass_decls` (kAccumB1: fb 44>0 ✓, roles≠0 ✓, kind≠BondOrder ✓ — many_body.hpp:99-126) + `assert_eam_symmetric_passes` (w_class не инспектирует, eam.hpp:120-136). Без флипа `want_closure_mask` для EAM ≡ 0 ⇒ ledger-assert в finalize ВАКУУМЕН — ровно 5-рецидивный класс structurally-dead.

Замки (`grep w_class|kEamPassDecls|eam_pass_decls` перед стартом; каждый ретаргет — с D5-ссылкой): **T-INERT** (test_w_contract.cpp:92-108) — ретаргет: «все шипнутые дескрипторы kCOnly, КРОМЕ EAM Density == ровно {kAccumB1, self|lo|hi, fb44}» (замок остаётся, не удаляется; static_assert :106 о ДЕФОЛТЕ PassDecl не трогается); **T9** (:303-314) — EAM выносится из all-kCOnly-цикла (sw/tersoff/meam остаются), добавляется **T9-EAM**: `want_closure_mask(eam_pass_decls(), j, n, pbc)` == вручную вычисленные маски (j=0/середина/n−1 free + pbc n=5 + n=1). `ValidateAcceptsAllShippedDescriptors` (:79-80) и T-EAM (:382-416) проходят без правки (проверено по коду: :414 мутирует Embedding, не Density).

## §8. L-CLOSE — честный анализ, G-LCLOSE и D7 (К4)

Обозначения: g_eam = 0.5(w−2rc) (EAM, eam_ring.hpp:462), g_pair = 0.5(w−rc) (парная формула, conveyor.hpp:76). Деривация (обязана попасть в WContract §3 и в приёмку measure-first):
- **Owned-ρ:** донации self(j)+cross(j−1,j)+cross(j,j+1) видят РОВНО те же три зоны, что окно {j−1,j,j+1} ⇒ owned-мультисет ≡ окну **при любом дрейфе** (обе стороны одинаково слепы к донору из j±2).
- **Расхождение читаемых halo** «(b∈j+1)↔(c∈j−1)»: окно эти пары СЧИТАЕТ, батча (j−1,j+1) НЕ существует; пара возможна лишь при D > g_pair.
- **Расхождение «как-прочитано vs финал»** (ребро (j+1,j+2), не донированное к finalize(j)): открывается при D > g_eam.
- **КЛАСС 3 (MUST-FIX верификации дизайна, ИЗМЕРЕН):** уже-донированное ВНЕ-оконное смежное ребро читаемого halo: персистентная ρ читаемого (j−1)-halo атома держит пары (j−2,j−1), которых окно {j−1,j,j+1} НЕ перечисляет (симметрично (j+1)-halo при finalize(j+2) держит cross(j,j+1)). Слэб-арифметика: такая пара, инцидентная ЧИТАЕМОМУ атому, существует ⟺ D > g_eam — т.е. открывается ровно на полосе (g_eam, g_pair]. На полосе донированный персистент СТРОЖЕ-ПОЛНЕЕ оконного на читаемых halo ⇒ пути НЕ «одинаково усечены».

⇒ **ИЗМЕРЕНО (пробник верификации, band-фикстура w=6.2/rc=3.0/D=1.5≤g_pair=1.6): G-A на полосе КРАСНЫЙ** (1 атом, |ΔF|=1.24 эВ/Å, через fp[b], читаемый на finalize(3): окно {2,3,4} не видит принятой пары (b,o), донированная ρ(b) держит её через cross(1,2)); контрольная no-drift цепочка — донированный ≡ zone_eam_pass ПОБИТОВО (memcmp сил==0, PE==) ⇒ сама конструкция G-A §3.6 здорова. PE остаётся побитово равным даже за g_pair (embedding фолдит owned-ПОЛНУЮ ρ; φ-набор идентичен) — **PE НЕ дрейф-зуб**. Следствия: D7b-зуб дизайнера A РЕИНСТЕЙТИТСЯ (НЕ structurally-dead); **g_eam — несущий для САМОЙ эквивалентности G-A**, не только для бит-финальности чтений; формулировка «обе стороны одинаково слепы» ЛОЖНА для читаемых halo. Свидетель бит-финальности чтений (независимость от свободы расписания W-1 для PR-2/M5b):

**G-LCLOSE (read-stability gate):** Trace пишет `(j, key[bb], raw)` для каждого ПРИНЯТОГО pass-3 соседа; после `finish()` тест сверяет каждый записанный raw с ФИНАЛЬНЫМ `state().rho` того же атома — **побитово**. Смысл: не-донированные к моменту чтения рёбра внесли литеральный ноль (открытый cutoff — L-CLOSE). Не-вакуумность: записей по НЕ-owned атомам > 0 на штатной фикстуре.

**Зуб D7 — пять измерений (все обязательны):** фикстура `lclose_drift_fixture` — ручная цепочка вдоль z, w чуть больше 2rc (rc=3.0, w=2rc+0.2 ⇒ g_eam=0.1, g_pair=1.6): owned o у верха слэба j; b∈j+1 в rc от o; d∈j+2 сдвинут вниз на D=0.5 так, что |d−b|<rc (ВЗАИМОДЕЙСТВУЮТ — зуб не вакуумен; готовой EAM-дрейф-фикстуры нет, прецедент формы — Conveyor.StaleZoneGuardFires, test_conveyor.cpp:359-378).
1. Дефолтный страж: `zone_eam_pass_donated` **THROW** (D=0.5 > g_eam=0.1) — честный HALT.
2. Мутант `g_override=g_pair` (реалистичный мутант копипасты — формула conveyor.hpp:76 без «2·»): страж молчит; **G-LCLOSE — КРАСНЫЙ** (ρ(b) при finalize(j) ≠ финальной: ребро (b,d) донируется позже) ⇒ страж load-bearing для контракта чтения.
3. Тот же мутант: G-A vs `zone_eam_pass` — **КРАСНЫЙ** (ИЗМЕРЕНО пробником верификации: класс-3 расхождение открывается при D>g_eam; |ΔF|=1.24 эВ/Å на band-фикстуре) — D7b-зуб дизайнера A реинстейтирован; аудит-буква «ослабленный g ⇒ гейт падает» выполняется УЖЕ на этой полосе.
4. Мутант `g_override=+inf` + вторая фикстура с D > g_pair: **G-A сам КРАСНЫЙ** (окно считает пары (j−1,j+1), донаций нет) — донации без стража расходятся с оракулом вообще; аудит-буква «ослабленный g ⇒ G-A падает» выполняется здесь.
5. **D7c (грефт A):** δ ≤ 0.9·g_eam ⇒ страж молчит И G-A побитово держится (страж не пере-затянут).

Ротационный зуб (структуры по zone_id под ≥2 полных оборота, pbc n≥5, побитово) — честно DEFER в PR-2: в serial `pass_rotation(h,n,pbc)` даёт слот≡метка (eam_zone.hpp:150-151); ключевание по zone_id зафиксировано типом состояния + SPEC п.(1) (:90-94).

## §9. Тест-план `Test_EAM_Donation` (tests/test_eam_donation.cpp) — гейт → kill-мутация

Фикстуры: **straddle** — `make_fcc(3,3,12,4.05)` + jitter ±0.06 (прецеденты test_eam_zone.cpp:33-58, test_eam_ring.cpp:28-49; nn=2.86 < rc=3.0, атомы на каждой границе при n∈{5,6,8}; анти-вакуумность — hook-счётчики per-boundary); **seam** — `make_fcc_pbc` (test_eam_ring.cpp:241-245, базис на z=0); **vacuum** — пустые зоны (test_eam_ring.cpp:206-235); **lclose_drift** (§8); **cap-trip** — `from_analytic(..., rho_max=5.0)` + обычный FCC (прецедент test_eam_spline.cpp:205-224 — SHOULD верификации: проще сжатого кластера); **fb40** — AnalyticEam с beta=3.3 (внутридеревный прецедент Q23.40: test_eam_zone.cpp:230, test_cuda_eam_ring.cu:491-492 — SHOULD верификации: не rho_amp).

| # | Гейт | Что сверяется | Kill-мутация |
|---|---|---|---|
| Т-1 | **G-A free** | n∈{1,2,3,5,6,8} × jitter-семёна: memcmp a.f ВСЕХ атомов ≡ `zone_eam_pass`; PE/min_r2 `EXPECT_EQ` (прецедент test_eam_zone.cpp:284-286); owned-ρ raw int64 ≡ тест-локальному O(N²) int64-пересчёту (мультисет eam.hpp:150-156; темплейт по DensAccum — обход fb44-хардвайра eam.hpp:191-194). **SPEC-запрет full-ρ-memcmp по halo — в шапках хедера И теста** | Т-5/Т-6/Т-7/Т-8 доказывают невакуумность |
| Т-2 | **G-A PBC** | n∈{1,5,6,8} (2..4 — throw donation_layout eam_zone.hpp:128-130; n=1 — дегенерат free-path + **дедуп-гейт К7**); hook-счётчик seam-пар (n−1,0) > 0 | убрать дедуп ⇒ pbc n=1 красный; drop шва ⇒ Т-6 |
| Т-3 | **G-EDGE (INV-8)** | Hook → `map<{gmin,gmax},int>` ≡ brute-оракулу всех min-image пар в rc (test_zones.cpp:63-76, :200-226); каждая пара cnt==1; анти-вакуумность: self>0 ∧ КАЖДЫЙ cross-стык>0 ∧ seam>0 | двойной `donate_cross` ⇒ ledger-THROW (exactly-once); `ledger_off` + дубль ⇒ cnt==2 |
| Т-4 | **G-ORACLE FP64** | донированные силы vs `eam_direct_fp64` (eam.hpp:41-99) <1e-10, free+PBC (прецедент test_eam_zone.cpp:178-204) — непереуступаемый свидетель выпавшей пары ВНУТРИ батча (леджер/1-vs-z/run-to-run слепы) | Т-7 ⇒ >1e-3 |
| Т-5 | **P-DROP** | скан без `donate_cross(2,3)`: (а) `finalize(2)` **ledger-THROW**; (б) с `ledger_off` — силы ≠ оракул И ≠ FP64 (двухслойно: бухгалтерия И физика) | сам зуб |
| Т-6 | **P-LATE-SEAM** | шов после `finalize(n−1)`: (а) ledger-THROW на finalize(n−1) (нет CrossHi-бита); (б) с `ledger_off` — силы owned(n−1) расходятся побитово ⇒ дедлайн load-bearing, не бухгалтерия (опровергнутый «хвостовой» вариант §5.2(iii)) | — |
| Т-7 | **skip_first_pair** | леджер ЧИСТ (не бросает), G-A красный, FP64 красный ⇒ честная граница леджера пришпилена | — |
| Т-8 | **one_sided** | G-A красный (лемма негации/двухконцовки load-bearing) | — |
| Т-9 | **G-ORDER (W-1)** | пошаговым API: реверс порядка батчей внутри позиций + swap self/cross ⇒ ПОБИТОВО идентично — исполнимый свидетель свободы §5.2(iv) | FP-состояние в батчах (double-аккумулятор вместо int64) |
| Т-10 | **G-HALT** | (а) НОВЫЙ тест старого пути: cap-trip ⇒ `zone_eam_pass` throw (фиксация оракула — дыра); (б) донированный throw на ТОМ ЖЕ конфиге, сообщение = owner-зона (К6); (в) near-cap-below ⇒ НИ ОДИН; (г) различие атрибуции старый-vs-новый документировано, НЕ гейтится | «cap-check на owned выключен» ⇒ (б) молчит |
| Т-11 | **G-LCLOSE + D7** | §8: read-stability (каждый прочитанный raw == финальному, записей>0) + пять измерений D7 | мутант `g_override=g_pair` ⇒ G-LCLOSE красный; g=∞+D>g_pair ⇒ G-A красный |
| Т-12 | **G-FB40** | fb40-фикстура: донированный ≡ оракул побитово в Q23.40; анти-вакуумность: квант образца при 44 и 40 различен (`EXPECT_NE`) | ключевание от статического дескриптора (44) вместо `density_fracbits()` |
| Т-13 | **Descriptor** | `assert_eam_donation_descriptor` напрямую: kCOnly-копия ⇒ THROW; шипнутый `eam_pass_decls()` ⇒ no-throw (страж отката флипа К1; свободная функция — MUST-FIX верификации) | откат eam.hpp:107 ⇒ ctor-путь драйвера красный |
| Т-14 | **G-VACUUM** | пустые зоны 1..4: батчи no-op, биты ставятся вакуумно, finalize не голодает, ≡ оракул побитово | убрать вакуумную установку бита ⇒ ledger-THROW |
| Т-15 | **D3-зубы** | (а) синт-setfl с отрицательным узлом ⇒ throw; (б) узлы ≥0, но межузловой undershoot <0 (резкий тейпер) ⇒ throw — **убивает мутант «проверка только узлов»**; (в) Al_zhou/Al_small/from_analytic грузятся (ровно-0 на краю принят); (г) `AnalyticEam` β<0 ⇒ finalize бросает | knots-only гард ⇒ (б) зелёный; `<=0` вместо `<0` ⇒ (в) красный |
| Т-16 | **F-NOOP-якорь** | вся существующая сюита зелёная без изменений (прод-пути w_class не читают) | — |

## §10. D4 — документы

**`docs/TD_MD_Core_WContract_v1_0.md`** (новый; весь [ENG]): §1 идея — W-фаза = донационные int64-батчи на CoRes, C-фаза = достройка; обобщение диссертационного w-scatter с сил на любой B1-пасс; «уточнение» = точное досуммирование, БЕЗ predictor/corrector. §2 нормативное расписание — `donation_layout` (eam_zone.hpp:86-146) + таблица ready/deadline + слот-vs-метка (`pass_rotation`/`slot_zone_id` :150-151) + ключевание zone_id + edge-порядок `close_cross_batch` (many_body.hpp:134-139). §3 леммы с честными границами — W-1 (условия: те же double-входы после идемпотентного drift, тот же `PairGeom::reduce`/`eval_rhoa`/fb; лемма негации §3.2), **L-CLOSE: LOAD-BEARING `membership_ok` g=0.5(width−2rcut) (eam_ring.hpp:462) + PBC-индукция в слот-пространстве + СЕРИАЛЬНАЯ НАХОДКА §8 (полоса (g_eam,g_pair] serial-эквивалентна; несёт g_eam бит-финальность чтений)**; честная граница леджера (видит целые батчи; FP64-оракулы непереуступаемы). §4 дескриптор — WClass/donor_roles/validate-правила + оговорка canon-dependent-addend ДОСЛОВНО (many_body.hpp:84-92) + fb-ловушка (runtime `density_fracbits()`, не статический 44). §5 леджер — closure_bit-раскладка, want-vs-планирование (разграничение eam_zone.hpp:157-161), вакуумный батч, pass-scoped/HALT-discard/no-partial-replay, точка проверки PR-2 — перед `ZoneFSM::apply(END)`. §6 скоуп — min_r2/PE/φ-once в C-фазе. §7 G-HALT-ATTR (§4 дословно) + D3-страж с rationale. §8 домен G-A: «halo-ρ легитимно расходится; full-ρ memcmp ложно-красный» (пример — ребро (j−2,j−1)). §9 РЕШЕНИЕ §8.1 (§5.2 дословно) + что связывает PR-2 (формулировка, не тест). §10 классификация пассов v1 (EAM Density=kAccumB1, Embedding=kWrapup, Force=C; SW/Tersoff честные нули; MEAM measure-gate). §11 do-nots: interior/frontier-предикат в силовом пути; второй тип сообщений/ρ в ZoneMsg (B5 нетронут); END/SEND и Λ-цепочка не переставляются; глобальных редукций нет; + указатель на пре-регистрацию R_W≥1.15 (PR-3b). §12 обязательства PR-2: хуки `on_zone_arrival`/`on_edge` В КОНЦЕПТ WindowForcePolicy (осознанный compile-break, НЕ `if constexpr requires`); T-ROT-runtime-зуб (≥2 оборота, n≥5, побитово); вакуумный батч исполнительно (D8); ring-повтор D7; реплика G-HALT-ATTR на кольце; ledger-проверка перед END.

**Аддендум [ENG] к `TD_MD_Core_ZoneFSM_v1_0.md`** (append-only; таблица §4 и `fsm.hpp` НЕ меняются): (1) T2 guard/action → CoRes-событие: донор — d/w-зона, триггер — ко-резидентность+drift, не «сосед перешёл в c»; arrival-вариант §5.2. (2) INV-8 timing-клауза: «ровно один батч на неупорядоченную пару зон за проход [на kAccumB1-пасс]» — привязка «при расчёте S_{i−1}» снимается только для kAccumB1. (3) Семантика d: d-зона может нести МАТЕРИАЛЬНЫЕ self/cross-донации в пер-узловом пассовом состоянии (не в ZoneMsg — B5 цел); INV-3 НЕ ослабляется (донации не интегрируют). (4) Снятый overclaim ДОСЛОВНО: «дизайн НЕ восстанавливает букву диссертации — он восстанавливает материальность w-вкладов [ENG]-перепланировкой донора». (5) Явный список нетронутого: таблица переходов, INV-3/4/5/6, INV-9-матрица (1-vs-z/run-to-run переживают донации по W-1), Λ-цепочка, чёт/нечёт §7.4, B5 §8.1, PBC-ротация §7.2 (finalize(0) в хвосте).

## §11. Честные границы serial + DEFER

Serial слеп к: выбору §8.1 (склейка — связывает формулировка §5.2); атрибуции rho_cap старый-vs-новый (склейка); транспорту/END-sync; конкурентности; HALT-яду поперёк узлов; ротации слотов (слот≡метка). 1-vs-z и run-to-run — НЕ свидетели выпавшего донора; `eam_direct_fp64` непереуступаем.

| DEFER | Куда | Почему |
|---|---|---|
| Хуки on_zone_arrival/on_edge в концепт | PR-2 | серийная сборка хуков не требует; mandatory-but-never-called = structurally-dead; расширение концепта — compile-break вместе с потребителем |
| T-ROT runtime (≥2 оборота побитово) | PR-2 | в serial нет ротации; ключевание zone_id зафиксировано типом + SPEC п.(1) |
| Вакуумный батч исполнительно на кольце (D8) | PR-2 | serial ставит биты вакуумно — конвенции не противоречит |
| Ring-side D7 + реплика G-HALT-ATTR на кольце | PR-2 | серийный эквивалент стража + пяти-зубый D7 закрывает букву PR-0a D7 |
| Абстракция «W-scheduler» | PR-2/никогда | scan-цикл драйвера И ЕСТЬ расписание |
| Ротационные device-зеркала состояния | PR-3 | — |

## §12. Приёмка и порядок работ

**Порядок:** D3-стражи → флип+замки (§7) → хедер (state → экзекьюторы → from_rho → класс → драйвер) → тесты в порядке §9 (Т-10а фиксирует оракул ДО Т-10б) → доки → **чистый ребилд обеих конфигураций** (урок Me2: stale-бинарник) → полная сюита → состязательная приёмка (standing rule).

**Критерии:** 1. `git diff` пуст на замороженном списке §2; у `eam_zone.hpp` — только комментарий-строки :105-107. 2. Существующая сюита зелёная байт-идентично (D3-гарды — no-op на всех шипнутых данных, измерено §6; флип — F-NOOP по прод-путям). 3. `Test_EAM_Donation` Т-1..Т-16 зелёные. 4. Приёмка ОБЯЗАНА: (а) воспроизвести измеренное поведение полосы (G-A КРАСНЫЙ на (g_eam,g_pair] — класс-3; контроль no-drift ПОБИТОВО зелёный); (б) инъекцией верифицировать мутанты «g→парная формула» (Т-11) и «knots-only D3» (Т-15б); (в) load-bearing свидетель ДО заявления green (урок Me5b/Me2). 5. Летопись CLAUDE.md/Roadmap — при лендинге.