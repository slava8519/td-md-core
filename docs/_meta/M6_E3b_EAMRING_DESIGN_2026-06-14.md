# TD-MD M6 PR-E3b — EamRing: threaded EAM TD-кольцо (дизайн, отдельный драйвер)

> Статус: **✅ РЕАЛИЗОВАНО 2026-06-14** (free-z; PBC-z — follow-up). `include/tdmd/potentials/eam_ring.hpp` + `Test_EAM_Ring` (8 кейсов: z=1≡serial-VV, 1-vs-z fixed&auto-dt, dual-format Q23.40, 150-шаг §3.6-style 1-vs-4, vacuum-gap пустые зоны, overlap-HALT, anti-deadlock z=1..6). Состязательная приёмка прошла: 1 реальный баг (пустые зоны → ложный HALT) исправлен+покрыт; send-delay/полнота окна/deadlock/FSM/изоляция — SOUND. Дизайн ниже — [DESIGN] (адверсариально проверен до кода). Целевой потенциал — EAM (Финнис—Синклер, PR-E1), ядро силы — `potentials::zone_eam_pass` (PR-E3, `include/tdmd/potentials/eam_zone.hpp`). Единицы LAMMPS `metal`, ансамбль NVE. Точность-эталон — `deterministic_fp64`.

---

## 0. Резюме и решение: ОТДЕЛЬНЫЙ драйвер EamRing

**Решение принято и не пересматривается:** EAM-кольцо реализуется **отдельным** драйвером `EamRing` (новый заголовок `include/tdmd/potentials/eam_ring.hpp`), а **не** правкой парного `TimeConveyor` (`include/tdmd/core/conveyor.hpp`).

Обоснование **[BUILT]**: парный конвейер несущий — от него побитово зависят все тесты `ring/LJ/Morse/§3.6` (`test_conveyor.cpp`, `test_cuda_conveyor.cu`, `test_mpi_conveyor.cu`, `Test_LJ_NIST`, `test_nve_invariant.cpp`). Любая хирургия внутри `compute_zone`/`end_zone`/`flush_sends` рискует сломать побитовость. EamRing **#include**-ит листовые примитивы и не редактирует ни байта `conveyor.hpp`/`transport.hpp`/`fsm.hpp`/`buffer.hpp`/`zones.hpp` (см. §7).

**Ключевое различие потенциалов** (определяет всю механику кольца):
- Парный конвейер — Newton-3 **CROSS**: `do_pair` пишет В ОБА слота (`conveyor.hpp:372-377`), с маской вкладов INV-3 (`conveyor.hpp:516,:521`) и `defer_head`-замыканием PBC (`:519-529`). Резидентное окно — `{S_j, S_{j+1}}` (вперёд-направленное).
- EAM — **FULL-NEIGHBOUR per OWNED**: сила каждого собственного атома пишется ОДИН раз его собственной зоной (`eam_zone.hpp:117-119`), по СИММЕТРИЧНОМУ трёхзонному окну `{S_{j-1}, S_j, S_{j+1}}` (`zone_eam_window`, `eam_zone.hpp:50-52`). НЕТ Newton-3 cross, НЕТ маски INV-3, НЕТ замыкания пары.

Из-за симметричного трёхзонного окна EAM требует, чтобы в момент расчёта силы зоны были одновременно резидентны **три** соседние зоны. Парный конвейер держит только две. Этот зазор закрывает **второй forward-hop** — задержка отправки на одну зону (§1).

---

## 1. Структура прохода и второй forward-hop (задержка отправки на одну зону) + доказательство детерминизма

### 1.1 Базовая механика парного конвейера (что мы наследуем)

В парном конвейере (`conveyor.hpp:495-531`) `compute_zone(j)` завершает зону j **внутри себя** (`end_zone(j)`, `:528`) и кладёт её в `outq`; СЛЕДУЮЩИЙ `flush_sends()` (`:463-491`) её отправляет. Поэтому при `compute_zone(j)` резидентны `{S_j, S_{j+1}}`, а `S_{j-1}` **уже перемещён/отправлен** (`std::move(s.msg)`, `:475`). Проход — `for j` с обёрткой чётно/нечётной I/O-чётности §7.4 (`:534-545`).

### 1.2 Второй forward-hop = задержка END+SEND на ОДНУ зону. КРИТИЧЕСКИЙ индекс финализации

**[DESIGN]** EamRing откладывает END каждой зоны на одну позицию скана. Резидентное окно растёт `2 → 3` зоны.

**КРИТИЧЕСКАЯ ПРАВКА (адверсариальная находка #2, fatal-class, исправлено):**
в `compute_eam(j)` финализируется **ЦЕНТРАЛЬНАЯ** зона **j** (а НЕ j-1).

Это вынуждено структурой окна. После задержки отправки на одну зону резидентны ровно `{S_{j-1}, S_j, S_{j+1}}`. Симметричное EAM-окно (`zone_eam_window`, `eam_zone.hpp:50-52`) зоны **j** есть `{S_{j-1}, S_j, S_{j+1}}` — **совпадает с резидентным набором**. Окно зоны j-1 было бы `{S_{j-2}, S_{j-1}, S_j}` — его нижний донор `S_{j-2}` к этому моменту уже эвакуирован ⇒ молчаливый сброс донора плотности (детерминированный ⇒ 1-vs-z НЕ ловит). Поэтому финализируем **центр j**.

Точное расписание `compute_eam(j)` (порядок identity, иллюстративно):
```
compute_eam(j):
  RECV:   ensure_arrival(j+1)                         // притянуть S_{j+1}
  DRIFT:  ensure_drift(j-1)[no-op], ensure_drift(j), ensure_drift(j+1)
  если j>=1: finalize_owned(j)  по окну {S_{j-1},S_j,S_{j+1}}   // финализ ЦЕНТРА j
            // S_j ещё НЕ отправляется: задержка держит его до compute_eam(j+1),
            // где он служит НИЖНИМ донором S_{(j+1)-1} зоны j+1.
```
- `j=0`: RECV S_0,S_1; drift; финализаций нет (нет полного окна центра 0 без S_{-1} при free; при PBC — см. tail-замыкание §1.4). Ничего не отправлено.
- `j=1..n-2`: финализ owned **1..n-2**; END(j); flush отправляет **S_{j-1}** (отложенный с прошлой итерации).
- `j=n-1`: финализ owned **n-1** (free: окно `{S_{n-2},S_{n-1}}`); END(n-1); flush отправляет S_{n-2}.
- **TAIL** (после цикла): отправить S_{n-1}; при PBC — финализ owned **0** по циклическому окну `{S_{n-1},S_0,S_1}` (см. §1.4), END(0), flush.

То есть `S_{j-1}` отправляется на **одну** субинтервал-итерацию позже парного кольца. Порядок отправок остаётся **монотонным** (j-1 перед j) ⇒ `hdr.sent_pos` инкрементируется в том же порядке, что и позиции прибытия; позиционная Λ-идентичность и проверки `ensure_arrival` (`step_h==h-1`, `sent_pos==arrived`, `zone_id==(r+arrived)%n_`, `conveyor.hpp:306-308`) сохраняются БЕЗ изменений.

### 1.3 Двухстадийная эвакуация payload vs force-matrix (адверсариальная правка #2/#3)

**[RISK→FIXED]** Эвакуация должна быть РАСЩЕПЛЕНА: матрица сил `ax/ay/az` слота i умирает на END(i) (как `conveyor.hpp:468-470`, гигиена INV-7) — она консумирована в `ZoneMsg.f`. Но **позиционный payload** (`x,y,z,vx,vy,vz,id,mass`) зоны i нужен ещё одну субинтервал-итерацию как read-only донор плотности/силы нижнему соседу.

**Точное правило удержания:** payload слота i живёт, пока owned зона **(i+1)** не финализирована в `compute_eam(i+1)`. Реализуется **двухстадийной очередью отправки** (НЕ единым deque): `pinned` (финализирована, но payload пришпилен как нижний донор) → `send_ready` (после `compute_eam(i+1)`) → `flush_sends` `std::move`-ит. Иными словами: `outq.push_back(i)` происходит на субинтервале END(i+1), а не END(i).

### 1.4 PBC: замыкание циклического окна (адверсариальная правка #1)

**[RISK→FIXED]** В обоих исходных дизайнах было удалено `defer_head`-замыкание, что роняло нижний донор owned-зоны 0 (`S_{-1}=S_{n-1}`). EAM-замыкания **пары** нет, но **циклическое окно всё равно оборачивается**: owned 0 нуждается в `{S_{n-1},S_0,S_1}`. Финализация owned 0 переносится в TAIL, где `S_{n-1}` (последняя собственная зона) и `S_0`, `S_1` все резидентны. Это **поворот удержания головы** (аналог `defer_head`-ротации `:519-529`, но как WINDOW-residence, НЕ как cross-пара). Индекс прохода `r=(h-1)%n_` переиспользуется как балансировочная ротация, НЕ как механизм замыкания пары.

### 1.5 Доказательство детерминизма (EamRing final ≡ serial `zone_eam_pass`)

**[DESIGN]** EamRing даёт состояние, побитово равное serial-оракулу, и 1-vs-z побитово в `deterministic_fp64`. Аргумент:
1. **Однократная запись силы.** Сила каждого owned-атома пишется ОДИН раз его 3-pass (`eam_zone.hpp:99-125`, full-neighbour, без Newton-3 cross). Нет кросс-зонной аккумуляции ⇒ нет порядка суммирования, который мог бы варьироваться. `Fx/Fy/Fz/rho` — фиксированный мультимножество int64-вкладов (`FixedAccum::add` квантует и суммирует в int64, `fixed_accum.hpp:38-47` — ассоциативно/порядок-независимо), не зависящий от того, какой узел/позиция собрали зону.
2. **Задержка отправки меняет ТОЛЬКО расписание** (когда зона END/SEND), не МНОЖЕСТВО вкладов и не входы квантователя (позиции заморожены после drift, §6).
3. **Drift зоне-локален и идемпотентен** (`s.drifted`, `:340`).
4. **PE-учёт** — `pe.add(F_i)` раз на owned-атом, `pe.add(phi)` раз на неориентированную пару при `i<j` (`eam_zone.hpp:104,123`); `EnergyAccum` фикс-точечный ⇒ pass-pe порядок-независим, z-независим.

Отсюда EamRing final == serial `zone_eam_pass` побитово и 1-vs-z побитово — тем же аргументом целочисленной ассоциативности, что и парный Gate-01.

**[RISK]** 1-vs-z — это ДЕТЕРМИНИЗМ, не КОРРЕКТНОСТЬ. Сброшенный донор плотности был бы детерминированным ⇒ 1-vs-z его НЕ ловит. Корректность ловит **только** тест `z=1 ≡ serial-оракул` (монолит — полное окно) и §3.6/PR-E6 multi-node оракул. Структурно корректность гарантирует residence-precondition в `finalize_owned` (§3.3) + guard `width≥2·rcut` (`eam_zone.hpp:153-157`).

---

## 2. Резидентность окна + slot-pool +1

**[DESIGN]** `compute_eam(j)` делает `{S_{j-1}, S_j, S_{j+1}}` резидентными-и-сдвинутыми ДО `finalize_owned(j)`:
- `S_{j+1}`: только что RECV (`ensure_arrival(j+1)`) и drift (`ensure_drift(j+1)`).
- `S_j`: RECV+drift на субинтервале j-1 (был `+1`-сосед); `s.drifted` делает повтор no-op.
- `S_{j-1}`: RECV+drift двумя субинтервалами ранее; **удержан живым задержкой отправки** (его payload ещё не перемещён, §1.3). `ensure_drift(j-1)` — оборонительный идемпотентный no-op.

Пиковая со-резидентность payload-ов: транзиентно 4 (окно `{S_{j-1},S_j,S_{j+1}}` финализируемого центра j ПЛЮС всё ещё пришпиленный `S_{j-2}` до его отправки), оседает на 3 после flush. Массив `slot[]` имеет размер n (как у парного); меняется счётчик живых payload-ов: окно+1.

**Транспорт:** ёмкость `SpscChannel` `n_+2` → `n_+3` (один доп. кредит на доп. in-flight зону задержки). **[RISK]** см. §5 — n+3 переклассифицирован адверсариальной находкой в «консервативный, не доказанно-необходимый»; реальная гарантия живости — tail-шаг. Оставляем n+3 как безопасный запас. **Эта правка ЛОКАЛЬНА** конструктору `RingTransport(z_, n_+3)` внутри EamRing и НЕ трогает `conveyor.hpp:223`.

---

## 3. Адаптация FSM/инвариантов (что EAM-кольцо хранит/отбрасывает vs парный конвейер)

### 3.1 ХРАНИМ verbatim
- **`fsm.hpp` ZoneFSM** `o→d→w→c→p→o` + `apply()` — потенциал-агностична. Каждая EAM-зона проходит RECV(o→d), SPHERE(d→w), START(w→c), END(c→p), SEND(p→o) раз за проход. **[BUILT]**
- **`node_io_order(node_id)` §7.4** (`fsm.hpp:71-74`), `initial_zone_type` §7.3 (`fsm.hpp:60-62`), `seed_first_zone` §7.1 (`fsm.hpp:66`) — verbatim. Статические свойства узла.
- **INV-3 `force_complete=true` на END** — применимо: EAM-зона завершена после своей 3-pass записи силы.

### 3.2 ОТБРАСЫВАЕМ (pair-cross-специфика)
- **Маска вкладов INV-3** (`Zone.contrib_mask`, `want_mask=(j==0?(defer_head?2:0):1)`, `conveyor.hpp:516,:521`): bit0=cross предшественника, bit1=PBC-замыкание пары. EAM пишет каждую силу ОДИН раз ⇒ учитывать нечего. `end_eam` НЕ воспроизводит этот ledger (`apply` всё ещё обнуляет `contrib_mask` безвредно, но он не читается).
- **`do_pair` Newton-3 dual-write** (`conveyor.hpp:372-377`) — НЕ переносим.
- **T2 SPHERE «частичная сила в преемника»** (`:513-518`) и `defer_head` как cross-пара (`:519-529`) — НЕ переносим (окно консумируется read-only; см. §1.4 — поворот используется только как WINDOW-замыкание).
- **INV-1 `.computed` 2-зонного окна** (`:505`) — переинтерпретируется: для EAM `eam_computed` означает «owned-зона финализирована»; финализация идёт монотонно (0..n-1 + tail), так что это естественный порядок скана; нарушение → `Halt::Internal`.

### 3.3 ЗАМЕНА INV-3: структурная residence-precondition (адверсариальная правка #2)
Вместо маски — СТРУКТУРНАЯ проверка внутри `finalize_owned(j)`:
> «сила owned-зоны j вычислена по полностью резидентному симметричному окну `{S_{j-1},S_j,S_{j+1}}`».

`finalize_owned` ассертит для всех трёх слотов окна: `present && drifted` **И** `msg.n()>0` (non-empty / not-moved-from flag) — иначе `throw logic_error → Halt::Internal`. **[RISK→FIXED]** проверка непустоты payload обязательна: без неё сброшенный донор молчит и ловится лишь z=1-оракулом. INV-8 для EAM = «pe накапливает каждую φ один раз и каждую F(ρ_i) один раз» — гарантируется телом ядра (`eam_zone.hpp:104,123`), порядок-независимо.

---

## 4. Λ-цепочка dt + INV-4 для EAM

**[DESIGN]** Λ-цепочка переиспользуется **БЕЗ изменений механики** (сильнейший reuse; адверсариальная ось `deadlock-lambda`: `broke_it=false`).
- Triple `Lambda{v,a,k2cap}` вычисляется на END из ПОСТ-kick per-atom (v,f,mass) через `buffer::speed2/accel2/k2_limited_dt_atom` — читает только интегрированное состояние, не cross-работу. END-редукции `agg.v=max, agg.a=max, agg.k2cap=min` (`conveyor.hpp:429-431`) переносятся в `end_eam` verbatim.
- **dt-handoff B2** (`conveyor.hpp:479-487`): `dt(h+1)=auto_dt(v_max(h-n+1), dt(h), k2cap(h-n+1))` решается на ПЕРВОЙ отправке, читая `slot[sent+1].lam_in` — ВХОДЯЩЕЕ значение из светового конуса прохода h-1, НЕ текущий END-aggregate (`:480`, ветка n≥2). Задержка сдвигает первую отправку (S_0) с `compute_eam(0)` на `compute_eam(1)`, но Λ-ЗНАЧЕНИЕ источено из лаг-прохода (лаг L=n-1), инвариантно к равномерному одношаговому сдвигу фазы ⇒ dt z-независим, БЕЗ Allreduce.
- **Откладывание END z-инвариантно** для агрегата: max/min — порядок-свободные редукции; pass-aggregate идентичен, завершён ли зона i на j=i (парный) или на отложенном END. Последняя отправленная зона использует `agg`, готовый до последнего flush в TAIL.
- **INV-4 forecast** `v_pred=v_full+a_full·dt·L`, `L=max(1,n-1)` при `arrived==min(1,n-1)` (`conveyor.hpp:321-330`) — действие на RECV-стороне, не затронуто задержкой. `compute_R_buf` verbatim.
- **dt(1) seed** через `make_preload` (`hdr.dt_next` на зоне 0; lam0 pre-history = t0 `max_speed/max_accel/temperature_limited_dt`) — verbatim.

**Единственное численное расхождение** (НЕ изменение Λ-цепочки): `ZoneDecomposition::build` вызывается с `reach_mult=2` (EAM EffectiveRange, плюмбинг PR-E0 через `ConveyorOptions.reach_mult/symmetric_reach`). Это даёт StaleZone-маржу `g=0.5·(width-2·rcut)` (`conveyor.hpp:617`) и build-guard'ы `width≥2·rcut` + (periodic) `n_zones≥5`. Сам StaleZone-guard (`membership_ok`) reuse verbatim; меняется только аргумент `reach_mult` и место вызова (на одну зону позже, в `end_eam`).

---

## 5. Свобода от дедлоков (§7.4)

**[DESIGN]** §7.4-чётность сохраняет свободу от дедлоков при задержке на одну зону.
- **Чётность статична** (`node_io_order(node_id)` — функция id узла, `fsm.hpp:71-74`): нечётный узел SEND-then-RECV, чётный RECV-then-SEND. В каждой соседней паре один узел отправляет первым, разрывая симметричный цикл all-block-on-recv / all-block-on-send. Это инвариантно к равномерному одношаговому сдвигу фазы отправок (задержка одинакова на всех узлах).
- **Счёт/направление транспорт-операций НЕ меняются**: каждый узел RECV ровно n зон и SEND ровно n зон за проход по forward-ребру `k→(k+1)%z`. Меняется только переплетение.
- **Backpressure — единственный новый риск**: чётный узел держит произведённую зону на одну субинтервал-итерацию дольше. **Достаточная страховка:** ёмкость `n_+2 → n_+3`. **[RISK]** Адверсариальная находка (`deadlock-lambda`) уточнила: доп. зона живёт в ЛОКАЛЬНОМ slot[]/outq, не в bounded transport-queue; transport in-flight count неизменен (всё ещё n отправок/проход). Поэтому n+3 — **консервативный запас, не доказанно-необходимый**; несущая гарантия живости — TAIL-шаг.
- **TAIL-шаг несущ для живости:** без завершающего `finalize_owned`(+ отправки S_{n-1}, при PBC + owned 0) последняя зона никогда не отправляется ⇒ downstream `ensure_arrival` блокируется навечно. Проход = n compute-шагов + 1 tail-шаг; §7.4-чётность оборачивает И цикл, И tail-отправку.
- **z=1**: межузлового ребра нет; задержка — чисто локальная бухгалтерия слотов; тривиально без дедлоков, но в матрице z=1..N. **n=1 монолит**: окно = вся система (`eam_zone.hpp:153` guard `n_zones>1`), задержка no-op.

---

## 6. Drift/kick (velocity-Verlet с зависимостью от всего окна)

**[DESIGN]** Split velocity-Verlet совместим с цельно-оконной силой EAM; kick сдвигается на одну зону позже (адверсариальная ось `drift-kick`: `broke_it=false`).
- **DRIFT (первый полу-kick + сдвиг x)** — на RECV, per-zone, verbatim `ensure_drift` (`conveyor.hpp:338-355`): `v += 0.5·dt·F_old/m; x += dt·v; clear force-matrix`. Зависит ТОЛЬКО от собственных x,v,f зоны из прохода h-1 — НЕ связан с соседями. Три оконные зоны дрифтуются НЕЗАВИСИМО, по разу, при появлении; `s.drifted` делает повторы no-op. **Барьер не нужен.**
- **Drift-completeness precondition** (не барьер): перед density-pass owned-зоны j все три `{S_{j-1},S_j,S_{j+1}}` имеют `.drifted==true`. `dt` — pass-uniform скаляр (установлен один раз при `arrived==0`, `conveyor.hpp:320`), так что все окно дрифтует с ОДНИМ dt ⇒ позиции согласованы.
- **FORCE**: 3-pass (density→embedding→force, `eam_zone.hpp:70-125`) по полному дрифтнутому окну, owned-силы пишутся ОДИН раз. Позиции заморожены между первым полу-kick (RECV) и оценкой силы.
- **KICK (второй полу-kick на END)** — СДВИГАЕТСЯ: для owned-зоны j он бежит, когда сила зоны j финализирована, т.е. внутри `compute_eam(j)` при финализации центра. `end_eam(j)` (аналог `end_zone`, тело `conveyor.hpp:381-452`) держит второй полу-kick `v += 0.5·dt·F_new/m`, зоне-локальные v/a/k2cap, Λ-aggregate, INV-4, StaleZone, `apply(END)`. Использует ТОЛЬКО свежую силу той же зоны — обратная связь по скорости не пересекает зоны ⇒ детерминизм сохранён.

Чистый двухфазный velocity-Verlet: [полу-kick+drift при прибытии] … [цельно-оконная сила при готовности] … [полу-kick при финализации]. x инвариантен через задержку ⇒ интегратор бит-точен к serial.

---

## 7. Изоляция парного пути

**[DESIGN]** Парный `TimeConveyor` остаётся **БАЙТ-НЕТРОНУТЫМ** (адверсариальная ось `isolation-edge`: pair-isolation PASS).
- EamRing — отдельный класс-шаблон `EamRing<Real,Math>` + свободная функция `run_eam_ring(atoms, box, pot, opts)` в `include/tdmd/potentials/eam_ring.hpp`. НЕ включает и не правит `conveyor.hpp`.
- Шарит листы только через `#include` (header-only, без правок): `transport.hpp` (RingTransport/SpscChannelT/ZoneMsg/ZoneHeader/ITransport — конструируется с ёмкостью `n_+3`), `fsm.hpp`, `buffer.hpp`, `zones.hpp`, `eam_zone.hpp`, `fixed_accum.hpp`.
- `conveyor_detail::Lambda` — POD копируется в `eam_ring_detail::Lambda` (не лезем в detail-namespace конвейера). `ConveyorOptions` переиспользуется как есть (несёт `reach_mult/symmetric_reach/mixed_transport/auto_step/ts/r_min_halt` — PR-E0), БЕЗ правки struct.
- **Обязательный pair-safe рефакторинг `eam_zone.hpp`** (адверсариальная находка `isolation-edge`, feasibility break #1): `zone_eam_pass_impl` — это **whole-system wrapper** над ГЛОБАЛЬНЫМ `AtomSoA<Real>&` (аллоцирует `Fx/Fy/Fz/pos` размера `a.n`, читает `a.x[i]` по ГЛОБАЛЬНОМУ индексу, пишет `a.fx[i]` для ВСЕХ i, `eam_zone.hpp:58-138`). Кольцо НЕ имеет глобального AtomSoA mid-pass — атомы фрагментированы в `slot[].msg` (ZoneMsg SoA, адресуется `(slot, local-index)`). Поэтому **нельзя** звать `zone_eam_pass` as-is. Решение (чистое ДОБАВЛЕНИЕ): извлечь 3-pass тело в свободную `eam_window_force<Real,Math,DensAccum>(window-payloads, pos_map, PairGeom, Math)` над явным window-списком, адресуемым через `(slot,local)`-карту. `zone_eam_pass` остаётся НЕИЗМЕНЁННЫМ оракулом, зовущим ту же свободную функцию над глобальным AtomSoA (бит-сохранность под существующим `test_eam_zone`-оракулом). Это трогает `eam_zone.hpp`, но НЕ `conveyor.hpp`.
- **Dual-format dispatch обязателен** (feasibility, риск): per-window density-формат выбирается через `density_fracbits()` (44 → `FixedAccum<44>` Q19.44; 40 → `<40>` Q23.40, `eam_zone.hpp:158-165`). EamRing зовёт рефакторенную window-force ЧЕРЕЗ тот же dispatch, НЕ через `eam_run_fixed` (хардвайр `<44>`, HALT на fb≠44). `DensAccum` template-параметр должен достигать per-zone unit.

Все парные тесты линкуют неизменный `conveyor.hpp`; `test_eam_ring.cpp` компилируется независимо; `gpu_gate` нетронут.

---

## 8. План PR (с приёмочными тестами на PR)

### PR-E3b-1 — `eam_window_force` рефакторинг + per-zone executor (pair-safe ADD)
**[DESIGN]** Извлечь 3-pass тело в свободную `eam_window_force<Real,Math,DensAccum>` над фрагментированными payload-ами + `(slot,local)`-картой; `zone_eam_pass` зовёт её над глобальным AtomSoA.
- **Приёмка:** существующий `test_eam_zone.cpp` оракул проходит БИТ-в-бит (wrapper-поведение сохранено, `memcmp==0`); новый микротест `eam_window_force` над 3 явными payload-ами ≡ `zone_eam_pass_impl` на той же конфигурации; dual-format (β=3.3, `density_fracbits()==40`) ветка не оборачивает int64.

### PR-E3b-2 — EamRing skeleton: задержка на одну зону, residence, FSM-адаптация, deadlock
**[DESIGN]** `EamRing<Real,Math>` + `run_eam_ring`; двухстадийная очередь отправки (§1.3); finalize-центр-j (§1.2); residence-precondition (§3.3); ёмкость `n_+3`; TAIL-шаг.
- **Приёмка:**
  - **Bitwise vs serial-оракул (КОРРЕКТНОСТЬ):** EamRing z=1 final `(x,v,f)` после K шагов ≡ serial velocity-Verlet, зовущий `zone_eam_pass` каждый шаг, `memcmp==0`. fixed И auto dt. Фикстура: Al-72 golden + `AnalyticEam` (Финнис—Синклер) из `test_eam_zone.cpp`.
  - **Window-residence single-step trace (адверсариальная правка #5):** явно ассертить, что owned-зона **j** финализируется на скане **j** (post-fix), все три `{S_{j-1},S_j,S_{j+1}}` payload-а non-empty. Единственный тест, отличающий корректное расписание от off-by-one (1-vs-z не может).
  - **Anti-deadlock z=1..N:** EamRing ≥2z шагов для z∈{1,2,3,4,5,6} (нечётные И чётные кольца) без зависания, `Halt::None`; включает z=1 (локальная задержка), n=1 монолит, и обязательный TAIL `finalize+flush`.
  - **HALT-кейсы (честное срабатывание):** (a) Overlap (совпадающие атомы → `Halt::Overlap` через min_r2, B10); (b) Causality INV-4 (быстрый атом → `Halt::Causality`); (c) StaleZone (атом сдвинут >g=0.5·(w-2·rcut) → `Halt::StaleZone`); (d) density-range ρ>ρ_cap → `runtime_error/Halt::Internal` (`eam_zone.hpp:93-94`); (e) узкая зона `width<2·rcut` → build/kernel throw (`eam_zone.hpp:153-157`); (f) corrupted-schedule residence-precondition → `Halt::Internal`. Каждый возвращает длиннейший валидный stats-префикс.

### PR-E3b-3 — Λ-цепочка dt z-независимость + 1-vs-z threaded
**[DESIGN]** Λ-chain dt-handoff + INV-4 (§4), z-независимость под задержкой.
- **Приёмка:**
  - **1-vs-z BITWISE в `deterministic_fp64`:** EamRing z=1 final ≡ z∈{2,3,4,5,6} побитово, при `n_zones≥5` (PBC reach_mult=2 build-guard) и free-BC случае `n_zones≥2`. fixed И auto dt (доказывает z-независимость dt-последовательности под задержкой). Зеркало `Determinism1vsZFixed/Auto`.
  - **Dual-format на кольце:** повтор 1-vs-z с β=3.3 `AnalyticEam` (`density_fracbits()==40`, Q23.40) — per-zone dispatch z-независим.
  - **Λ-chain auto-dt оракул:** реплей `dt(h+1)=auto_dt(v_max(h-n+1),dt(h),k2cap(h-n+1))` из PassStats ≡ dt-последовательности EamRing для z≥2 (доказательство no-Allreduce z-независимости).

### PR-E3b-4 — §3.6-реплика на EAM + NVE-инвариант
**[DESIGN]** Длинный детерминизм-прогон.
- **[RISK→FIXED] Re-фикстура §3.6 (адверсариальная находка `isolation-edge`, decisive break):** литеральный «Al-72, n_zones=2, rcut=4.0» **БРОСАЕТ** на EAM build-guard'е (Al-72 z-extent 8.1 Å, n_zones=2 → width 4.05 < 2·rcut=8.0). Использовать **высокий FCC-слаб** (как в `test_eam_zone`: cx=cy=3, cz=12, a0=4.05 ⇒ z=48.6 Å), rcut=3.0, `n_zones≥5`, причём width **КОМФОРТНО** выше 2·rcut (напр. n_zones=6 ⇒ width 8.1, g=1.05 Å), чтобы 25 900-шаговый термический прогон не словил spurious StaleZone (g→0 на минимуме width=2·rcut). Z (число узлов) задаётся независимо (напр. Z=4 при n_zones≥5).
- **Приёмка:**
  - **§3.6-реплика на EAM:** высокий FCC-слаб, free/periodic z, auto-step, 25 900 шагов, 1 узел vs N узлов (Z=4) — НУЛЕВОЕ отклонение координат И скоростей (`memcmp==0`). Прямой EAM-аналог `Replica36BitwiseDeterminism`.
  - **PBC three-zone wrap:** periodic-z, `n_zones≥5`, EamRing ≡ serial `zone_eam_pass` с циклическим окном (`S_{-1}=S_{n-1}`); импульс сохранён (Σp дрейф ≤ fixed-point floor). Подтверждает head/tail PBC-замыкание окна (§1.4).
  - **Zone-order / send-delay consistency:** EamRing final ≡ serial `zone_eam_pass` по произвольной перестановке порядка зон (`validate_zone_order`) — расписание отложенного END не меняет мультимножество вкладов.
  - **NVE-инвариант (long run):** EamRing 50k шагов, multi-zone ≡ 1-zone побитово; секулярный тренд в калиброванном потолке (аналог `test_nve_invariant.cpp`).

---

## 9. Сводка адверсариальной проверки (найдено + исправлено)

| Ось | broke_it | Находка | Правка (внесена в дизайн) |
|---|---|---|---|
| **determinism** | **да (fatal)** | Off-by-one индекс финализации: исходные дизайны финализировали owned **j-1** по окну `{S_{j-2},S_{j-1},S_j}`, но задержка на одну зону держит резидентными `{S_{j-1},S_j,S_{j+1}}`. Нижний донор `S_{j-2}` уже эвакуирован ⇒ сброс донора. Z-однороден ⇒ 1-vs-z НЕ ловит. PBC: оба дизайна удалили head-замыкание ⇒ ронялся нижний донор owned-0. | **§1.2:** финализировать **ЦЕНТР j** (окно центра ≡ резидентный набор, подтверждено `zone_eam_window`, `eam_zone.hpp:50-52`). **§1.3:** двухстадийная очередь, payload слота i держится до финализации owned (i+1). **§1.4:** PBC owned-0 финализируется в TAIL по `{S_{n-1},S_0,S_1}`. **§3.3:** residence-precondition ассертит `present&&drifted&&n()>0`. **§8 PR-E3b-2:** добавлен z=1-vs-serial-оракул (КОРРЕКТНОСТЬ) и single-step window-residence trace. |
| **deadlock-lambda** | нет | Λ-цепочка z-независима (dt решается на первой отправке из light-cone-lag значения, не из текущего END-aggregate, `conveyor.hpp:479-487`); drift/kick split цел (`dt` pass-uniform, `:320`). Уточнение: ёмкость **n+3 НЕ доказанно-необходима** — доп. зона в локальном outq/slot[], transport in-flight count неизменен. | **§5:** n+3 переклассифицирован в «консервативный запас»; несущая гарантия живости — **TAIL-шаг** в §7.4-обёртке. Λ-цепочка/drift-kick — verbatim reuse (§4, §6). |
| **isolation-edge** | **да** | (1) Парная изоляция держится. (2) Design-A claim «verbatim `zone_eam_pass` reuse» ЛОЖЕН: это whole-system wrapper над глобальным AtomSoA; кольцо имеет фрагментированные payload-ы. (3) §3.6-фикстура «Al-72 n_zones=2 rcut=4.0» **БРОСАЕТ** на EAM reach_mult=2 guard (width 4.05 < 8.0). (4) StaleZone-маржа g=0.5·(w-2·rcut) туже парной; на минимуме width g→0. | **§7:** обязательный pair-safe рефакторинг — свободная `eam_window_force` над фрагментированными payload-ами + `(slot,local)`-карта; `zone_eam_pass` остаётся неизменённым оракулом. Dual-format dispatch сохранён (НЕ через `eam_run_fixed`). **§8 PR-E3b-4:** §3.6 пере-фикстурена на высокий FCC-слаб (n_zones≥5, width КОМФОРТНО >2·rcut, g≈1.05 Å). |

**Итог:** все валидные адверсариальные находки внесены. Критический детерминизм-разрыв (off-by-one) и feasibility-разрыв (whole-system wrapper, §3.6-фикстура) закрыты. Дизайн готов к реализации по плану §8.
