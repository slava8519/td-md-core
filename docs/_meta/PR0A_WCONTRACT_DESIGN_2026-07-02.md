# PR-0a — W-контракт, шаг 0: дизайн (состязательный, финальный)

**2026-07-02 · ветка `opus` · дизайн-workflow `wf_ef54881d-1ba` (4 читателя → 2 дизайнера → судья → 4 верификатора; все ACCEPT-WITH-FIXES).**
Требования: `AUDIT_W_PHASE_NEIGHBORS_2026-07-02.md` §3.1/§3.3/§5 (буллет PR-0a) + гигиена §7 п.2/п.4.

## Применённые MUST-FIX верификации (вшиты в реализацию)

1. **`template class`-инстанциации колец в compile-TU статически ill-formed** (дефолтный
   convenience-ctor конструирует GPU-политику одним аргументом, всем нужен `(params, box)`)
   ⇒ основная форма = **member-scoped explicit instantiation `run()`** (implicit-инстанцирует
   класс ⇒ class-scope концепт-static_assert срабатывает; ill-formed ctor не трогается).
2. **Include-граф TU**: `eam_conveyor_gpu.cuh` обязан идти ПЕРЕД `eam_window_force_gpu.cuh`
   (`using namespace eam_sn_detail` определён только там); `sw_ring.hpp`/`eam_ring.hpp`
   включаются явно (policy-хедеры SW/EAM ring-хедеров не тянут). Ложный комментарий
   «транзитивно» из дизайна удалён.
3. **Делегация GPU-гейта требует `#include "tdmd/potentials/eam.hpp"`** в
   `eam_window_force_gpu.cuh` (не достижим транзитивно; cycle-safe — eam.hpp не тянет cuda).

## Отклонение реализации (measure-first, зафиксировано)

- **Файловая карта §1 скорректирована фактом**: НОВЫХ кодовых файлов ТРИ (`tests/test_w_contract.cpp`,
  `tools/cuda_compile_check.cu`, `tools/cuda_compile_check_tersoff.cu`) — §1 ниже писался до
  измеренного TU-сплита; §8-сниппет с `template class` в одном TU — SUPERSEDED этим заголовком
  (обе правки: member-run() форма + сплит).
- **TU расщеплён на два**: `zone_sw.cuh` и `zone_tersoff.cuh` оба определяют
  `tdmd::cuda::kMaxNbr` ⇒ SW и Tersoff не живут в одном TU (та же причина, по которой
  тестовые TU раздельны). `cuda_compile_check.cu` = {EAM,MEAM,SW},
  `cuda_compile_check_tersoff.cu` = Tersoff. `zone_*.cuh` байт-нетронуты (F-NOOP).

---

Все несущие факты обоих дизайнов сверены с живым деревом (include-граф `eam_zone.hpp:11 → eam.hpp`, локальность `kPasses`, идиома `if constexpr requires` во всех 4 кольцах, сигнатура драйвера :121-128, MUST-FIX-формулировки аудита §3.1/§3.3/§3.6/§5/§7/§8). Ниже — слитый финальный дизайн-док.

---

# PR-0a — W-контракт, шаг 0: инертный дескриптор + `donation_layout` + леджер + firewall-зубы + CI compile-TU + закрытие FIREWALL GAP

**ФИНАЛЬНЫЙ ДИЗАЙН (слияние Designer A minimal-diff + Designer B future-proof; конфликты разрешены явно — §0.1). 2026-07-02, ветка `opus`. Исходник требований: `docs/_meta/AUDIT_W_PHASE_NEIGHBORS_2026-07-02.md` §3.1/§3.3/§3.6/§5 (буллет PR-0a) + гигиена §7 п.2/п.4.** Замечание о нумерации (не распространять путаницу): PR-лестница аудита цитирует FIREWALL GAP как «§8.4», но заголовок гигиенических находок — «## 7.» (внутренняя нумерация аудита сдвинута на 1); в коде и здесь ссылаемся «аудит §7 п.4».

**Контракт PR — чисто-аддитивный F-NOOP.** Ни одно кольцо не потребляет новые примитивы; дескрипторы 4 потенциалов не меняют значений (новые поля — только хвостовые с инертными дефолтами `kCOnly`/`0`); 14-арг `compute()` заморожен; `fsm.hpp`/`zone.hpp`/`conveyor.hpp`/`transport.hpp`/все `cuda/zone_*.cuh`/`sw.hpp`/`tersoff.hpp`/`meam.hpp`/`ci.yml` — git-байт-нетронуты. **Честная оговорка (прецедент `35cf18b`):** PR не «нулевой» буквально — закрытие FIREWALL GAP добавляет throw на пути, который раньше шёл без стража, а CPU-EAM-кольцо впервые получает живой (но no-op на легальном EAM) firewall-вызов. F-NOOP скоуплен на все ЛЕГАЛЬНЫЕ существующие входы; единственное поведенческое отличие — новые THROW на нелегальных («латентно-неверное → громкое»). Гейт приёмки: весь suite байт-в-байт после ЧИСТОГО ребилда (CPU 41/41 + CUDA 58/58) + новые зубы non-vacuous (§9).

---

## 0. Резюме объёма

| # | Пункт задания | Решение (кратко) |
|---|---|---|
| 1 | WClass + donor_roles | `many_body.hpp`: два хвостовых поля `PassDecl` с инертными дефолтами; позиционные 5-элементные инициализаторы всех 4 потенциалов и 6 тестовых TU компилируются без правок |
| 2 | donation_layout(j,n,pbc) | чистая per-j функция в `eam_zone.hpp` сразу под `eam_window_layout` (буква аудита §3.1, анти-E5b-F4); MUST-FIX pbc(n−1,0)-до-finalize(n−1) вшит СТРУКТУРНО + нормативная таблица ready/deadline и независимый симуляционный зуб |
| 3 | want_mask / леджер | `DonorRole`+`donor_bit`+`closure_bit`+`want_closure_mask`+`close_cross_batch`+хелперы slot↔label; хранилище НЕ добавляется — `core::Zone::contrib_mask` (сброс на RECV/SEND, `fsm.hpp:47,51`) уже готовый pass-scoped INV-3-леджер; PR-0a поставляет только want-сторону |
| 4 | Firewall-зубы | `validate_pass_decls()`: kAccumB1 ⇒ `accum_fracbits>0`, запрещён на BondOrder (ζ-класс FP-редукции) + стражи ролей/ёмкости/iterative; честная оговорка про canon-dependent-addend-класс — дословно в комментарии |
| 5 | CI-слепота | `tools/cuda_compile_check.cu` (исполняемый, explicit instantiation с измеренным фолбэком) в tools-блоке `if(TDMD_WITH_CUDA)`; **ci.yml не меняется вообще** |
| 6 | FIREWALL GAP + concept | `eam_gpu_run_singlenode`: хвостовой `passes`-параметр с дефолтом = единый namespace-источник `kEamPassDecls` + БЕЗУСЛОВНЫЙ гейт первой строкой; `GpuEamWindowForce::assert_supported` ДЕЛЕГИРУЕТ общему гейту (одна истина); concept `WindowForcePolicy` (class-scope static_assert во всех 4 кольцах) убивает opt-in `if constexpr requires` |
| 7 | Приёмка | §10: чистый ребилд, 42 CPU + 58 CUDA байт-в-байт, gpu_gate чисто, облачный cuda-compile зелёный (обе ноги mpi), A/B-hex-пояс, kill-мутации ревьюером |

### 0.1. Разрешённые конфликты A↔B (несущие решения слияния)

| # | Вопрос | A | B | ВЫБОР + почему |
|---|---|---|---|---|
| К1 | Форма `donation_layout` | per-j эмиссия `DonationBatches` (батчи позиции j исполняются ДО finalize(j)) | плоские `DonationBatch{ready_scan,deadline_scan}` + `donation_schedule` | **A (per-j)** — это операционная форма, которую PR-1/2 вставят в скан-цикл одной строкой, и она делает MUST-FIX «шов до finalize(n−1)» непредставимо-нарушимым (у API нет способа выразить «шов в хвосте»). Из B графтится: нормативная таблица ready/deadline как документация + **независимый симуляционный зуб T-SIM** (главный анти-off-by-one), хелперы `pass_rotation`/`slot_zone_id`. Открытый вопрос аудита §8.1 (arrival vs внутри-finalize(k−1)) per-j форма НЕ предрешает: оба кандидата живут ВНУТРИ окна позиции j «после drift, до finalize(j)» — микроточку выбирает PR-1 |
| К2 | Источник EAM-дескриптора | namespace-scope `kEamPassDecls` в `eam.hpp`, член класса удаляется | оставить член, дефолт драйвера = `EamPotential<double,EamSetfl<double>>::kPasses` | **A** — грепом подтверждено: `kPasses` EAM нигде не используется вне `eam.hpp`; namespace-константа не требует Math-инстанциации у драйвера и исключает две копии истины (зуб T-SRC). SW/Tersoff/MEAM члены не трогаются (их файлы — байт-свидетели F-NOOP) |
| К3 | GPU-гейт: дубликат+parity-тест (A, D5/R6) vs делегация общему (B) | дубликат ~15 строк + вечный parity-зуб T-PAR | делегация | **B (делегация)** — `eam_window_force_gpu.cuh` всё равно правится (переписывание honest-scope комментария), делегация ~5 строк убивает риск дрейфа двух копий НАВСЕГДА вместо вечного полисинга тестом; существующие CUDA-зубы проверяют accept/reject-семантику (не тексты) и остаются зелёными; тексты сообщений сохраняются дословно с префиксом `who`. T-PAR исчезает |
| К4 | Строгость validate | roles только на kAccumB1; kAccumB1+roles==0 → throw | roles разрешены и на kAccumCanonSuffix; без правила roles≠0 | **A (строго)** + графт правила B `iterative ⇒ kCOnly`. Валидатор, принимающий формы с неопределённой семантикой (canon-донации не спроектированы), — ровно overclaim-класс; ослабить в PR-потребителе — осознанная однострочная правка со своими зубами, ужесточить постфактум — больно. Оба ослабления задокументированы как relax-точки |
| К5 | Куда класть static_assert концепта | в run() у firewall-блока | class-scope кольца | **B (class-scope)** — срабатывает на ЛЮБОЙ инстанциации кольца (даже TU, который только конструирует), а не только при инстанциации run() |
| К6 | 5 существующих тест-пинов `static_assert(requires{...})` | оставить + добавить рядом концепт-ассерты | заменить телом на концепт | **Ни то, ни то: оставить нетронутыми, ничего в те TU не добавлять.** Class-scope static_assert (К5) субсуммирует их глобально; концепт-ассерты GPU-политик едут в `cuda_compile_check.cu` (проверяются В ОБЛАКЕ — тестовые TU облако не компилирует), CPU-политик — в `test_w_contract.cpp`. Минимальный дифф + облачное покрытие |
| К7 | Compile-TU: сила инстанциации | поверхность тестовых TU (консервативно, D3) | explicit instantiation definitions (все члены) | **B с измеренным фолбэком A** — explicit instantiation строго сильнее и защищает PR-1..4 (весь смысл TU — страховка W-лестницы); риск «впервые не скомпилится» снимается measure-first: первый локальный билд решает; при провале — фолбэк на odr-use-поверхность + находка в PR-тексте. Как landed PR инертен в обоих исходах |
| К8 | MPI compile-TU | defer (D1) | SHOULD в PR-0a | **A (defer)** — вне буквы PR-0a-буллета аудита; hoist `find_package(MPI)` — лишний configure-риск в F-NOOP-PR. Рецепт B зафиксирован в §11 как готовый follow-up |
| К9 | Расположение wclass-проверки | кольца зовут `validate_pass_decls` отдельно, ДО `assert_supported` | validate первой строкой внутри всех 7 `assert_supported` | **A** (слоистость: validate = самосогласованность дескриптора, assert_supported = capability политики) + графт B: общий EAM-гейт `assert_eam_symmetric_passes` ВНУТРИ зовёт validate (⇒ драйвер и прямые вызовы EAM-гейта защищены). Побочный выигрыш: `{sw,tersoff,meam}_window_force_gpu.cuh` остаются байт-нетронутыми |

---

## 1. Файловая карта

**НОВЫЕ файлы (2):**
- `tests/test_w_contract.cpp` → таргет `test_w_contract`, ctest-имя `Test_W_Contract` (CPU, без CUDA/LAMMPS; 42-й CPU-тест).
- `tools/cuda_compile_check.cu` → таргет `cuda_compile_check` (compile-surface, §8).

**ИЗМЕНЯЕМЫЕ файлы (10):**
1. `include/tdmd/potentials/many_body.hpp` — `WClass`/`DonorRole`/поля `PassDecl`/`validate_pass_decls`/`closure_bit`/`close_cross_batch`/concept `WindowForcePolicy` (+`#include <concepts>`, `<cstdint>`, `<stdexcept>`; `<span>`/`<string>`/`fixed_accum.hpp` уже есть).
2. `include/tdmd/potentials/eam_zone.hpp` — `DonationBatches`/`donation_layout`/`want_closure_mask`/`pass_rotation`/`slot_zone_id` (под `eam_window_layout:70-83`); тела `zone_eam_window`/`eam_window_layout`/`eam_window_force`/`zone_eam_pass` — байт-нетронуты.
3. `include/tdmd/potentials/eam.hpp` — namespace-scope `kEamPassDecls` + `eam_pass_decls()` + `assert_eam_symmetric_passes()`; `EamPotential::passes()` возвращает `{kEamPassDecls,3}`; член `kPasses[3]` (:105-109) и out-of-line определение (:145) удаляются.
4. `include/tdmd/potentials/eam_ring.hpp` — `CpuEamWindowForce::assert_supported` (новый, делегирует); class-scope static_assert концепта; firewall-блок run() (:113-119) → безусловный.
5. `include/tdmd/potentials/sw_ring.hpp` — то же (static_assert + безусловный блок :174-175) + `SwWinForceLocalKey` (:121, test-only) получает делегирующий `assert_supported`.
6. `include/tdmd/potentials/tersoff_ring.hpp` — то же (:211-212) + `TersoffWinForceLocalKey` (:158) — делегация.
7. `include/tdmd/potentials/meam_ring.hpp` — то же (:221-222; CPU-политика уже имеет assert_supported).
8. `include/tdmd/cuda/eam_conveyor_gpu.cuh` — закрытие GAP (:114-128, §7).
9. `include/tdmd/cuda/eam_window_force_gpu.cuh` — `assert_supported` → делегация общему гейту (К3) + honest-scope блок (:254-262) переписан «оба зазора ЗАКРЫТЫ PR-0a».
10. `CMakeLists.txt` — `cuda_compile_check` (tools-блок) + `test_w_contract` (tests-блок, зеркало `test_metrics` :355-357).
11. `tests/test_cuda_eam_ring.cu` — +1 тест T-GAP (§9).

**БАЙТ-НЕТРОНУТЫ (свидетели F-NOOP, проверяются `git diff` на приёмке):** `core/fsm.hpp` (обязательство аудита §3.6 — таблица переходов байт-неизменна во ВСЕХ PR), `core/zone.hpp`, `core/conveyor.hpp`, `core/transport.hpp`, `potentials/{sw,tersoff,meam}.hpp` (дескрипторы добираются хвостовыми дефолтами), все `cuda/zone_*.cuh`, `cuda/{sw,tersoff,meam}_window_force_gpu.cuh` (следствие К9), `cuda/conveyor_gpu.cuh`, `.github/workflows/ci.yml`, все 5 существующих тест-пинов и 6 рукописных PassDecl-массивов в тестах.

---

## 2. Дескриптор: WClass + donor_roles (`many_body.hpp`)

Дословно форма аудита §3.3, приложенная к живому `PassDecl` (:39-45). Поля — **строго в хвост**: все инстансы используют позиционную агрегатную инициализацию `{kind, reuse, transpose, iterative, fracbits}` (eam.hpp:105-109, sw.hpp:269-272, tersoff.hpp:313-316, meam.hpp:1731-1735 + рукописные массивы 6 тестовых TU, напр. `test_cuda_eam_ring.cu:556-558`); вставка в середину сломала бы их все.

```cpp
// === PR-0a (W-контракт, аудит §3.3) ===
// W-классификация пасса: КАК пасс может исполняться донационными батчами int64
// при событиях ко-резидентности зон (CoRes). СЕМАНТИКА: w_class объявляет
// СПОСОБНОСТЬ, не обязательство — C-only потребитель (сегодня: каждое кольцо и
// eam_gpu_run_singlenode) легально ИГНОРИРУЕТ поле и гонит монолитный путь;
// поэтому флип EAM Density -> kAccumB1 в PR-1 не ломает ни одного потребителя.
// В PR-0a ничто не потребляет поле — дефолт kCOnly воспроизводит сегодняшнее
// поведение бит-в-бит.
// kAccumB1 определён B1-ассоциативностью int64 (лемма W-1): для ЛЮБОГО разбиения
// мультимножества вкладов на батчи и любого порядка их исполнения сырой int64
// равен монолитной сумме. => kAccumB1-пасс ОБЯЗАН копить в FixedAccum
// (accum_fracbits>0); FP-редукция (Tersoff zeta: FP64 by design, fb=0 INERT —
// tersoff.hpp:308-312) под W-1 не подпадает, её класс — kAccumCanonSuffix
// (DEFER-AND-GATE, v1 не реализует).
// Донационные батчи НЕ трогают min_r2 и PE/phi-once — те остаются в force-пассе
// C-фазы (eam_zone.hpp:129-143); при будущем раздвоении force-пасса это условие
// пере-специфицировать (аудит §3.3 MUST-FIX).
enum class WClass : uint8_t {
  kCOnly,             // только C-фаза (дефолт = сегодняшнее поведение)
  kAccumB1,           // int64-накопляемый: донационные батчи в d/w (W-1/W-2)
  kAccumCanonSuffix,  // FP-редукция под блочно-зонным каноном + carry (DEFER-AND-GATE)
  kWrapup             // нелинейная пер-атомная свёртка — C-фаза
};

// Роли доноров зоны k (слот-пространство прохода): self(k); cross(k-1,k);
// cross(k,k+1). kReserved3 зарезервирован под будущий screening/transpose-донор
// класс (MEAM, предикат depth(i) > rc+g — аудит §3.4); в v1 бит запрещён.
// Раскладка бита в леджере: closure_bit(pass_idx, role) = pass_idx*4 + role,
// pass_idx < 8 (uint32 = 4 роли x до 8 пассов — расширение INV-3-леджера
// core::Zone::contrib_mask).
enum class DonorRole : uint8_t { kSelf = 0, kCrossLo = 1, kCrossHi = 2, kReserved3 = 3 };
inline constexpr uint8_t donor_bit(DonorRole r) { return uint8_t(1u << unsigned(r)); }
inline constexpr uint8_t kDonorRolesValidMask = 0x07;  // kReserved3 запрещён в v1

struct PassDecl {
  PassKind kind;
  bool reuses_candidates = true;
  bool needs_transpose = false;
  bool iterative = false;
  int  accum_fracbits = 40;
  // --- PR-0a: ИНЕРТНЫЕ дефолты (нулевое изменение поведения) ---
  WClass  w_class     = WClass::kCOnly;
  uint8_t donor_roles = 0;   // битмаска donor_bit(DonorRole); 0 <=> kCOnly/kWrapup
};
```

**Дескрипторы конкретных потенциалов НЕ меняются** — даже `EAM Density` остаётся `kCOnly` (классификационная таблица аудита §3.4 — ЦЕЛЬ v1; флип — только когда кольцо начнёт потреблять, PR-1/2). Замок — зуб T-INERT (§9).

### 2.1. validate_pass_decls — машинно-проверяемые firewall-зубы

Факт as-built (несущий для правил): `accum_fracbits` СЕГОДНЯ не имеет рантайм-потребителя (кольца берут fb из `Math::density_fracbits()` — eam_ring.hpp:109); Tersoff BondOrder fb=0 и MEAM Embedding fb=0 несут ноль как ДОКУМЕНТАЦИЮ FP-фолда. `validate_pass_decls` превращает эту конвенцию из комментария в машинный инвариант, ничего не требуя ретроактивно (все shipped-пассы kCOnly — валидны как есть).

```cpp
// PR-0a firewall-зубы (аудит §3.3, MUST-FIX вшит). САМОсогласованность дескриптора,
// независимая от политики WinForce; кольца зовут её безусловно ПЕРЕД
// WinForce::assert_supported (см. §6); assert_eam_symmetric_passes зовёт её сам.
//
// ЧЕСТНАЯ ОГОВОРКА (обязательная, не переформулировать): машинная проверка ловит
// ТОЛЬКО zeta-класс (kAccumB1 объявлен на пассе, чья редукция — FP: маркеры
// accum_fracbits==0 / kind==BondOrder). Canon-dependent-addend-класс — int64-лейны
// MEAM, чьи double-адденды зависят от S_ij=P_k (порядко-чувствительное произведение
// ДО квантования) — машинно НЕ ловится: такой пасс легально несёт fracbits>0 и
// kind==Density. Его свидетели — только состязательный дизайн классифицирующего PR
// + независимый FP64-оракул (*_direct_fp64, класс MB2). Не расширять претензии
// kAccumB1 за пределы того, что доказывает эта функция.
inline void validate_pass_decls(std::span<const PassDecl> passes) {
  if (passes.empty() || passes.size() > 8)
    throw std::runtime_error("validate_pass_decls: 1..8 passes "
        "(closure_bit packs 4 roles x 8 passes into uint32)");
  for (std::size_t p = 0; p < passes.size(); ++p) {
    const PassDecl& d = passes[p];
    if (d.donor_roles & ~kDonorRolesValidMask)
      throw std::runtime_error("validate_pass_decls: reserved donor_roles bit set (pass "
          + std::to_string(p) + ")");
    if (d.iterative && d.w_class != WClass::kCOnly)
      throw std::runtime_error("validate_pass_decls: iterative pass (QEq/CG) cannot be "
          "donated (pass " + std::to_string(p) + ")");
    if (d.w_class == WClass::kAccumB1) {
      if (d.accum_fracbits <= 0)
        throw std::runtime_error("validate_pass_decls: kAccumB1 requires accum_fracbits>0 "
            "(int64 FixedAccum is WHAT makes donation batches order-free, lemma W-1); "
            "an FP-reduction pass cannot be kAccumB1 (pass " + std::to_string(p) + ")");
      if (d.kind == PassKind::BondOrder)
        throw std::runtime_error("validate_pass_decls: kAccumB1 forbidden on BondOrder "
            "(zeta is an FP64 fold by design — tersoff.hpp; the canon class is "
            "kAccumCanonSuffix) (pass " + std::to_string(p) + ")");
      if (d.donor_roles == 0)
        throw std::runtime_error("validate_pass_decls: kAccumB1 with empty donor_roles "
            "(nothing would ever close the ledger) (pass " + std::to_string(p) + ")");
    } else if (d.donor_roles != 0) {
      // СТРОГО в v1 (осознанно, К4): roles разрешены ТОЛЬКО на kAccumB1. Канон-донации
      // (kAccumCanonSuffix) не спроектированы — их PR ослабит это правило вместе со
      // своей семантикой и зубами. kAccumB1+roles==0 тоже throw (вероятная опечатка);
      // первый PR-потребитель может ослабить осознанно.
      throw std::runtime_error("validate_pass_decls: donor_roles set on a non-kAccumB1 "
          "pass (pass " + std::to_string(p) + ")");
    }
  }
}
```

Правило BondOrder НЕ ретроактивно: Tersoff BondOrder `kCOnly` fb=0 — валиден; MEAM/EAM Embedding — валидны. `kAccumCanonSuffix` без ролей validate ПРИНИМАЕТ (легальное будущее значение; гейт потребителей — в canon-PR).

---

## 3. donation_layout — единый источник расписания донаций (`eam_zone.hpp`)

Ставится **непосредственно под `eam_window_layout`** (:70-83) — буква аудита §3.1 и прецедент E5b-F4; все четыре кольца + GPU-драйвер уже включают этот хедер (`eam_ring.hpp:22`, `sw_ring.hpp:23`, `tersoff_ring.hpp:24`, `meam_ring.hpp:24`, `eam_conveyor_gpu.cuh:13`) ⇒ нулевой include-churn у потребителей PR-1/2. `eam_zone.hpp` уже включает `eam.hpp` (:11) и `many_body.hpp` транзитивно.

**Семантика (слот-пространство, PASS-инвариантно).** `donation_layout(j,n,pbc)` возвращает батчи, исполняемые на скан-позиции j — в точке `eam_ring.hpp:386→387`: ПОСЛЕ `ensure_arrival(j+1)` (:385) + `ensure_drift(j-1,j,j+1)` (:386), СТРОГО ДО `finalize_owned(j)` (:387). Внутри позиции: сначала self-батчи, затем cross. Объединение по j=0..n−1 — полное расписание §3.1, каждый батч ровно один раз (обобщение INV-8, ZoneFSM:106).

```cpp
// === PR-0a (W-контракт, аудит §3.1) — единый источник истины расписания донаций.
// ЧИСТАЯ функция; в PR-0a её НЕ потребляет ни одно кольцо (потребители — PR-1/2).
// Контракт (SPEC; нарушение любого пункта = регрессия MUST-FIX аудита):
//  (1) СЛОТ-пространство прохода: slot = позиция прибытия; label=(r+slot)%n,
//      r=(h-1)%n (eam_ring.hpp:182,204; ZoneFSM §7.2; хелперы pass_rotation/
//      slot_zone_id ниже — ЕДИНЫЙ источник формул). Расписание PASS-инвариантно.
//      ВСЕ будущие ПЕРСИСТЕНТНЫЕ структуры (rho-аккумуляторы, CSR, эпохи,
//      device-зеркала) ключуются по zone_id, НЕ по слоту; зуб PR-2 — PBC-прогон
//      >=2 полных оборотов ротации, n>=5, побитово.
//  (2) self(k): предусловие — false->true фронт ensure_drift(k) с dt прохода h
//      (НЕ «на RECV»: dt приезжает с arrival 0, drift ленивый — eam_ring.hpp:219,
//      231-246). Идемпотентность drift — несущая (finalize_owned:314 self-дрейфит
//      wrap-членов повторно).
//  (3) cross(a,b): на CoRes(a,b) — оба слота прибыли И дрейфнуты.
//  (4) pbc-шов cross(n-1,0): доступен с CoRes(slot n-1, slot 0) = прибытия
//      ПОСЛЕДНЕЙ зоны (ready=n-2), обязан исполниться СТРОГО ДО finalize_owned(n-1),
//      который идёт IN-SCAN (eam_ring.hpp:387-388 — хвост defer_head откладывает
//      ТОЛЬКО finalize(0)+отправки, :390-396). Здесь это закодировано СТРУКТУРНО:
//      шов эмитится на j=n-1, а по контракту вызова батчи позиции j исполняются
//      ДО finalize_owned(j). Вариант «в хвосте прохода» ОПРОВЕРГНУТ верификатором
//      аудита — не воскрешать. Микроточку ВНУТРИ окна позиции (arrival-вариант vs
//      внутри finalize(k-1) после START — открытый вопрос аудита §8.1) эта функция
//      НЕ предрешает: оба кандидата лежат в окне «после drift, до finalize» той же
//      позиции; выбор — состязательный дизайн PR-1.
//  (5) Ровно один батч на неупорядоченную пару соседних зон за проход + ровно
//      один self на слот (обобщение INV-8).
//  (6) Донационные аккумуляторы и леджер строго pass-scoped (сброс на RECV —
//      fsm.hpp:47); mid-pass HALT отбрасывает проход ЦЕЛИКОМ, частичный реплей
//      батчей запрещён. Донации НЕ трогают min_r2/PE/phi-once.
//  (7) LOAD-BEARING предпосылка полноты «только смежные рёбра»: страж
//      membership_ok с g = 0.5*(width - 2*rcut) (eam_ring.hpp:441-459).
//      PR, ослабляющий g или меняющий членство, ОБЯЗАН пере-дериваировать эту
//      функцию (мутационный зуб с дрейф-фикстурой — PR-1).
//  (8) Домен: free — любое n>=1; pbc — n==1 (вырождение в free-путь, шва нет)
//      или n>=5 (ZoneDecomposition::build отвергает periodic 2..4 при
//      reach_mult=2). pbc n in [2,4] -> throw — ЯВНЫЙ страж (eam_window_layout
//      молча дублирует слоты; здесь PR-1 ошибиться не может).
struct DonationBatches {
  int n_self = 0, n_cross = 0;
  int self[2] = {-1, -1};                  // слоты self(k); исполняются ПЕРВЫМИ
  int cross[2][2] = {{-1, -1}, {-1, -1}};  // неупорядоченные пары {a,b}; после self
};

inline DonationBatches donation_layout(int j, int n, bool pbc) {
  if (n < 1 || j < 0 || j >= n)
    throw std::invalid_argument("donation_layout: j out of [0,n)");
  if (pbc && n >= 2 && n <= 4)
    throw std::invalid_argument("donation_layout: periodic n_zones 2..4 rejected (reach_mult=2)");
  DonationBatches b;
  if (n == 1) { b.self[b.n_self++] = 0; return b; }   // free-путь; шва нет
  if (j == 0) {                                       // scan 0: слоты 0,1 прибыли+дрейфнуты
    b.self[b.n_self++] = 0;
    b.self[b.n_self++] = 1;
    b.cross[0][0] = 0; b.cross[0][1] = 1; b.n_cross = 1;
    return b;
  }
  if (j + 1 < n) {                                    // 1 <= j <= n-2
    b.self[b.n_self++] = j + 1;                       // новый дрейф этого скана
    b.cross[0][0] = j; b.cross[0][1] = j + 1; b.n_cross = 1;
    return b;
  }
  if (pbc) { b.cross[0][0] = n - 1; b.cross[0][1] = 0; b.n_cross = 1; }  // (4): шов ДО finalize(n-1)
  return b;
}

// Хелперы slot<->label — единый источник формул для персистентных структур PR-1/2
// (формулы = замороженные конвенции кольца, eam_ring.hpp:182,204; зуб T-ROT).
inline int pass_rotation(long h, int n, bool pbc) { return pbc ? int((h - 1) % n) : 0; }
inline int slot_zone_id(int r, int slot, int n) { return (r + slot) % n; }
```

**Нормативная таблица ready/deadline (документация + вход зуба T-SIM; графт из B).** Выведена из scan-loop `eam_ring.hpp:373-401`: arrival слота s — к скану max(0,s−1); первый drift слота s — скан max(0,s−1); finalize(j) in-scan для всех j, кроме pbc j=0 → хвост (defer_head).

| Батч | эмиссия donation_layout | ready_scan | deadline_scan (первый in-scan finalize-читатель) |
|---|---|---|---|
| self(k) | max(0,k−1) | max(0,k−1) | free: max(0,k−1); pbc: k≤1→1 (finalize(0) отложен), k≥2→k−1 |
| cross(k,k+1), k≤n−2 | k | k | free: k; pbc: k=0→1, иначе k |
| **шов cross(n−1,0)** (pbc) | **n−1** | n−2 | **n−1** ← вшитый MUST-FIX §3.1 |

Инвариант: `эмиссия ∈ [ready, deadline]` всюду (эмиссия = ready для всех батчей, кроме шва, где эмиссия = deadline с in-position порядком «батчи ДО finalize»). Хвостовой finalize(0) (defer_head) читает рёбра (n−1,0) и (0,1) — оба эмитированы к позиции n−1 ⇒ покрытие держится. Зуб T-SIM (§9) перепроверяет таблицу НЕЗАВИСИМОЙ симуляцией событий скана, не доверяя ни ей, ни коду.

---

## 4. want_mask / леджер (`many_body.hpp` — биты; `eam_zone.hpp` — геометрия)

Хранилище НЕ добавляется: `core::Zone::contrib_mask` (uint32, `zone.hpp:31`) со сбросом на RECV/SEND (`fsm.hpp:47,51`) — готовый pass-scoped INV-3-леджер; пар-конвейер уже пишет/проверяет его (`conveyor.hpp:527,532` / `:431-434`). PR-0a поставляет только чистую want-сторону + примитив пометки.

```cpp
// many_body.hpp — раскладка бита леджера полноты (рядом с полем donor_roles):
// uint32 = 4 роли x до 8 пассов; расширение INV-3-леджера contrib_mask.
inline constexpr uint32_t closure_bit(int pass_idx, DonorRole role) {
  return 1u << (unsigned(pass_idx) * 4u + unsigned(role));
}
// Пометка cross-батча В ОБЕ маски (единственное место, где PR-2 ошибся бы вручную):
// CrossHi у нижней зоны ребра, CrossLo у верхней.
inline constexpr void close_cross_batch(uint32_t& mask_lo, uint32_t& mask_hi, int pass_idx) {
  mask_lo |= closure_bit(pass_idx, DonorRole::kCrossHi);
  mask_hi |= closure_bit(pass_idx, DonorRole::kCrossLo);
}
```

```cpp
// eam_zone.hpp (под donation_layout) — END-ожидание зоны в слоте j.
// КОНВЕНЦИЯ ВАКУУМНОГО БАТЧА (аудит §3.3 MUST-FIX): бит леджера значит «батч
// ИСПОЛНЕН (возможно, вакуумно)», НЕ «внёс >=1 пару». Пустая зона (n()==0
// легально резидентна — eam_ring.hpp:318) ставит биты вакуумно — иначе END
// голодает. Want-сторона по построению не зависит от населённости (параметра
// населённости у неё НЕТ).
// РАЗГРАНИЧЕНИЕ (PR-2 не смешивать): want-set ПЛАНИРОВАНИЯ finalize(j) — это
// самости всех wслотов окна + внутренние рёбра (j-1,j),(j,j+1) (закрываются
// deadline-планированием, зуб T8/T-SIM); want_closure_mask(j) — БУХГАЛТЕРИЯ END
// самой зоны j (только её роли self/lo/hi) — прецедент пар-конвейера
// (conveyor.hpp:427-434). Точка будущей проверки (PR-2, НЕ здесь): end_eam
// непосредственно перед ZoneFSM::apply(END) (eam_ring.hpp:299).
inline uint32_t want_closure_mask(std::span<const potentials::PassDecl> passes,
                                  int j, int n, bool pbc) {
  potentials::validate_pass_decls(passes);
  (void)donation_layout(j, n, pbc);   // общий доменный страж (throw на pbc 2..4 / j вне)
  const bool cyc    = pbc && n > 1;
  const bool has_lo = cyc || j > 0;
  const bool has_hi = cyc || j + 1 < n;    // рёбра отпадают на free-z границах,
  uint32_t w = 0;                          // ровно как eam_window_layout роняет слоты
  for (std::size_t p = 0; p < passes.size(); ++p) {
    if (passes[p].w_class != potentials::WClass::kAccumB1) continue;
    const uint8_t dr = passes[p].donor_roles;
    using potentials::DonorRole; using potentials::donor_bit; using potentials::closure_bit;
    if (dr & donor_bit(DonorRole::kSelf))                w |= closure_bit(int(p), DonorRole::kSelf);
    if (has_lo && (dr & donor_bit(DonorRole::kCrossLo))) w |= closure_bit(int(p), DonorRole::kCrossLo);
    if (has_hi && (dr & donor_bit(DonorRole::kCrossHi))) w |= closure_bit(int(p), DonorRole::kCrossHi);
  }
  return w;   // все shipped-дескрипторы (сплошной kCOnly) -> 0 всюду = F-NOOP-якорь
}
```

---

## 5. eam.hpp: единый источник EAM-дескриптора + общий симметричный гейт

Член `EamPotential<Real,Math>::kPasses` — пер-инстанциационный и недостижим без Math-типа ⇒ GAP-гейту (§7) нужен namespace-scope источник. Grep подтвердил: `kPasses` EAM нигде не используется вне `eam.hpp` ⇒ рефактор локален (К2).

```cpp
// (namespace tdmd::potentials, перед EamPotential)
// PR-0a: ЕДИНЫЙ источник EAM-дескриптора (нужен eam_gpu_run_singlenode-гейту, у
// которого нет объекта потенциала). EamPotential::passes() возвращает ЕГО ЖЕ —
// двух копий истины нет (зуб T-SRC).
inline constexpr PassDecl kEamPassDecls[3] = {
    {PassKind::Density, /*reuse*/ true, false, false, 44},
    {PassKind::Embedding, /*reuse*/ false, false, false, 30},  // local map
    {PassKind::Force, /*reuse*/ true, false, false, 40},
};
inline constexpr std::span<const PassDecl> eam_pass_decls() { return {kEamPassDecls, 3}; }

// PR-0a: общий гейт «симметричный EAM [Density,Embedding,Force]» — семантика
// GpuEamWindowForce::assert_supported (eam_window_force_gpu.cuh:263-283) дословно;
// тексты сообщений сохранены, префикс who. Потребители: GpuEamWindowForce (ДЕЛЕГАЦИЯ,
// К3 — одна истина вместо parity-полисинга), CpuEamWindowForce (новый),
// eam_gpu_run_singlenode. Внутри зовёт validate_pass_decls (К9) => каждый EAM-вход
// отвергает и wclass-нелегальные дескрипторы.
inline void assert_eam_symmetric_passes(std::span<const PassDecl> passes, const char* who) {
  validate_pass_decls(passes);
  auto fail = [&](std::string m) { throw std::runtime_error(std::string(who) + ": " + std::move(m)); };
  if (passes.size() != 3)
    fail("only the EAM 3-pass (Density,Embedding,Force) sequence is supported — got "
         + std::to_string(passes.size()) + " passes");
  const PassKind want[3] = {PassKind::Density, PassKind::Embedding, PassKind::Force};
  for (std::size_t p = 0; p < 3; ++p) {
    if (passes[p].kind != want[p]) fail("unexpected pass kind at " + std::to_string(p));
    if (passes[p].needs_transpose)
      fail("needs_transpose UNSUPPORTED — the symmetric int64 accumulator q(j)=-q(i) cannot "
           "run a non-symmetric angular/bond-order term (force to a third atom k)");
    if (passes[p].iterative) fail("iterative pass (QEq/CG) UNSUPPORTED");
  }
}
```

В `EamPotential`: `std::span<const PassDecl> passes() const override { return {kEamPassDecls, 3}; }`; член :105-109 и out-of-line :145 удалить. `sw.hpp`/`tersoff.hpp`/`meam.hpp` члены `kPasses` НЕ трогаются (файлы — байт-свидетели). `GpuEamWindowForce::assert_supported` (eam_window_force_gpu.cuh:263-283) → однострочная делегация `assert_eam_symmetric_passes(passes, "GpuEamWindowForce")`; honest-scope блок :254-262 переписывается фактом с датой: «оба поименованных зазора ЗАКРЫТЫ PR-0a (гейт драйвера + concept)».

---

## 6. Concept `WindowForcePolicy` + обязательный firewall в кольцах

Дискредитируемая дыра (урок аудита «opt-in обходится молча», `eam_window_force_gpu.cuh:257-258`; живой пример — CPU-EAM: `CpuEamWindowForce` без `assert_supported` ⇒ `if constexpr requires` мёртв на `eam_ring.hpp:117-119`, в отличие от MEAM, где CPU-политика гейт имеет). Обещание кода «promote the policy contract to a C++20 concept + static_assert» — просрочено с MEAM; PR-0a исполняет его.

```cpp
// many_body.hpp (после PassDecl/validate_pass_decls)
// PR-0a: контракт WinForce-политики, ПРОМОТИРОВАН из duck-typing (обещание MB1/MB2
// из eam_window_force_gpu.cuh). Требуется:
//  - copy_constructible (кольца хранят политику по значению; trivially_copyable НЕ
//    требуется — GPU-политики держат shared_ptr device-состояния);
//  - static assert_supported(span<PassDecl>) — ОБЯЗАТЕЛЬНЫЙ: убивает молчаливый
//    обход `if constexpr requires` (политика с опечаткой — теперь ошибка КОМПИЛЯЦИИ);
//  - ЗАМОРОЖЕННЫЙ 14-арг const compute() (rho_cap — 9-й параметр, у SW/Tersoff/MEAM
//    named-discarded; НЕ менять).
// СКОУП PR-0a (честно): хуки on_zone_arrival/on_edge из аудита §3.3 войдут в концепт
// ВМЕСТЕ со своими потребителями (PR-1/2) — обязательный-но-никем-не-зовущийся хук
// сегодня был бы structurally-dead кодом (5 прецедентов-рецидивов); расширение
// концепта — компайл-тайм-принуждение, забыть его позже невозможно.
template <typename WF>
concept WindowForcePolicy =
    std::copy_constructible<WF> &&
    requires(const WF wf, const double* d, const long* k, int i, const int* ip,
             const core::PairGeom& g, double rc,
             std::vector<core::fixed::ForceAccum>& f,
             core::fixed::EnergyAccum& pe, double& mr,
             std::span<const PassDecl> ps) {
      { WF::assert_supported(ps) } -> std::same_as<void>;
      { wf.compute(d, d, d, k, i, ip, i, g, rc, f, f, f, pe, mr) } -> std::same_as<void>;
    };
```

**Правка всех 4 колец** (идентичная): (а) class-scope (К5):

```cpp
  static_assert(potentials::WindowForcePolicy<WinForce>,
      "WinForce must model WindowForcePolicy (static assert_supported + 14-arg const "
      "compute) — the opt-in `if constexpr requires` firewall was silently bypassable "
      "(PR-0a; see many_body.hpp)");
```

(б) в run() замена идиомы (`eam_ring.hpp:113-119`, `sw_ring.hpp:174-175`, `tersoff_ring.hpp:211-212`, `meam_ring.hpp:221-222`):

```cpp
    potentials::validate_pass_decls(pot_.passes());   // W-зубы дескриптора (§2.1)
    WinForce::assert_supported(pot_.passes());        // безусловно — концепт гарантирует наличие
```

**Индуцированные обязательные добавки (без них не компилится — это и доказывает зубастость):**
1. `CpuEamWindowForce::assert_supported` (`eam_ring.hpp:66-84`): `static void assert_supported(std::span<const potentials::PassDecl> p) { assert_eam_symmetric_passes(p, "CpuEamWindowForce"); }` — CPU-политика гоняет ТОТ ЖЕ симметричный `eam_window_force`, accept-набор ≡ GPU. На легальном EAM — no-op ⇒ F-NOOP (свидетель — `Test_EAM_Ring` байт-в-байт).
2. Test-only ядовитые политики `SwWinForceLocalKey` (`sw_ring.hpp:121`) / `TersoffWinForceLocalKey` (`tersoff_ring.hpp:158`): однострочная делегация `static void assert_supported(std::span<const potentials::PassDecl> p) { SwWinForce<Real>::assert_supported(p); }` (соотв. Tersoff). Их яд — в compute, не в гейте; поведение тестов байт-идентично.
3. **Шаг имплементера (обязательный):** `grep -rn "EamRing<\|SwRing<\|TersoffRing<\|MeamRing<" include tests tools` — каждая политика-аргумент обязана моделировать концепт; любая неучтённая получает делегирующий `assert_supported` (поведение-сохраняющий). Известный полный список: CpuEamWindowForce (новый), 2 LocalKey (делегация), 4 GPU-политики + CPU Sw/Tersoff/Meam WinForce (уже имеют).

5 существующих тест-пинов `static_assert(requires{...})` — НЕ трогаются (К6): class-scope концепт субсуммирует их; облачную проверку GPU-политик несёт `cuda_compile_check.cu` (§8). Анти-регресс-замечание: после промоушена мутация «вернуть if-constexpr» безвредна (концепт гарантирует наличие ⇒ ветка всегда true); несущий элемент — class-scope static_assert, его зуб — T-CPT-негатив.

---

## 7. Закрытие FIREWALL GAP: `eam_gpu_run_singlenode` (`eam_conveyor_gpu.cuh:114-128`)

Констатация (проверено): драйвер не имеет НИКАКОГО дескриптора (единственный potential-параметр — сырой `EamSetfl`, дескриптор живёт на `IManyBodyPotential::passes()`); 8 колл-сайтов, все EAM-легальные (tests/test_cuda_eam_ring.cu:111,155,161,340; test_cuda_eam_rdf.cu:106; tools/eam_drift.cu:94,103; eam_rdf_stat.cu:122; eam_coexist.cu:146). `symmetric=false` (:125) — Oracle-A poison, форма ОКНА, ортогональна дескриптору — обязана работать как прежде. Include-цикл исключён конструкцией: гейт в `eam.hpp`, уже в include-графе (`eam_conveyor_gpu.cuh:13 → eam_zone.hpp:11 → eam.hpp`); имплементеру ЗАПРЕЩЕНО включать `eam_window_force_gpu.cuh` из `eam_conveyor_gpu.cuh` (циклический include).

**Ключевое анти-structurally-dead решение:** гейт по вкомпилированной константе никогда не бросает ⇒ мёртв и нетестируем (5-раз-наступленные грабли: Te1 fc_d, Me1 partial-S, Me5 taper, Me5b OOB). Поэтому дескриптор — **хвостовой параметр с дефолтом** = единый источник §5; все 8 вызовов не меняются ни байтом, а зуб получает живую ручку.

```cpp
// FIREWALL GAP — CLOSED (PR-0a; обещание «gate the single-node driver when MEAM
// lands» из eam_window_force_gpu.cuh, MB1). МАНДАТОРНЫЙ дескриптор-гейт ПЕРВОЙ
// строкой (НЕ `if constexpr` opt-in), ДО любой device-работы: копипаст-реюз этого
// драйвера под MEAM/Tersoff (needs_transpose) теперь бросает, а не молча гонит
// симметричный аккумулятор. `passes` по умолчанию = единый источник kEamPassDecls;
// параметр существует, чтобы (а) реюзер/генерализация НАСЛЕДОВАЛИ гейт, а не
// обходили его, (б) зуб гейта был non-vacuous (poison-дескриптор -> throw, T-GAP).
// symmetric=false (Oracle-A poison) — форма ОКНА, ортогональна дескриптору,
// работает как прежде. ЧЕСТНО (не переоверклеймить): первичная защита остаётся
// типовой (в функцию нельзя подать MeamParams); ценность — инжект-зуб + mandatory-
// паттерн + снятие просроченного обещания.
template <typename Real>
void eam_gpu_run_singlenode(core::AtomSoA<Real>& a, const core::Box& box,
                            const core::ZoneDecomposition& zd,
                            const potentials::EamSetfl<double>& setfl, long steps,
                            double dt, bool symmetric = true,
                            std::vector<double>* out_fx = nullptr,
                            std::vector<double>* out_fy = nullptr,
                            std::vector<double>* out_fz = nullptr,
                            std::span<const potentials::PassDecl> passes =
                                potentials::eam_pass_decls()) {
  potentials::assert_eam_symmetric_passes(passes, "eam_gpu_run_singlenode");
  using namespace eam_sn_detail;
  ...  // тело байт-неизменно
```

Комментарий-якорь :114-120 переписать на «CLOSED (PR-0a)».

---

## 8. CI compile-surface TU (`tools/cuda_compile_check.cu`)

Факт (verified): облачная cuda-compile джоба (ci.yml:140-148) конфигурится `-DTDMD_BUILD_TESTS=OFF`; `{sw,tersoff,meam}_window_force_gpu.cuh` включаются ТОЛЬКО тестовыми TU; `GpuTersoffRing`/`GpuMeamRing` — alias-темплейты (ничего не инстанцируют), у SW alias-а нет вовсе; EAM-политика уже покрыта (`bench_eam_ring.cu:34`). **Форма — исполняемый с тривиальным main** (дом-паттерн tools; ловит и device-link/nvlink-класс, который OBJECT-библиотека пропускает). **Сила — explicit instantiation definitions (К7)** с измеренным фолбэком.

```cpp
// tools/cuda_compile_check.cu — PR-0a, аудит §7 п.2 (CI-слепота): облачный
// cuda-compile идёт с TDMD_BUILD_TESTS=OFF, а угловые GPU-политики + Gpu*Ring
// инстанциации включались ТОЛЬКО тестовыми TU. Этот TU компилирует их в облаке.
// Explicit instantiation НАМЕРЕННО (сильнее тестовой поверхности — инстанцирует
// ВСЕ члены): поломка любого члена кольца в PR-1..4 ловится CI, а не следующим
// локальным GPU-прогоном. Ничего не исполняет.
#include <cstdio>
#include "tdmd/cuda/sw_window_force_gpu.cuh"
#include "tdmd/cuda/tersoff_window_force_gpu.cuh"
#include "tdmd/cuda/meam_window_force_gpu.cuh"
#include "tdmd/cuda/eam_window_force_gpu.cuh"   // + транзитивно eam_conveyor_gpu.cuh

namespace tp = tdmd::potentials;
namespace tc = tdmd::cuda;

// Концепт-зубы GPU-политик — проверяются В ОБЛАКЕ (тестовые TU облако не видит).
static_assert(tp::WindowForcePolicy<tc::GpuSwWinForce<double>>);
static_assert(tp::WindowForcePolicy<tc::GpuTersoffWinForce<double>>);
static_assert(tp::WindowForcePolicy<tc::GpuMeamWinForce<double>>);
static_assert(tp::WindowForcePolicy<tc::GpuEamWindowForce>);

template class tp::SwRing<double, tc::GpuSwWinForce<double>>;
template class tp::TersoffRing<double, tc::GpuTersoffWinForce<double>>;   // = GpuTersoffRing
template class tp::MeamRing<double, tc::GpuMeamWinForce<double>>;         // = GpuMeamRing
template class tp::EamRing<double, tp::EamSetfl<double>, tc::GpuEamWindowForce>;
template void tc::eam_gpu_run_singlenode<double>(
    tdmd::core::AtomSoA<double>&, const tdmd::core::Box&,
    const tdmd::core::ZoneDecomposition&, const tp::EamSetfl<double>&, long, double,
    bool, std::vector<double>*, std::vector<double>*, std::vector<double>*,
    std::span<const tp::PassDecl>);

int main() { std::puts("tdmd cuda compile surface: OK"); return 0; }
```

**Фолбэк (записать в PR-текст, если сработал):** если КАКОЙ-ТО член кольца не компилируется под GPU-политикой (никогда не инстанцировался) — это само по себе находка; для проблемного кольца заменить explicit instantiation на odr-use тестовой поверхности (never-run функция за недостижимым runtime-гардом, конструирующая политику и зовущая `run_*_ring` — построчно зеркаля `test_cuda_sw_ring.cu:40` / sibling'ов), находку зафиксировать. Решается первым локальным билдом (measure-first).

**CMake** (в tools-блок `if(TDMD_WITH_CUDA)`; **ci.yml не трогается** — джоба билдит ALL при TESTS=OFF, канарейка приёмки — лог обеих ног матрицы mpi={OFF,ON} содержит компиляцию TU):

```cmake
  # PR-0a (аудит §7 п.2): CI compile-surface — угловые GPU-политики + Gpu*Ring
  # инстанциации компилируются в облачном cuda-compile (TDMD_BUILD_TESTS=OFF
  # пропускал их). Никогда не исполняет device-код. --fmad=false per-target
  # (tdmd_eam_cuda_flags определён внутри tests-блока :415 — НЕ поднимать).
  add_executable(cuda_compile_check tools/cuda_compile_check.cu)
  target_link_libraries(cuda_compile_check PRIVATE tdmd_core Threads::Threads)
  target_compile_options(cuda_compile_check PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
```

---

## 9. План тестов и таблица kill-мутаций (каждый зуб non-vacuous)

**Новый CPU-таргет `test_w_contract`** (`tests/test_w_contract.cpp`, линк `tdmd_core GTest::gtest_main`, регистрация зеркалом `test_metrics` CMakeLists:355-357):

| # | Тест | Проверяет | Kill-мутация (тест ОБЯЗАН упасть) |
|---|---|---|---|
| T1 | `ValidateRejectsAccumB1WithoutFracbits` | kAccumB1+fb=0 → throw | удалить правило fb>0 |
| T2 | `ValidateRejectsAccumB1OnBondOrder` | ζ-класс FP-редукции → throw | удалить правило BondOrder |
| T3 | `ValidateGuards` | size>8; бит kReserved3; roles≠0 на kCOnly; kAccumB1+roles==0; iterative+kAccumB1 | удалить любой из 5 стражей — соответствующий EXPECT_THROW падает |
| T-VAC | `ValidateAcceptsAllShippedDescriptors` | позитив: passes() всех 4 потенциалов + `eam_pass_decls()` проходят validate + легальный синтетический kAccumB1{Density,fb=44,roles=0x07} | вырождение validate в «всегда бросает» (анти-вакуумность T1-T3) |
| T-INERT | `ShippedDescriptorsAreInert` | цикл по passes() EAM/SW/Tersoff/MEAM: `w_class==kCOnly && donor_roles==0`; компайл-тайм `static_assert` на 5-элементном агрегате `PassDecl{...}`: хвостовые поля = дефолты | флип любого shipped-поля / вставка поля в СЕРЕДИНУ PassDecl |
| T4 | `DonationLayoutExactlyOncePerPair` | ∀n∈1..8 free, n∈{1,5..8} pbc: объединение по j = {self(k)}∀k ∪ {cross(k,k+1)}k<n−1 ∪ pbc{cross(n−1,0)}, каждый РОВНО раз (обобщение INV-8) | дубль/потеря любого батча |
| **T5** | `PbcSeamFiresBeforeFinalizeN1` ⭐ зуб MUST-FIX §3.1 | pbc n≥5: пара {n−1,0} эмитится ровно на j=n−1 и нигде больше; карта fin_pos (fin_pos(j)=j; fin_pos(0)=«хвост» при defer_head): emit_pos(шва) ≤ fin_pos(n−1) при документированном порядке «батчи ДО finalize внутри позиции» | перенос эмиссии шва «в хвост» / невыдача — ровно опровергнутый верификатором вариант |
| T5b | `DonationLayoutFeasibility` | ∀j: эмитированные self ⊆ {0,1}∪{j+1} ⊆ прибывшие {0..j+1}; endpoints всех cross ⊆ бегущему объединению self до j включительно (предусловия (2)/(3) машинно) | эмиссия self(j+2) / cross до self endpoint'а |
| T6 | `FreeEdgesDropAndN1AndSmallPbcThrow` | free j=n−1 → пусто; n=1 (free и pbc) → один self(0); pbc n∈{2,3,4} → throw | эмиссия шва в free / снятие pbc-стража |
| **T-SIM** | `ScheduleConsistentWithScanSimulation` ⭐ (графт B, главный анти-off-by-one) | независимая мини-симуляция событий скана ПО СПЕКЕ (arrival max(0,s−1), drift max(0,s−1), finalize in-scan, defer_head+хвост — НЕ по коду кольца): у каждого батча предусловия выполнены на позиции эмиссии; первый читающий finalize каждого батча — на позиции ≥ эмиссии (шов: РОВНО на ней, с in-position порядком); свип n=1..16 × {free, pbc n∈{1,5..16}}; сверка с нормативной таблицей §3 | любой ±1 в эмиссии/таблице |
| T8 | `ConsistentWithEamWindowLayout` | ∀(j,n,pbc) в домене: самости wслотов `eam_window_layout(j,n,pbc)` и рёбра (j−1,j),(j,j+1) внутри окна закрыты к fin_pos(j); ребро (j+1,j+2) НЕ требуется (L-CLOSE-открытость) | требование (j+1,j+2) / потеря self(j−1) |
| T9 | `WantMaskAllCOnlyIsZero` | все shipped-дескрипторы → want==0 ∀(j,n,pbc) — F-NOOP-якорь want-стороны | любой бит из kCOnly-пасса / вывод маски из kind вместо w_class |
| T10 | `WantMaskSyntheticAccumB1` | синтетический [Density kAccumB1 roles=self\|lo\|hi fb=44]: интерьер → 3 бита; free j=0 без lo; free j=n−1 без hi; pbc — все 3 на каждом j; pbc n=1 — только self; позиции битов проверены и через `closure_bit`, И через сырые константы (двойная запись ловит мутацию упаковки); 2 kAccumB1-пасса → разные нибблы; >8 пассов → throw (через validate) | смена упаковки (4↔8, role↔pass) / потеря edge-dropping |
| T-CROSS | `CloseCrossBatchMarksBothZones` (графт B) | `close_cross_batch` ставит CrossHi у lo-зоны и CrossLo у hi-зоны | зеркальная путаница ролей |
| T-ROT | `RotationHelpersMatchRingConvention` (графт B) | `pass_rotation`/`slot_zone_id`: замороженные значения на 2 полных оборота, n=5 (h=1..11), free r=0 | `h%n` вместо `(h−1)%n` и т.п. (сид ротационного зуба PR-2) |
| T-EAM | `AssertEamSymmetricPassesTeeth` | accept `eam_pass_decls()`; reject: count 2/4, чужой kind, needs_transpose и iterative на КАЖДОМ слоте (цикл 0..2 — анти-off-by-one, паттерн test_cuda_eam_ring.cu:568-574); wclass-нелегальный-но-shape-легальный (Embedding.donor_roles=1) → throw ЧЕРЕЗ гейт (докво проводки validate внутри гейта, экс-Z7) | выпотрошить любую ветку гейта / убрать вызов validate из гейта |
| T-SRC | `EamPassDeclsSingleSource` | `EamPotential<double,EamSetfl<double>>(…).passes().data() == kEamPassDecls` | ре-дупликация дескриптора |
| T-CPT | `ConceptTeeth` (компайл-тайм) | позитив: `static_assert(WindowForcePolicy<CpuEamWindowForce<EamSetfl<double>>>)` + CPU Sw/Tersoff/Meam-политики + 2 LocalKey; НЕГАТИВ: локальные `BadPolicyNoAssert` (только compute), `BadPolicyWrongSign` (нестатический assert_supported), `BadPolicyWrongArity` → `static_assert(!WindowForcePolicy<…>)` | ослабление концепта (выкинуть assert_supported/const/арность) — ровно «silent bypass», ради которого промоушен |

**CUDA-добавка** (в существующий `tests/test_cuda_eam_ring.cu`, метка cuda — счётчик CUDA-таргетов остаётся 58; гоняется `gpu_gate.sh`):

| # | Тест | Проверяет | Kill-мутация |
|---|---|---|---|
| T-GAP | `SingleNodeDriverFirewallGate` | steps=0, малая фикстура: дефолтный вызов НЕ бросает (позитив уже покрыт :155/:161 — symmetric=true/false); копии kEamPassDecls с needs_transpose=true (цикл по слотам), count=4, kAccumB1+fb=0, Embedding.donor_roles=1 → EXPECT_THROW ДО device-работы | удалить гейт из головы `eam_gpu_run_singlenode` — все poison-EXPECT_THROW падают |

**Wiring-цепочка колец** (почему «кольцо ЗОВЁТ гейт» доказано без нового рантайм-зуба): class-scope концепт делает наличие `assert_supported` условием КОМПИЛЯЦИИ кольца (T-CPT-негатив); безусловный вызов виден осмотром 4 идентичных блоков; исполнение на живом пути покрыто существующими firewall-тестами всех колец (test_sw_ring.cpp:250-269 и т.д.), остающимися зелёными. Прямая инжекция poison-дескриптора В КОЛЬЦО невозможна без новой машинерии (кольца хардкодят свой потенциал, passes() фиксирован) — честно фиксируется в комментарии; инжектируемая ручка есть у драйвера (T-GAP).

---

## 10. Протокол приёмки (гейт PR-0a)

1. **Чистый ребилд обязателен** (урок Me2, stale-бинарник): `rm -rf build build-cuda` → оба дерева с нуля.
2. **CPU:** `ctest --test-dir build` — существующий 41 таргет зелёный **байт-в-байт** (все bitwise-гейты: 1-vs-z, реплика §3.6, LAMMPS-goldens, Test_Physical_Oracle) + новый `Test_W_Contract`. Итог 42.
3. **CUDA:** `./scripts/gpu_gate.sh build-cuda` — 58 таргетов зелёные байт-в-байт (вкл. новый T-GAP внутри test_cuda_eam_ring); memcheck+racecheck+initcheck чисто (новые пути — host-side throw до cudaMalloc, но прогнать обязательно).
4. **Пояс A/B-hex (графт B):** `eam_drift` (N=864, 500 шагов) до/после PR — hex финальной энергии/координат идентичен HEAD-у (дёшево ловит случайное касание горячего пути, включая рефактор kPasses и хвостовой параметр драйвера).
5. **F-NOOP осмотром:** `git diff --stat` ⊆ файловой карте §1; `git diff` ПУСТ на списке «байт-нетронуты» (в т.ч. `fsm.hpp`, `sw.hpp/tersoff.hpp/meam.hpp`, все `zone_*.cuh`, `{sw,tersoff,meam}_window_force_gpu.cuh`, ci.yml).
6. **Облако:** push → джоба `cuda-compile` зелёная в ОБЕИХ ногах матрицы mpi={OFF,ON}; в логе — компиляция `cuda_compile_check` (первое облачное появление трёх угловых политик + concept-static_assert'ов). Если сработал фолбэк К7 — находка записана в PR-текст.
7. **Kill-мутации:** приёмочный ревьюер применяет ≥3 мутации из таблиц §9 (обязательно: T5 «шов в хвост», T-GAP «снять гейт», T1) и убеждается в красноте — антирецидив structurally-dead (5 прецедентов).
8. Летопись: запись в CLAUDE.md; `Bench` не трогается (перф-претензий у PR нет).

---

## 11. Явные DEFER-ы (что осознанно НЕ в PR-0a и почему)

- **D1. MPI compile-TU** (нога mpi=ON компилирует ноль MPI-кода: `mpi_ring_edge.hpp` включается только `test_mpi_conveyor.cu`, таргет за тройным if CMakeLists:526-541): реальная дыра, но вне буквы PR-0a-буллета; hoist `find_package(MPI)` — лишний configure-риск в F-NOOP-PR. → follow-up PR по готовому рецепту B: `tools/mpi_compile_check.cu` (include `mpi_ring_edge.hpp` + odr-use `MpiRingEdge`) под `if(TDMD_WITH_CUDA AND TDMD_WITH_MPI)` + hoist find_package (идемпотентен в тестах).
- **D2. Хуки `on_zone_arrival`/`on_edge` в концепте** (аудит §3.3 называет их обязательными): в PR-0a их НИКТО не зовёт ⇒ обязательный-но-мёртвый хук у 7+ политик — ровно structurally-dead-класс. Входят в концепт в PR-1/2 ВМЕСТЕ с потребителями (осознанный compile-break, форсящий явные no-op у CPU-оракулов; расширение концепта забыть невозможно — не компилится).
- **D3. Проверка ρ_a(r)≥0 при загрузке setfl** (аудит §3.3): может ложно упасть на существующих golden-setfl (сплайновые осцилляции у нуля) ⇒ сломала бы F-NOOP. Measure-first: проверить фактические setfl офлайн в PR-1 и там же вшить страж (он и семантически принадлежит донациям: эквивалентность rho_cap-триггер-сета).
- **D4. `TD_MD_Core_WContract_v1_0.md` + FSM-аддендум (§3.6):** документные обязательства привязаны к ИСПОЛНЕНИЮ донаций (T2-guard→CoRes, timing INV-8, семантика d) — семантики в PR-0a нет; контракт-на-сегодня несут doc-comment'ы `donation_layout`/`want_closure_mask` (все MUST-FIX-клаузы §3.1/§3.3 дословно). Док — в PR-1 (там же решается §8.1 arrival-vs-finalize(k−1)). `fsm.hpp` байт-неизменен — соблюдено тривиально.
- **D5. Флипы `w_class` shipped-дескрипторов** (вкл. EAM Density → kAccumB1) — PR-1/2; замок T-INERT снимается осознанно.
- **D6. `kAccumCanonSuffix`: роли и гейт потребителей** — canon-PR (validate принимает значение без ролей; строгое правило К4 ослабляется там же).
- **D7. Мутационный зуб membership_ok/g с дрейф-фикстурой** — PR-1 (предпосылка (7) уже вшита doc-comment'ом); ротационный кольцевой зуб (PBC ≥2 оборота, n≥5, побитово) — PR-1/2 (сид — T-ROT).
- **D8. Вакуумный батч исполнительно** (END голодает без бита на пустой зоне) — PR-2; PR-0a несёт конвенцию + `close_cross_batch` + отсутствие параметра населённости у want-стороны.

## 12. Риски

- **R1. Скрытая инстанциация кольца с политикой без assert_supported** → ошибка компиляции (не молчаливая регрессия); митигация — grep-шаг §6 п.3, известный список закрыт.
- **R2. Позиционные PassDecl-инициализаторы** → правило «только в хвост» + компайл-тайм часть T-INERT.
- **R3. Рефактор kPasses в eam.hpp** → grep подтвердил локальность; свидетели — весь EAM-suite байт-в-байт + T-SRC + hex-пояс п.4 приёмки.
- **R4. Делегация GPU-гейта меняет тексты исключений** → существующие зубы проверяют throw/no-throw, не тексты; семантика accept/reject сохранена дословно, префикс `who`.
- **R5. Explicit instantiation впервые вскрывает некомпилирующийся член** → фолбэк К7, решается первым локальным билдом, находка в PR.
- **R6. Циклический include при гейте драйвера** → исключён конструкцией (гейт в eam.hpp); явный запрет в §7.
- **R7. Время облачной сборки** (+1 TU, инстанцирующий 4 кольца, ~bench_eam_ring) → замерить в первом прогоне джобы.

**Смета:** ~550-700 новых строк (из них ~300 — тесты), ~80 изменённых; 2 новых файла, 11 правленых; один состязательный цикл (дизайн ДО — этот документ; приёмка ПОСЛЕ — протокол §10). Зависимостей нет (PR-0b/0c независимы); после мержа PR-1 (serial-оракул EAM-донаций) потребляет §3–§4 без пере-деривации.