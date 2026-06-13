# K-aware fallback (И1): привязка к ZoneHeader / flush_sends · 2026-06-13

`[НОВОЕ, ENG]` — превращает режимный downside persistent-списка в «нет downside, режимный upside». Строится поверх эскиза из `VERLET_SKIN_DESIGN_2026-06-13.md` §7 (поля `skin_consumed`, `rebuild_now`; рекуррентность в `flush_sends`; ветвление в `ensure_drift`). Не валидировано.

**Идея (синтез §5–И1).** Предсказанный фактор переиспользования `K` — это **глобальный z-независимый скаляр** (функция от `skin` и `R_buf`, оба уже на Λ-цепочке). При `K_pred < K*` (≈2, точка безубыточности из синтеза §4.2) персистентный список не амортизируется → детерминированно, для всего набора зон, откатываемся на текущий cell-raster путь (Базис A): список не материализуется, налог памяти 0.84 ГБ не платится, силовое ядро прометает 27-ячеечную окрестность как сейчас. Худший случай движка становится «= текущий», а не «текущий + налог».

---

## 1. Что уже есть (эскиз гибрида, для самодостаточности)

Из `VERLET_SKIN_DESIGN` §7:

```cpp
// core::ZoneHeader (transport.hpp)
struct ZoneHeader {
    // ... существующие поля (dt_next, агрегаты Λ-цепочки) ...
    double  skin_consumed;   // НОВОЕ (гибрид): несомый skin-бюджет
    uint8_t rebuild_now;     // НОВОЕ (гибрид): флаг перестройки, реплицируется во ВСЕ заголовки
};
```

```cpp
// flush_sends, блок sent==0 (там же, где dt_next):
//   R_buf = C_buf * v_pred * dt  (уже посчитан при arrival 0)
charge        = 2.0 * R_buf;           // НЕСУЩАЯ правка: 2·R_buf, не 2·v_pred·dt
skin_consumed += charge;
rebuild_now    = (skin_consumed >= skin);
if (rebuild_now) skin_consumed = 0.0;
// записать skin_consumed И rebuild_now во ВСЕ исходящие заголовки (реплика, не p2p)
```

```cpp
// ensure_drift:
if (rebuild_now) { /* cell_count + scan + scatter + материализация Verlet-списка */ }
else             { /* пропуск, переиспользование списка */ }
```

---

## 2. Что добавляет И1

### 2.1. Новое поле заголовка

```cpp
struct ZoneHeader {
    // ...
    double  skin_consumed;
    uint8_t rebuild_now;
    uint8_t verlet_active;   // НОВОЕ (И1): 1 = Verlet-путь, 0 = cell-raster fallback (Базис A)
};
```

В `lam0_`/preload: `verlet_active = cfg.verlet_default` (см. §5 — холодный старт). `skin_consumed=0`, `rebuild_now=0`.

### 2.2. Решение в flush_sends (там же, у головы)

Голова — единственный писатель (как `dt_next`/`rebuild_now`). `K_pred = skin / charge = skin / (2·R_buf)`. Гистерезис двумя порогами против дребезга:

```cpp
// flush_sends, блок sent==0, ПОСЛЕ вычисления charge=2*R_buf:
const double K_pred = skin / charge;        // charge = 2*R_buf > 0

// гистерезис: вкл при K_pred>=K_on, выкл при K_pred<K_off
//   K_on ≈ 3.0, K_off ≈ 1.5  (K_off — точка безубыточности, K_on — запас от дребезга)
uint8_t va = verlet_active_prev;            // состояние с прошлого прохода
if (va == 0 && K_pred >= cfg.K_on)  va = 1;
if (va == 1 && K_pred <  cfg.K_off) va = 0;
verlet_active = va;

// бюджет/флаг считаем ТОЛЬКО когда Verlet активен:
if (verlet_active) {
    skin_consumed += charge;
    rebuild_now    = (skin_consumed >= skin);
    if (rebuild_now) skin_consumed = 0.0;
} else {
    skin_consumed = 0.0;                     // в fallback бюджет неактуален
    rebuild_now   = 0;                       // материализация Verlet не нужна
}

// реплицировать skin_consumed, rebuild_now, verlet_active во ВСЕ исходящие заголовки
```

### 2.3. Эффективная перестройка (переход 0→1 требует rebuild)

При включении Verlet списка нет — первое переиспользование невозможно без материализации. Поэтому:

```cpp
// на arrival 0 / в ensure_drift:
const bool turned_on    = (hdr.verlet_active == 1) && (verlet_active_prev == 0);
const bool eff_rebuild  = hdr.rebuild_now || turned_on || migration_pass;

if (hdr.verlet_active) {
    if (eff_rebuild) { /* cell_count+scan+scatter + материализация Verlet-списка */ }
    else             { /* переиспользование списка */ }
} else {
    /* Базис A: cell-raster путь — силовое ядро прометает 27-ячеечную окрестность,
       Verlet-список НЕ материализуется и НЕ итерируется */
}
verlet_active_prev = hdr.verlet_active;
```

Переход `1→0` бесплатен (просто перестаём итерировать список). Переход `0→1` форсирует одну перестройку (через `turned_on`).

---

## 3. Машина состояний

```
                K_pred < K_off
        ┌──────────────────────────────┐
        ▼                              │
  ┌───────────┐   K_pred ≥ K_on   ┌───────────────┐  skin_consumed<skin   ┌──────────────┐
  │ CELL_RASTER│ ───(force reb.)──▶│ VERLET_REBUILD │ ─────────────────────▶│ VERLET_REUSE │
  │ (Базис A) │                   │ (материализация)│◀──── rebuild_now ─────│ (переисп.)   │
  └───────────┘                   └───────────────┘   (или migration_pass)  └──────────────┘
        ▲                                                                          │
        └──────────────────── K_pred < K_off ─────────────────────────────────────┘
```

- `CELL_RASTER`: `verlet_active=0`. Текущее поведение движка.
- `VERLET_REBUILD`: `verlet_active=1 && eff_rebuild`. Материализация списка.
- `VERLET_REUSE`: `verlet_active=1 && !eff_rebuild`. Дешёвый шаг (~28 кандидатов вместо 142).
- Миграция форсирует `VERLET_REBUILD` (если Verlet активен) или остаётся в `CELL_RASTER`.

---

## 4. Почему 1-vs-z сохраняется (главное свойство)

`verlet_active` — детерминированная функция от `skin` и `R_buf`:
- `R_buf = C_buf·v_pred·dt` — из агрегатов Λ-цепочки, **уже доказанных z-независимыми** (тот же путь, что `dt_next`);
- `skin` — FP64-константа конфигурации;
- `K_pred = skin/(2·R_buf)` — один FP64-делёж бит-идентичных значений;
- сравнения `K_pred ≥ K_on`, `K_pred < K_off` — булевы над бит-идентичными FP64;
- гистерезис детерминирован: `verlet_active_prev` тоже z-независим по индукции (стартовое значение — общая константа preload).

⇒ `verlet_active` бит-идентичен на любом z, run-to-run, в mixed. **Все зоны набора флипают вместе, на одном логическом шаге, на каждом z.** Переход и форс-rebuild при `0→1` происходят синхронно во всех декомпозициях ⇒ суперсет не рвётся, траектория не расходится. Аргумент тот же, что для `rebuild_now` (правка-каретка синтеза §3). Реплика всех трёх полей в каждый заголовок — обязательна по той же причине.

> Без гистерезиса при `K_pred ≈ K*` `verlet_active` дребезжал бы 0↔1, и каждый `0→1` форсировал бы перестройку ⇒ деградация хуже текущего движка. Два порога (`K_on > K_off`) убирают дребезг детерминированно.

---

## 5. Память, холодный старт, прочее

- **Память при долгом fallback.** Когда `verlet_active` устойчиво 0 (длинный ударный прогон), персистентный Verlet-буфер (~0.84 ГБ на 1e7) мёртв. **Не освобождать** — освобождение/реаллокация GPU-буферов на ходу дорого и фрагментирует; бюджет 3.5 ГиБ терпит простой буфер (синтез §5). Держать аллокированным, не итерировать.
- **Холодный старт.** На проходе 0 `R_buf` ещё не известен (считается при arrival 0 первого реального прохода). `preload` ставит `verlet_active = cfg.verlet_default`: разумно `0` (cell-raster) для безопасного старта — на первом проходе списка всё равно нет, надо строить; со следующего прохода `verlet_active` считается из `R_buf`. Либо инициализировать из стартовой оценки `v_max` при setup, если режим прогона известен заранее.
- **Зона/halo.** `make_zone_grid` halo → `rcut+skin`, ширина `w>rcut+skin` валидируется **всегда** (envelope фиксирован под `skin`, даже если на эпохе `verlet_active=0` — чтобы переход `0→1` не требовал переразбиения). Размер зоны под `skin` платится памятью один раз; авто-адаптируется лишь *каденция* (через `rebuild_now`) и *включённость* (через `verlet_active`), не геометрия.
- **MPI.** `verlet_active` едет в `ZoneHeader` по `MpiRingEdge` вместе с `skin_consumed`/`rebuild_now` — новых транзакций нет.

---

## 6. Что добавить в план валидации

И1 вводит новое состояние (`verlet_active`) и переход с форс-rebuild → расширить гейты:

- **Boundary-crossing sweep (расширение Gate-01/Gate-03):** прогон с медленно меняющимся `v_max`, пересекающим порог `K*` туда-обратно. Проверить:
  - **1-vs-z идентичность через переход:** `z=1,2,4,8,16` флипают `verlet_active` на **одном** логическом шаге; бит-идентичность конечного состояния.
  - **Нет дребезга:** число переходов 0↔1 ограничено (гистерезис работает); каждый `0→1` сопровождается ровно одной перестройкой.
  - **Stale-list negative:** после `0→1` НЕ переиспользуется старый (невалидный) список — `turned_on` форсировал rebuild. Намеренно подавить `turned_on` в тестовой сборке и убедиться, что Physical Oracle ловит пропущенную пару (т.е. тест действительно проверяет этот путь).
- **Graceful degradation (расширение shock sweep):** в режиме, схлопывающем `K`, движок уходит в `CELL_RASTER` и его время шага `≈` текущему движку (а не «текущий + налог»). Метрика: суммарное время шага в глубоком ударе ≤ baseline-A + ε.

---

## 7. Минимальный diff (сводка точек врезки)

| Файл / функция | Изменение |
|---|---|
| `core::ZoneHeader` (transport.hpp) | `+uint8_t verlet_active;` (рядом с `skin_consumed`, `rebuild_now`) |
| `lam0_`/preload | `verlet_active = cfg.verlet_default` (≈0); добавить `cfg.K_on≈3.0`, `cfg.K_off≈1.5` |
| `flush_sends` (блок `sent==0`) | вычислить `K_pred`; гистерезис → `verlet_active`; считать `skin_consumed`/`rebuild_now` только при `verlet_active`; реплика трёх полей во все заголовки |
| `ensure_drift` / arrival 0 | `eff_rebuild = rebuild_now \|\| turned_on \|\| migration_pass`; ветка `verlet_active ? (eff_rebuild?материализация:переисп.) : cell-raster`; `verlet_active_prev = …` |
| `make_zone_grid` | halo/ширина под `rcut+skin` валидируются всегда (envelope фиксирован) |
| тесты | boundary-crossing sweep; graceful-degradation; stale-after-0→1 negative |

---

_Связь: `SKIN_CRITERION_SYNTHESIS_2026-06-13.md` §5 (И1), §4.2 (точка безубыточности `K≈2`), §7 (CI). Эскиз кода — на структурах `VERLET_SKIN_DESIGN_2026-06-13.md` §7. Имена полей/функций — гипотетические по эскизу; сверить с актуальным transport.hpp перед врезкой._
