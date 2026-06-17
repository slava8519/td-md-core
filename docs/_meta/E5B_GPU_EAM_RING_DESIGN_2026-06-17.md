# E5b — EAM на GPU-кольце (EamGpuConveyor): состязательный дизайн

**2026-06-17.** Состязательно спроектирован (understand 4 читателя → design 3 линзы →
синтез). Цель: прогнать многопроходный EAM (density→embedding→force) через GPU
TimeConveyor-кольцо, **побитово ≡ CPU `EamRing`**, разблокировать EAM-бейкофф.

## Центральный вызов
GPU **пар-кольцо** даёт **rcut-halo** (Newton-3 кросс-пары NEXT/PREV, читают живой
DevSlot соседа на месте) — НИКОГДА не материализует 2·rcut окно. EAM нельзя так
разложить: плотность `ρ_j` граничного атома — непарная редукция по ЕГО полной
rcut-окрестности, а `F'(ρ_j)` (нужна для силы на owned-`i`) зависит от доноров в
ТРЕТЬЕЙ зоне. ⇒ нужно материализовать 3-зонное окно {S_{j-1},S_j,S_{j+1}} +
финализировать ЦЕНТР j + задержка-на-зону (second-forward-hop). CPU `EamRing` это
уже решает (`finalize_owned`, `eam_ring.hpp`); E5b портирует на стримы.

**Стратегия (ключ к bitwise):** `gather_window` D2D-копирует 3 живых DevSlot в ОДИН
непрерывный буфер → запускает **E5-ядра (`zone_eam.cuh`) ДОСЛОВНО**. int64-сумма
ассоциативна ⇒ зависит только от МУЛЬТИМНОЖЕСТВА окна, не порядка gather ⇒ GPU≡CPU
по построению (при key=atom id).

## Архитектура (единогласно)
- **Отдельный класс `EamGpuConveyor<Real,Math>`** (НЕ template-split пар-кольца — оно
  bitwise-связано с ~12 GPU-тестами). Пар-кольцо БАЙТ-НЕТРОНУТО.
- **Extract транспорта** → новый `conveyor_gpu_transport.cuh` (чистый code-move:
  `StreamEdge`, `StreamTransport`, `IBoundaryEdge`/`RingPart`, `DevSlot`, `GpuHeader`,
  `EndScalars`, `NvtxScope`, `TDMD_CU`, `reset_zone_scalars_kernel`,
  `membership_guard_kernel`, `snap_i32`/`pack`/`unpack`). `conveyor_gpu.cuh` включает
  его. Доказать пар-ring byte-identical: `gpu_gate.sh` зелёный. (НЕ дублировать
  протокол INV-2/5/6 — две копии = determinism-ловушка.)

## Блокеры/мейджоры (зафиксированы в дизайне)
| # | Sev | Находка (в коде) | Фикс |
|---|---|---|---|
| F1 | **blocker** | Пар-ring освобождает слот-источник на onward-copy (`conveyor_gpu.cuh:1193`); дословно ⇒ продюсер перезапишет донора S_{j-1} до чтения finalize(j+1) ⇒ ρ усечён ⇒ 1-vs-z GREEN | **Schedule-based send-delay** (порт `eam_ring.hpp:337`): на скане j слать ТОЛЬКО j-2; tail-flush n-2,n-1 (free-z) / defer_head tail-batch (PBC). Существующая free-машинерия тогда корректна БЕЗ изменений. |
| F2 | **blocker** | `DevSlot` БЕЗ atom-id (kArrays=10); `eam_force_kernel` φ-once на `key=id` (`zone_eam.cuh:96,111`); window-order≠id-order ⇒ PE неверна | Добавить пер-слот device `long id[cap]` в EAM-узел, заполнять из `atoms_.id`, нести через D2D, gather в `key[]`. Пар-DevSlot НЕ трогать. Shuffled-id регресс-тест. |
| F3 | **blocker** | 1-vs-z И self-loop S=n+2 ОБА маскируют выпавшего донора (z-однородная ошибка / все зоны резидентны) | Обязательны **restricted-window (z>1, S=⌈n/Z⌉+5) superset-оракулы A-D** vs `eam_direct_fp64`/монолит. **НЕ принимать по 1-vs-z.** |
| F4 | major ✅ DONE | center-vs-edge off-by-one окна детерминирован ⇒ невидим 1-vs-z | **`eam_window_layout(j,n,pbc,wslots)→nw`** в `eam_zone.hpp` — единый источник для CPU+GPU. (Реализовано, `test_eam_ring` bitwise.) |
| F5 | major | `min_r2` пар-ring reset = `+inf` (`:687`); CPU EAM-оракул seed `1e300` ⇒ CPU↔GPU mismatch на пустых окнах | Reset `d_mr2 = pos_double_bits(1e300)` в EamGpuConveyor, НЕ inf. |
| F6 | major | Новый `.cu` без `tdmd_eam_cuda_flags` ⇒ FMA-fusion `f_over_r`/Horner ⇒ 1-ULP CPU↔GPU расхождение | Линковать `tdmd_eam_cuda_flags` (--fmad=false); CMake-assert; `NearKnotFlagAudit` в CI. Gather БЕЗ FP. |
| F7 | major | EAM working-set 4 зоны {j-2 pending,j-1,j,j+1} vs пар 3 ⇒ риск дедлока | self-loop S=n+2; multi-node S=min(n+2, max(9, ⌈n/Z⌉+5)) — +1 над пар-ring. Anti-deadlock тест НАХОДИТ маржу. |
| F10 | minor | hardwire fracbits=44 ⇒ Q23.40 расходится | Диспатч `dens_scale`/`DensAccum` на `pot.math.density_fracbits()` (44/40), как `eam_ring.hpp:62`. |

## Минимальный shippable scope (E5b)
1. **Transport extraction** → `conveyor_gpu_transport.cuh` (пар-ring byte-identical, gpu_gate).
2. **`EamGpuConveyor`** — **fp64 D2D**, **O(m²)** gathered окна, free-z + PBC defer_head,
   schedule-based delayed send, +1 co-residency, EAM-slot id-массив, E5-ядра дословно.
   - `run()`: `ZoneDecomposition::build(reach_mult=2)` (width≥2rcut+skin), upload
     `EamSetflView`, `fb_=density_fracbits()` → dispatch scale/DensAccum, t0 через
     serial `zone_eam_pass`, slot-pool по F7.
   - DevSlot += `long id[cap]`. Staging (once): `window_{x,y,z}`(3cap), `window_key`(long),
     `window_owned`(int cap), `d_rho`/`wF{x,y,z}`(ll 3cap), `d_fp`(double 3cap).
   - `compute_zone(center j)`: ensure_arrival/drift окна по `eam_window_layout`; assert
     present&&drifted (empty n==0 легитимно); **gather** (D2D, no FP, offsets [0,n_{j-1},…],
     owned=центр-range, m=Σ); density→embedding→force ДОСЛОВНО; scatter wF→центр slot
     arr(6..8); `end_zone` (FSM, 2nd-half, зон-локальные v/a/k2/KE, INV-4, StaleZone
     margin g=0.5(width−2rcut), END); enqueue END-scalar D2H (batched sync_check).
   - **send schedule (F1):** скан j → `send_slot(j-2)` (guard !defer_head && j≥2); tail
     n-2,n-1. PBC defer_head: SEND no-op, finalize owned-0 в tail (cyclic {n-1,0,1}),
     tail-batch 1..n-1,0.
   - **guards/min_r2:** per-pass reset `d_mr2=1e300`, `d_pe=0`, `d_flags=0`; sync_check:
     flags(rho-cap/quantize)→Halt::Internal (симм. CPU throw), min_r2→Overlap, StaleZone,
     INV-4. Λ-chain dt + R_buf (v_pred=v+a·dt·max(1,n-1)) + auto_dt — из header-пути.
3. **Приёмка `tests/test_cuda_eam_ring.cu`** (label cuda, линк `tdmd_eam_cuda_flags`) —
   **НЕ принимать по 1-vs-z:**
   - Порт bitwise-сьюта GPU `EamGpuConveyor` ≡ CPU `EamRing`: SingleNodeMatchesSerialVV;
     OneVsZ z∈{2,3,5,6} free+PBC; AutoDtOneVsZ; DualFormatQ23OneVsZ; LongRunReplica(1-vs-4);
     VacuumGapEmptyZones; PBC Momentum/LongRun; OverlapHalt; NveEnergyConservation(1000);
     **AntiDeadlock z=1..6 free + 1,5,6 PBC (находит +1 маржу)**.
   - **Oracle A (решающий):** forward-only `{j,j+1}` gather-опция, z=3, assert РАСХОДИТСЯ
     с `eam_direct_fp64` (maxdev>1e-6) — доказывает, что тест ВИДИТ выпавшего донора.
   - **Oracle B:** z=5(free)+z=6(PBC) restricted pool S=⌈n/Z⌉+5, один проход high-T
     jittered FCC, силы vs `eam_direct_fp64` (независимый FP64), maxdev<1e-10.
   - **Oracle C:** пер-owned raw int64 ρ_j bit-equals whole-system монолит `eam_window_force`.
   - **Oracle D:** z>1, ~50 шагов, T гонит атомы в пределах g граней, GPU-ring ≡
     `eam_direct_fp64`-serial-VV per step.
   - Shuffled-within-zone id-тест; `NearKnotFlagAudit`; `gpu_gate.sh` (memcheck+racecheck).

## Отложено (E5c+, каждое — свой состязательный дизайн)
EAM cell-lists/verlet на кольце; **cells-vs-verlet бейкофф** (нужен verlet+EAM —
циклическая зависимость, perf-follow-up, НЕ корректность E5b); mixed_transport int32 для
EAM; MPI/RingPart EAM-граница. **Ценность E5b = гейт корректности + разблок архитектуры,
НЕ перф.**

## Инварианты/ловушки
INV-9 int64 (gather сохраняет мультимножество); 1-vs-z bitwise; пар-`TimeConveyor` +
CPU `EamRing` БАЙТ-НЕТРОНУТЫ (EamRing — оракул); density-аккумулятор order-independence;
EAM-spline transcendental-free ⇒ CPU↔GPU **bitwise** (не ~1e-12). **ГЛАВНАЯ ловушка:**
выпавший донор в 2·rcut окне ⇒ ρ усечён ⇒ неверно на ВСЕХ z, 1-vs-z GREEN — ловят только
serial `zone_eam_pass` + restricted-window superset (Oracle A-D). Финализировать ЦЕНТР j
(не j-1). Не перерабатывать слот-донора до finalize потребителя.
