# Achieved occupancy (ncu) — Tier-2 кластерное ядро M3

**Дата:** 2026-06-12 · закрывает отложенный пункт M3 («ncu achieved-occupancy») и
часть критерия M4 «achieved occupancy зафиксирована и сопоставлена с прогнозом
M2.5/M2.7» — для ядра M3; ядра конвейера M4 получат собственное измерение.

## Разблокировка профилирования

До 2026-06-12 ncu падал с `ERR_NVGPUCTRPERM` (`RmProfilingAdminOnly=1`).
Снято разовой админ-операцией:

```
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-profiling.conf
update-initramfs -u   # + перезагрузка
```

Проверка: `grep RmProfilingAdminOnly /proc/driver/nvidia/params` → `0`.

## Методика

```
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,launch__grid_size \
    --kernel-name regex:cluster_force_kernel \
    ./build-cuda/test_cuda_cluster \
    --gtest_filter='CudaCluster.MatchesCpuClustered11k:CudaCluster.Fp32PairMathOnGpu'
```

ncu 2025.4.0, CUDA 13.1, RTX 5080 (sm_120, 84 SM, 1536 threads/SM, лимит 24 blocks/SM).

## Результаты vs прогноз

| Инстанциация | regs/thread | Теоретич. per-SM occupancy (blocks/SM) | Grid | Achieved (sm__warps_active) |
|---|---|---|---|---|
| fp64-pair, N=10 976 | 56 | **75 %** (9×128/1536) | 86 blocks | **6.5 %** |
| fp32-pair, N=1 372 | 62 | **67 %** (8×128/1536) | 11 blocks | **7.9 %** |

**Вывод — прогноз Tier-1/Tier-2 подтверждён измерением:** per-SM occupancy не
является узким местом (75 % теоретических при 56 регистрах, блок 128 = 4
варп-кластера); провал achieved до ~7 % — чисто **дефицит варпов в полёте**:
grid 86 блоков ≈ 1 блок/SM при резидентной ёмкости 84 SM × 9 = 756 блоков.
Для насыщения резидентности этому маппингу нужно ≈ 756 блоков × 4 кластера ×
32 атома ≈ **10⁵ атомов на запуск** — согласуется с выводом зонда M2.5
(«одиночная тонкая зона при N=10⁶ заполняет 12.2 %»).

## Следствия для M4/M5a (без изменений плана — подтверждение)

1. Ядра конвейера на малых системах живут только за счёт **конкурентности
   стримов** (батчинг ~зон/стрим, A2) — одиночный запуск зоны GPU не загрузит.
2. Рычаг occupancy на кластере — `decomposition.ring.steps_per_node` k>1
   (Гл. 3.4, ур. 49–51): до k зон в полёте на узел (B4).
3. Перф-замеры M4 на N=10⁵–10⁶ (методика Bench) — ниже этой границы измеряется
   латентность запуска, а не дизайн конвейера.
