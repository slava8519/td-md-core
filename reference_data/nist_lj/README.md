# NIST SRSW: Lennard-Jones Fluid Reference Calculations (Cuboid Cell)

Эталонные данные NIST Standard Reference Simulation Website для валидации
энергий/сил/вириала LJ-кода (см. `docs/_meta/VALIDATION_EXPERIMENT_2026-06-12.md`).

## Источник

- Страница: <https://www.nist.gov/mml/csd/chemical-informatics-research-group/lennard-jones-fluid-reference-calculations>
  (канонический URL на самой странице:
  <https://www.nist.gov/mml/csd/chemical-informatics-group/lennard-jones-fluid-reference-calculations-cuboid-cell>).
  Заголовок: «Lennard-Jones Fluid Reference Calculations: Cuboid Cell».
  На странице: «Created November 29, 2012, Updated April 8, 2026».
- Архив конфигураций (скачан вербатим):
  <https://www.nist.gov/system/files/documents/mml/csd/informatics_research/lj_sample_configurations-tar.gz>
  (ссылка со страницы: `https://www.nist.gov/document/ljsampleconfigurations-targz`).
- Дата обращения: **2026-06-12**.

## Файлы (вербатим из tar-архива NIST)

| Файл | N | Бокс (x=y=z), σ | ρ* | sha256 |
|---|---|---|---|---|
| `lj_sample_config_periodic1.txt` | 800 | 10.0 | 0.8      | `6fa4bf9d4cb84c07d4766e4c1143c5fdf0c560c47a695336c6d06b19e425d295` |
| `lj_sample_config_periodic2.txt` | 200 | 8.0  | 0.390625 | `d6714977c486280577f2fe69fdfd52188b47a1bdace653edf47ad44b29f37c9a` |
| `lj_sample_config_periodic3.txt` | 400 | 10.0 | 0.4      | `c8e13fef51b21cea5f958ae911c0f45fe3f7ef4f92578d68569258bbb29113ec` |
| `lj_sample_config_periodic4.txt` | 30  | 8.0  | 0.05859375 | `aa823314ac4abf597cbe9faf9b9caab267a78ddab1433779a6d7da72f59e1883` |
| `metadata.README` | — | — | — | `9eb104d5fa749c84f70d19e7625bcde260426c2ace9424cc393ca34e8853e2c8` |
| `lj_sample_configurations-tar.gz` (исходный архив) | — | — | — | `3082426689d384196295709961056c7b833c6e525fd890b8eb3933cb1ea100ba` |

## Формат файлов (по `metadata.README` NIST, проверено)

- **Строка 1:** размеры периодического объёма по x, y, z (три числа).
- **Строка 2:** число LJ-атомов N.
- **Строки 3…N+2:** колонка 1 — ID атома (= номер строки − 2, 1…N);
  колонки 2–4 — координаты x, y, z.
- Все координаты и размеры — в **приведённых единицах LJ** (σ-единицы);
  числа в формате `%.12E` (13 значащих цифр), разделители — пробелы.
- Ячейка кубоидная (углы 90°), PBC/minimum image по всем трём осям.

## Эталонная таблица NIST (вербатим, все значения в приведённых единицах)

Три схемы обрезки: **LRC** — потенциал обрезан на r_c с длиннодействующей
поправкой; **LFS** (linear-force shift = cut-and-force-shifted) — обрезка
на r_c со сдвигом потенциала и силы. **5 значащих цифр** (мантисса
с 4 десятичными знаками в exp-нотации; нули LFS напечатаны как `0.0000E0`).

| Configuration | LRC r_c*=3.0: U_pair* | W_pair* | U_LRC* | LRC r_c*=4.0: U_pair* | W_pair* | U_LRC* | LFS r_c*=3.0: U_pair* | W_pair* | U_LRC* |
|---|---|---|---|---|---|---|---|---|---|
| 1 (N=800) | -4.3515E+03 | -5.6867E+02 | -1.9849E+02 | -4.4675E+03 | -1.2639E+03 | -8.3769E+01 | -3.8709E+03 | 3.1754E+02 | 0.0000E0 |
| 2 (N=200) | -6.9000E+02 | -5.6846E+02 | -2.4230E+01 | -7.0460E+02 | -6.5599E+02 | -1.0226E+01 | -6.2012E+02 | -4.4533E+02 | 0.0000E0 |
| 3 (N=400) | -1.1467E+03 | -1.1649E+03 | -4.9622E+01 | -1.1754E+03 | -1.3371E+03 | -2.0942E+01 | -1.0210E+03 | -9.3578E+02 | 0.0000E0 |
| 4 (N=30)  | -1.6790E+01 | -4.6249E+01 | -5.4517E-01 | -1.7060E+01 | -4.7869E+01 | -2.3008E-01 | -1.5001E+01 | -4.3096E+01 | 0.0000E0 |

**ВАЖНО — W_LRC:** на странице NIST (версия от 2026-04-08) **нет** ни столбца,
ни формулы W_LRC (поправки к вириалу). Таблица содержит только U_pair*, W_pair*,
U_LRC*. При сверке давления поправку к вириалу нужно брать из учебной формулы
(Allen–Tildesley / Frenkel–Smit), но эталонного числа NIST для неё здесь нет.

## Определения NIST (вербатим, раздел «3. Definitions» страницы)

A. Потенциал Леннард-Джонса:

$$V_{LJ}\left(r\right)=4\epsilon\left[\left(\dfrac{\sigma}{r}\right)^{12}-\left(\dfrac{\sigma}{r}\right)^6\right]$$

B. Схема LRC («Long-Range Correction»): реальный потенциал в симуляции —

$$V\left(r\right) = \begin{cases} V_{LJ} \left( r \right) & r \leq r_c \\ 0 & r > r_c \end{cases}$$

C. Схема LFS («Linear-Force Shift», cut-and-force-shifted):

$$V\left( r \right) = \begin{cases} V_{LJ} \left( r \right) - V_{LJ} \left(r_c\right) - \left. \dfrac{\partial V_{LJ}}{\partial r}\right|_{r_c} \left(r-r_c\right) & r \leq r_c \\ 0 & r > r_c \end{cases}$$

D. Парная внутренняя энергия (V(r) — симулируемый потенциал, т.е. B или C):

$$U_{pair} = \sum_{i=1}^{N-1} \sum_{j=i+1}^N V\left(r_{ij}\right)$$

E. Длиннодействующая поправка, **per particle** (на частицу):

$$U_{LRC} = \dfrac{1}{2}\, 4 \pi \rho \int_{r_c}^{\infty} dr\, r^2\, V_{LJ}\left( r \right)$$

F. Мгновенный парный вириал — **без множителя 1/3, со знаком минус, через ∂V/∂r**
(давление из него: P = ρk_BT + W/(3V) + поправки):

$$W_{pair} = -\sum_{i=1}^{N-1} \sum_{j=i+1}^N r_{ij} \left.\dfrac{\partial V}{\partial r}\right|_{r_{ij}}$$

Ссылки NIST: [1] Allen & Tildesley, *Computer Simulation of Liquids* (Oxford, 1989);
[2] Frenkel & Smit, *Understanding Molecular Simulation*, 2nd ed. (Academic, 2002), pp. 37–38.

### Примечание о нормировке U_LRC в таблице

Формула E дана «per particle», но значения **в таблице — экстенсивные**
(вся система): табличное U_LRC* = N × (формула E)
= (8/3)πρ*N[⅓(r_c*)⁻⁹ − (r_c*)⁻³]. Проверено численно: для конфигурации 1,
r_c*=3.0 формула даёт −198.489 ≈ −1.9849E+02 (совпадает с таблицей); расхождение
≤1 ед. последней значащей цифры для всех 8 значений U_LRC.

## Независимая верификация (2026-06-12)

Все 36 табличных значений пересчитаны напрямую из скачанных файлов
(double, minimum image, формулы B/C/D/F дословно): максимальное отклонение
от таблицы NIST — в пределах округления 5-й значащей цифры
(худший случай |Δ| = 5.0E-02 при значении ~1.16E+03, т.е. <0.5 ед. последней
цифры). Целостность: N в каждом файле совпадает с заявленным, ID атомов
строго 1…N, первая строка — размеры бокса согласно `metadata.README`.
