# Split-encoder с diff фич shared_encoder в actor/oracle ветку

Дата: 2026-06-07
Ветка: `dev_memory_oracle`

## Цель

Вариант split-encoder моделей, в котором:

- `shared_encoder` получает на вход **только кадр** (1 канал), без diff-канала из обёртки;
- модель хранит **предыдущий выход** `shared_encoder` и подаёт в `actor_oracle_encoder`
  **разность** между текущим и предыдущим выходом shared (diff на уровне фич);
- `critic_encoder` по-прежнему получает текущий выход shared.

Идея: actor/oracle ветка видит «движение» (изменение латентного представления между шагами),
critic — статическое текущее представление.

## Контекст (как сейчас)

Базовые классы (ветка `dev_memory_oracle`, WIP):

- `A3CRules2378OracleSplitEncoders` — `shared_encoder` → две ветки
  (`actor_oracle_encoder`, `critic_encoder`), декодер от actor/oracle фич, restoration loss.
- `A3CRules2378OracleSplitEncodersIntrinsicCritic` — то же + intrinsic-голова критика (6-й выход).

Текущий конфиг (`run_config_oracle_split_8_64_v2.ini`) использует обёртку
`NormalizedEnvGameNormNoNormDiffOnNorm`, которая отдаёт **2 канала** `[normalized_obs, diff]`;
оба сейчас идут в `shared_encoder` (`num_inputs = 2`).

Поток данных, который остаётся неизменным:

- `shared_encoder` сжимает вход до `[B, 64, 4, 4]`.
- Декодер восстанавливает `num_inputs` каналов; restoration loss сравнивает `x_restored`
  с `G_t = next_states - states` (тоже `num_inputs` каналов) через косинус.
- Каждый воркер в `train.py` создаёт **свою локальную копию** модели — хранить состояние
  в атрибуте модели безопасно (как уже сделано для `running_mem` / `prev_x_conv`).

## Дизайн

### 1. Два новых класса в `src/model.py`

Имена:
- `A3CRules2378OracleSplitEncodersSharedDiff(A3CRules2378OracleSplitEncoders)`
- `A3CRules2378OracleSplitEncodersSharedDiffIntrinsicCritic(A3CRules2378OracleSplitEncodersIntrinsicCritic)`

Каждый:

- В `__init__`: вызвать `super().__init__(...)`, затем `self.prev_shared = None`.
  Слои **не меняются** — diff имеет ту же форму `[B, 64, 4, 4]`, поэтому
  `actor_oracle_encoder` остаётся `Conv2d(64, 64, 3, 1, 1)`.

- Переопределённый `forward`:
  ```python
  shared, _, _, _ = self.shared_encoder(inputs)
  shared = shared.view(shared.size(0), 64, 4, 4)

  if self.prev_shared is None:           # первый кадр эпизода: diff = 0
      diff = torch.zeros_like(shared)
  else:
      diff = shared - self.prev_shared
  self.prev_shared = shared.detach()     # храним детачнутым — без BPTT между шагами

  actor_oracle_feat = F.relu(self.actor_oracle_encoder(diff))   # вход = ТОЛЬКО diff
  critic_feat       = F.relu(self.critic_encoder(shared))       # вход = текущий shared

  actor_oracle_flat = actor_oracle_feat.view(actor_oracle_feat.size(0), -1)
  critic_flat       = critic_feat.view(critic_feat.size(0), -1)

  if self.monitor_s:
      self.s_values.append(shared.detach().cpu())

  x_restored = self.decoder(actor_oracle_feat)

  hx = torch.Tensor([0]); cx = torch.Tensor([0])
  ```

- Возврат:
  - `SharedDiff`: `critic_linear(critic_flat), actor_linear(actor_oracle_flat), hx, cx, x_restored`
    (5 выходов, как у `A3CRules2378OracleSplitEncoders`).
  - `SharedDiffIntrinsicCritic`: дополнительно
    `value_intrinsic = critic_linear_intrinsic(actor_oracle_flat)` и 6-й выход
    (как у `A3CRules2378OracleSplitEncodersIntrinsicCritic`).

Решения по семантике:
- `actor_oracle_encoder` получает **только** diff (не concat с shared) — по требованию.
- `prev_shared` хранится **detached**: предыдущий выход — константа, градиент течёт только
  через текущий `shared`. Backward по батчу остаётся корректным без `retain_graph`.
- «Первый предыдущий == первому»: при `prev_shared is None` diff нулевой, т.е. эквивалентно
  «предыдущий равен текущему».

### 2. Сброс состояния — `src/player_util.py`

Рядом с существующим сбросом `running_mem` / `prev_x_conv` по `done` (в `action_train`
и в `action_test`) добавить:

```python
if hasattr(self.model, 'prev_shared'):
    self.model.prev_shared = None
```

Это даёт diff = 0 на первом кадре каждого нового эпизода.

### 3. Конфиг эксперимента (`configs/`)

Новые `.ini` (копии `run_config_oracle_split_8_64_v2.ini`) с изменениями:

- `model_type = A3CRules2378OracleSplitEncodersSharedDiff`
  (и отдельный конфиг с `...SharedDiffIntrinsicCritic`).
- `input_normalization_class = NormalizedEnv_orig` — **одноканальная** обёртка
  (running mean/std нормализация кадра, как `normalized_obs` в текущей обёртке).
- `normalization_alpha = 0.999` — явно, чтобы поведение нормализации совпало с текущим
  (у `NormalizedEnv_orig` дефолт 0.9999, у текущей обёртки — 0.999).

Одноканальная обёртка автоматически делает `num_inputs = 1`, поэтому `shared_encoder`,
`decoder` и `G_t = next_states - states` становятся 1-канальными и согласованными —
restoration loss работает без изменений.

При необходимости — соответствующие `slurm_scripts/our/parallel_*.sh` по образцу
существующих split-скриптов.

## Что НЕ трогаем

- `src/environment.py` — используем существующий `NormalizedEnv_orig`, новых обёрток нет,
  дефолты не меняем (alpha задаётся в конфиге).
- `src/train.py`, `src/eval.py` — новые классы наследуются от split-классов, поэтому все
  существующие `isinstance`-проверки ловят подклассы автоматически.
- `shared_encoder`, `actor_oracle_encoder`, `critic_encoder`, декодер, головы — сигнатуры
  без изменений.

## Проверка корректности

- Размерности: вход 1×80×80 → `shared_encoder` → `[B,64,4,4]`; diff той же формы; ветки
  дают `[B,1024]`; головы и декодер согласованы.
- Графы: `prev_shared` detached → per-step backward корректен, без `retain_graph`.
- Границы эпизода: сброс `prev_shared=None` в `player_util` (train и test пути).
- Восстановление/loss: `num_inputs=1` делает `x_restored` и `G_t` одноканальными — косинус
  считается на согласованных тензорах.
