# ECG Signal Segmentation using 1D U-Net

Проект реализует пайплайн сегментации ЭКГ-сигналов с использованием 1D U-Net на PyTorch.

Модель работает с многоканальными ЭКГ-сигналами и предназначена для поиска следующих сегментов:

- обычный QRS;
- SPIKES;
- QRS после SPIKE.

## Возможности

- обработка ЭКГ-сигналов до 12 каналов;
- обучение 1D U-Net;
- работа со скользящими окнами;
- нормализация сигнала;
- поддержка разметки в JSON и масок в `.npy`;
- постобработка найденных сегментов;
- сохранение результата в JSON;
- визуализация сигнала, вероятностей и предсказанных сегментов.

## Структура проекта

```text
wave-segmentation/
├── checkpoints/              # сохранённые модели
├── data/                     # данные
│   ├── data_with_spikes/     # сигналы и JSON-разметка со SPIKES/QRS_AFTER_SPIKE
│   ├── segmentation/         # сигналы и npy-маски с QRS
│   └── segmentation_kvachadze_npy/
├── datasets/                 # Dataset для обучения
├── ecg_signal_processor/     # загрузка сигналов и JSON-разметки
├── models/                   # архитектура UNet1D
├── scripts/                  # обучение, экспорт и инференс
├── utils/                    # метрики и вспомогательные функции
├── config.py                 # основные параметры
└── tox.ini                   # запуск Ruff через tox
```

## Классы модели

Внутри модели используется единая схема классов:

```text
0 -> background
1 -> QRS
2 -> SPIKES
3 -> QRS_AFTER_SPIKE
```

Для JSON-разметки и `.npy`-масок классы преобразуются в `datasets/ecg_dataset.py`.

## Настройки

Основные параметры находятся в `config.py`:

```python
WINDOW = 512
STEP = 64
BATCH_SIZE = 32
EPOCHS = 30
LR = 5e-5
DEVICE = "cuda"
SEED = 42
```

## Обучение

Запуск обучения:

```bash
python -m scripts.train
```

Лучшая модель сохраняется в:

```text
checkpoints/best_model.pth
```

## Инференс

Запуск предсказания:

```bash
python -m scripts.predict_unlabeled
```

Результат сохраняется в JSON-формате:

```text
data/data_with_spikes/prediction_markin/<file_id>.json
```

## Экспорт в ONNX

```bash
python -m scripts.export_onnx
```

После экспорта модель сохраняется в:

```text
checkpoints/best_model.onnx
```

## Проверка и автоформатирование кода

Для запуска Ruff через tox:

```bash
python -m tox -e lint
```

Автоисправление и форматирование:

```bash
python -m tox -e format
```

## Зависимости

Основные библиотеки:

- PyTorch;
- NumPy;
- scikit-learn;
- matplotlib;
- tqdm;
- Ruff;
- tox.

