# CS454 Team 4 Project

We are Team 4 (Suhwan Kim, Jaeduk Seo, Minwoo Noh, Heejun Lee).

In this project, we are doing git commit defect detection.

```sh
PYTHONPATH=./ python src/trainer/trainer.py --model codebert_test_predictor --data_path ...
```

## Datasets

1. V1, V2: `cache/data_basic/data.json`
2. V3: `cache/data_commit_meta_natural/data_ex.json`, `cache/data_commit_meta/data_ex.json`

## How To

### Replicate The Experiments

```sh
# V1
PYTHONPATH=./ python src/trainer/trainer.py --data_path ./cache/data.json --allow_oracle_past_state --window_size 5

# V2
PYTHONPATH=./ python src/trainer/trainer.py --model codebert_atten --past_commit_filter only_buggy --experiment_name codebert_only_buggy --window_size 10 --allow_oracle_past_state --data_path ./cache/data.json

# V3
## V3: Natural format prompt
PYTHOPATH=./ python src/trainer/trainer.py --model codebert_atten --past_commit_filter only_buggy --experiment_name codebert_only_buggy_commit_meta_natural --window_size 20 --allow_oracle_past_state --data_path ./cache/data_commit_meta_natural/data_ex.json --batch_size 2 --gradient_accumulation_steps 4

## V3: JSON format prompt
PYTHONPATH=./ python src/trainer/trainer.py --model codebert_atten --past_commit_filter only_buggy --experiment_name codebert_only_buggy_commit_meta --window_size 10 --allow_oracle_past_state --data_path ./cache/data_commit_meta/data_ex.json
```

Tested system: `Ryzen 3950x, 64GB RAM, RTX 4090 24GB VRAM`.

### How to add your model?

1. Create model file in `src/models/__your_model__.py`.

For example, look at `src/models/codebert.py`

2. After implement model, register your model creation function into registry.

For example, look at `src/models/codebert.py:codebert_test_predictor` and `src/models/codebert.py:codebert_test_predictor_lstm_l2`

```py
# Simple example
... your model code ...

from src.models.registry import register

@register("your_model_name_and_options")
def your_model_name_and_options():
    return YourModel(
        options=blah blah
    )

@register("your_model_name_and_other_options")
def your_model_name_and_other_options():
    return YourModel(
        options=foobar
    )
```

After create register function, you have to import your model file in `src/models/__init__.py`.

```py
from .__your_model__ import YourModel
```

3. Now your model is visible in trainer.

```sh
PYTHONPATH=./ python src/trainer/trainer.py --model your_model_name_and_options
```

### How to change dataset?

You can pass json path into trainer.
```sh
PYTHONPATH=./ python src/trainer/trainer.py --data_path ./path/to/data.json
```