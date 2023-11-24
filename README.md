# CS454 Team 4 Project

We are Team 4 (Suhwan Kim, Jaeduk Seo, Minwoo Noh, Heejun Lee).

In this project, we are doing git commit defect detection.

```sh
PYTHONPATH=./ python src/trainer/trainer.py --model codebert_test_predictor --data_path ...
```

## How To

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