# Quality Predictor

ML-based quality score prediction for datasets using historical quality data.

## Quick Example

```python
from clean.quality_predictor import QualityPredictor, predict_quality

# Train on historical datasets
predictor = QualityPredictor()
predictor.fit(
    datasets=[df1, df2, df3, df4, df5],
    quality_scores=[0.7, 0.8, 0.6, 0.9, 0.75],
    label_columns=["label"] * 5
)

# Predict quality of new dataset
prediction = predictor.predict(new_df, label_column="label")
print(f"Predicted quality: {prediction.quality_score:.2f}")
print(f"Confidence: {prediction.confidence_level}")
```

## API Reference

### PredictorConfig

Configuration for the quality predictor.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_type` | str | `"gradient_boosting"` | Model type: `"gradient_boosting"`, `"random_forest"`, or `"linear"` |
| `n_estimators` | int | `100` | Number of estimators for ensemble models |
| `max_depth` | int | `10` | Maximum tree depth |
| `min_samples_for_training` | int | `5` | Minimum datasets required for training |
| `confidence_percentile` | float | `0.9` | Percentile for confidence intervals |
| `cache_features` | bool | `True` | Whether to cache extracted features |

### QualityPredictor

Main predictor class for ML-based quality estimation.

#### `__init__(config: PredictorConfig | None = None)`

Initialize the predictor with optional configuration.

#### `fit(datasets, quality_scores, label_columns=None, text_columns=None) -> QualityPredictor`

Train the predictor on historical data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `datasets` | `list[pd.DataFrame]` | List of training datasets |
| `quality_scores` | `list[float]` | Known quality scores (0-1) |
| `label_columns` | `list[str \| None] \| None` | Label column for each dataset |
| `text_columns` | `list[list[str] \| None] \| None` | Text columns for each dataset |

#### `predict(data, label_column=None, text_columns=None) -> QualityPrediction`

Predict quality score for a new dataset.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pd.DataFrame` | Dataset to evaluate |
| `label_column` | `str \| None` | Label column name |
| `text_columns` | `list[str] \| None` | Text column names |

### QualityPrediction

Prediction result dataclass.

| Field | Type | Description |
|-------|------|-------------|
| `quality_score` | float | Predicted quality score (0-1) |
| `confidence` | float | Prediction confidence (0-1) |
| `confidence_level` | str | `"low"`, `"medium"`, or `"high"` |
| `confidence_interval` | tuple[float, float] | Lower and upper bounds |
| `prediction_time_ms` | float | Time taken for prediction |
| `features_used` | list[str] | Features used in prediction |
| `feature_importances` | dict[str, float] | Feature importance scores |
| `warnings` | list[str] | Any warnings generated |

#### `to_dict() -> dict`

Convert prediction to dictionary.

### Convenience Functions

#### `predict_quality(data, predictor, label_column=None, text_columns=None) -> QualityPrediction`

Quick prediction using an existing predictor.

```python
from clean.quality_predictor import predict_quality

prediction = predict_quality(df, trained_predictor, label_column="target")
```

## Example Workflows

### Training with Multiple Data Types

```python
from clean.quality_predictor import QualityPredictor, PredictorConfig

config = PredictorConfig(
    model_type="random_forest",
    n_estimators=200,
    max_depth=15
)

predictor = QualityPredictor(config=config)
predictor.fit(
    datasets=training_datasets,
    quality_scores=known_scores,
    label_columns=label_cols,
    text_columns=text_cols  # For datasets with text
)
```

### Batch Prediction

```python
predictions = []
for df in new_datasets:
    pred = predictor.predict(df, label_column="label")
    predictions.append({
        "score": pred.quality_score,
        "confidence": pred.confidence_level,
        "warnings": pred.warnings
    })
```
