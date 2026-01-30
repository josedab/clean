# Detectors

Clean provides several detectors for different types of data quality issues.

## LabelErrorDetector

Detects label errors using confident learning.

::: clean.detection.label_errors.LabelErrorDetector
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - fit
        - detect
        - fit_detect

## DuplicateDetector

Detects exact and near-duplicate samples.

::: clean.detection.duplicates.DuplicateDetector
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - fit
        - detect
        - fit_detect

## OutlierDetector

Detects outliers using multiple methods.

::: clean.detection.outliers.OutlierDetector
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - fit
        - detect
        - fit_detect

## ImbalanceDetector

Detects class imbalance issues.

::: clean.detection.imbalance.ImbalanceDetector
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - fit
        - detect
        - fit_detect

## BiasDetector

Detects bias and fairness issues.

::: clean.detection.bias.BiasDetector
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - fit
        - detect
        - fit_detect

## DetectorResult

All detectors return a `DetectorResult` object:

::: clean.detection.base.DetectorResult
    options:
      show_root_heading: true
      show_source: false
