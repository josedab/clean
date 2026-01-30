# ADR-0001: Cleanlab as Core Label Error Detection Engine

## Status

Accepted

## Context

Label errors are one of the most impactful data quality issues in ML datasets. Research shows that 1-10% of labels in major datasets (ImageNet, MNIST, CIFAR) are incorrect, and these errors directly degrade model performance. Clean needed a robust, research-backed approach to detect mislabeled samples.

Several approaches were considered:

1. **Build from scratch**: Implement confident learning algorithms internally
2. **Use Cleanlab**: Integrate the established cleanlab library as a dependency
3. **Ensemble approach**: Combine multiple simple heuristics (model disagreement, low confidence, etc.)
4. **LLM-based**: Use large language models to verify labels

Key requirements:
- High recall for label errors (catch most actual errors)
- Reasonable precision (avoid too many false positives)
- Works with any classifier (model-agnostic)
- Scientifically validated methodology
- Active maintenance and community support

## Decision

We adopted **Cleanlab** as the core engine for label error detection, wrapping it in our `LabelErrorDetector` class.

```python
# From detection/label_errors.py
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores

class LabelErrorDetector(BaseDetector):
    def detect(self, features, labels):
        # Use cleanlab's confident learning under the hood
        pred_probs = self._get_pred_probs(features, labels)
        issue_mask = find_label_issues(labels, pred_probs)
        quality_scores = get_label_quality_scores(labels, pred_probs)
        # ... wrap results in our LabelError dataclass
```

The wrapper provides:
- Consistent interface matching other Clean detectors
- Integration with our type system (`LabelError` dataclass)
- Cross-validation probability estimation when needed
- Configuration options exposed through our API

## Consequences

### Positive

- **Proven methodology**: Cleanlab implements "Confident Learning" from peer-reviewed research, providing state-of-the-art label error detection
- **Faster development**: Avoided 3-6 months of algorithm R&D and validation
- **Community trust**: Cleanlab has 8k+ GitHub stars and is used in production by major companies
- **Maintained dependency**: Active development means bug fixes and improvements flow to Clean automatically
- **Model-agnostic**: Works with any classifier that outputs probability estimates

### Negative

- **External dependency**: Cleanlab version updates could introduce breaking changes
- **Limited customization**: Deep algorithmic modifications require forking or contributing upstream
- **Dependency weight**: Cleanlab brings its own dependencies (though minimal)
- **Attribution complexity**: Users may not realize Clean uses Cleanlab internally

### Neutral

- **API alignment**: Our detector interface differs slightly from Cleanlab's native API, requiring a thin wrapper
- **Version pinning**: We pin `cleanlab>=2.0.0` to ensure API compatibility

## References

- [Cleanlab GitHub](https://github.com/cleanlab/cleanlab)
- [Confident Learning paper](https://arxiv.org/abs/1911.00068)
- [Finding Label Errors in ImageNet](https://labelerrors.com/)
