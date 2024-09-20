from .baseline import Baseline
from .evidential import Evidential
from .baseline_topk import BaselineTopK

models = {
    'baseline': Baseline,
    'evidential': Evidential,
    'baseline_topk': BaselineTopK,
}
