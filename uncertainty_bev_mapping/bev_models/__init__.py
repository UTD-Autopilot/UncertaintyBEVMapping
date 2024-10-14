from .baseline import Baseline
from .evidential import Evidential
from .baseline_topk import BaselineTopK
from .evidential_topk import EvidentialTopK

models = {
    'baseline': Baseline,
    'evidential': Evidential,
    'baseline_topk': BaselineTopK,
    'evidential_topk': EvidentialTopK,
}
