from .baseline import Baseline
from .evidential import Evidential
from .evidenital_topk import EvidentialTopK

models = {
    'baseline': Baseline,
    'evidential': Evidential,
    'evidential_topk': EvidentialTopK,
}
