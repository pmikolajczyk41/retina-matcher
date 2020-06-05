from typing import NamedTuple


class GraphStats(NamedTuple):
    n: int
    m: int
    weight_sum: float
    leaves: int
    min_degree: int
    max_degree: int
    avg_degree: float
    med_degree: int
    std_degree: float
    diameter: float
    unweighted_diameter: int
    cc: int
    msf: float
