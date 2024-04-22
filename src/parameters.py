from dataclasses import dataclass, field
from typing import List

@dataclass
class pid_params:
    pos_p: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.2])
    pos_i: List[int] = field(default_factory=lambda: [0, 0, 0])
    pos_d: List[int] = field(default_factory=lambda: [0, 0, 0])
    
    vel_p: List[int] = field(default_factory=lambda: [1, 1, 100])
    vel_i: List[float] = field(default_factory=lambda: [0.1, 0.1, 0])
    vel_d: List[int] = field(default_factory=lambda: [0, 0, 0])
    
    att_p: float = 5
    
    rate_p: float = 8
