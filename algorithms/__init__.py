import os
from . import *

ALGORITHMS = {
    "K-Nearest Neighbor": {
        "fields": ["Chiều dài đài hoa (cm) [Số]", "Chiều rộng đài hoa (cm) [Số]",
                   "Chiều dài cánh hoa (cm) [Số]", "Chiều rộng cánh hoa (cm) [Số]"],
        "path": os.path.join(os.path.dirname(__file__), "knn.py"),
    },
}
