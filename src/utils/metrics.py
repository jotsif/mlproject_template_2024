from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import r2_score


def calc_metrics(y_true: NDArray[np.float32], y_pred: NDArray[np.float32]) -> Any:
    r2 = r2_score(y_true, y_pred)
    return {"r2": r2}
