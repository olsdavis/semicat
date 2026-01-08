"""
Utils for Tensor shapes.
"""

from torch import Tensor


def view_for(t: Tensor, other: Tensor) -> Tensor:
    """
    Reshape tensor `t` to be broadcastable with `other`.
    """
    return t.view(other.size(0), *((1,) * (other.ndim - 1)))
