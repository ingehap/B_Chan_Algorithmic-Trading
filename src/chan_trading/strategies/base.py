from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """Abstract strategy interface."""

    @abstractmethod
    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return target positions indexed like `prices`."""
        raise NotImplementedError
