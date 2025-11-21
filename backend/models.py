from pydantic import BaseModel
from typing import List, Optional, Dict, Union

class OptimizationRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    strategy: str
    risk_free_rate: float = 0.02
    target_return: Optional[float] = None
    target_volatility: Optional[float] = 0.10
    mar: Optional[float] = 0.0  # Minimum Acceptable Return
    confidence_level: Optional[float] = 0.95
    benchmark_ticker: Optional[str] = "SPY"
    constraints: Dict[str, Union[float, bool]] = {} # min_weight, max_weight, etc.

class OptimizationResponse(BaseModel):
    weights: Dict[str, float]
    performance: Dict[str, float]
    metrics: Dict[str, float]
    frontier_volatility: Optional[List[float]] = None
    frontier_returns: Optional[List[float]] = None

class DataRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
