from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from .models import OptimizationRequest, OptimizationResponse, DataRequest
from .data_loader import fetch_data
from .optimizer import PortfolioOptimizer
from .report_generator import ReportGenerator

app = FastAPI(title="Portfolio Optimization Engine")

@app.post("/load-data")
def load_data_endpoint(request: DataRequest):
    try:
        data = fetch_data(request.tickers, request.start_date, request.end_date)
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found")
        return {"message": "Data loaded successfully", "assets": list(data.columns), "rows": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize", response_model=OptimizationResponse)
def optimize_endpoint(request: OptimizationRequest):
    try:
        # Fetch Data
        prices = fetch_data(request.tickers, request.start_date, request.end_date)
        if prices.empty:
            raise HTTPException(status_code=404, detail="No data found")
            
        # Drop NaNs for common history
        prices = prices.dropna(axis=0, how='any')
        if prices.empty:
            raise HTTPException(status_code=400, detail="Insufficient common history")

        optimizer = PortfolioOptimizer(prices)
        
        # Extract constraints
        min_weight = request.constraints.get("min_weight", 0.0)
        max_weight = request.constraints.get("max_weight", 1.0)
        weight_bounds = (min_weight, max_weight)

        weights = {}
        performance = ()
        
        # Select Strategy
        if request.strategy == "Mean Variance - Maximize Sharpe Ratio":
            weights, performance = optimizer.optimize_mean_variance(objective='max_sharpe', risk_free_rate=request.risk_free_rate, weight_bounds=weight_bounds)
        elif request.strategy == "Mean Variance - Minimize Volatility":
            weights, performance = optimizer.optimize_mean_variance(objective='min_volatility', weight_bounds=weight_bounds)
        elif request.strategy == "Mean Variance - Maximize Return at Target Volatility":
            weights, performance = optimizer.optimize_target_volatility(target_volatility=request.target_volatility, weight_bounds=weight_bounds)
        elif request.strategy == "Equal Weighted":
            weights, performance = optimizer.optimize_equal_weighted()
        elif request.strategy == "Conditional Value-at-Risk (CVaR)":
            weights, performance = optimizer.optimize_cvar(weight_bounds=weight_bounds)
        elif request.strategy == "Risk Parity":
            weights, performance = optimizer.optimize_risk_parity()
        elif request.strategy == "Kelly Criterion":
            weights, performance = optimizer.optimize_kelly(weight_bounds=weight_bounds)
        elif request.strategy == "Sortino Ratio":
            weights, performance = optimizer.optimize_sortino(risk_free_rate=request.risk_free_rate, mar=request.mar, weight_bounds=weight_bounds)
        elif request.strategy == "Omega Ratio":
            weights, performance = optimizer.optimize_omega(mar=request.mar, weight_bounds=weight_bounds)
        elif request.strategy == "Maximum Drawdown":
            weights, performance = optimizer.optimize_max_drawdown(weight_bounds=weight_bounds)
        elif request.strategy == "Tracking Error":
             if not request.benchmark_ticker:
                 raise HTTPException(status_code=400, detail="Benchmark ticker required for Tracking Error")
             benchmark_data = fetch_data([request.benchmark_ticker], request.start_date, request.end_date)
             weights, performance = optimizer.optimize_tracking_error(benchmark_data, weight_bounds=weight_bounds)
        elif request.strategy == "Information Ratio":
             if not request.benchmark_ticker:
                 raise HTTPException(status_code=400, detail="Benchmark ticker required for Information Ratio")
             benchmark_data = fetch_data([request.benchmark_ticker], request.start_date, request.end_date)
             weights, performance = optimizer.optimize_information_ratio(benchmark_data, weight_bounds=weight_bounds)
        else:
            raise HTTPException(status_code=400, detail="Unknown strategy")

        # Metrics Mapping
        metrics = {
            "Expected Return": performance[0],
            "Volatility": performance[1],
            "Sharpe Ratio": performance[2] if len(performance) > 2 else 0.0
        }
        
        # Comprehensive Metrics
        benchmark_data = fetch_data([request.benchmark_ticker], request.start_date, request.end_date)
        if benchmark_data.empty:
             # Fallback if benchmark fetch fails or not provided (though default is SPY)
             benchmark_data = pd.DataFrame(index=prices.index, columns=[request.benchmark_ticker])
             benchmark_data.fillna(0, inplace=True) # Should ideally handle better

        comprehensive_metrics = optimizer.calculate_comprehensive_metrics(
            weights, 
            benchmark_data, 
            risk_free_rate=request.risk_free_rate
        )
        
        # Frontier (only for MVO for now)
        frontier_vol = []
        frontier_ret = []
        if "Mean Variance" in request.strategy:
             frontier_vol, frontier_ret = optimizer.get_efficient_frontier_points()

        return OptimizationResponse(
            weights=weights,
            performance=metrics,
            metrics=comprehensive_metrics,
            frontier_volatility=frontier_vol,
            frontier_returns=frontier_ret
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
