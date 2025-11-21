from pypfopt.efficient_frontier import EfficientFrontier, EfficientCVaR
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt import risk_models, expected_returns, objective_functions
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import pypfopt


class PortfolioOptimizer:
    def __init__(self, prices):
        self.prices = prices
        self.mu = expected_returns.mean_historical_return(prices)
        self.S = risk_models.sample_cov(prices)

    def optimize_mean_variance(self, objective='max_sharpe', risk_free_rate=0.02, weight_bounds=(0, 1)):
        """
        Mean-Variance Optimization.
        objective: 'max_sharpe' or 'min_volatility'
        weight_bounds: tuple (min, max) or list of tuples
        """
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=weight_bounds)
        
        if objective == 'max_sharpe':
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif objective == 'min_volatility':
            weights = ef.min_volatility()
        else:
            raise ValueError("Unknown objective for Mean-Variance")
        
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        return cleaned_weights, performance


    def optimize_cvar(self, target_return=None, weight_bounds=(0, 1)):
        """
        Conditional Value-at-Risk (CVaR) Optimization.
        Minimizes CVaR.
        weight_bounds: tuple (min, max) or list of tuples
        """
        # EfficientCVaR requires returns, not prices, but PyPortfolioOpt handles it? 
        # Actually EfficientCVaR takes expected returns and returns series usually.
        # Let's check docs usage. It usually needs historical returns.
        returns = self.prices.pct_change().dropna()
        ec = EfficientCVaR(self.mu, returns, weight_bounds=weight_bounds)
        
        if target_return:
            ec.efficient_return(target_return)
        else:
            ec.min_cvar()
            
        cleaned_weights = ec.clean_weights()
        performance = ec.portfolio_performance(verbose=False)
        return cleaned_weights, performance


    def optimize_risk_parity(self):
        """
        Hierarchical Risk Parity (HRP).
        """
        returns = self.prices.pct_change().dropna()
        hrp = HRPOpt(returns)
        weights = hrp.optimize()
        cleaned_weights = hrp.clean_weights()
        performance = hrp.portfolio_performance(verbose=False)
        return cleaned_weights, performance

    def get_efficient_frontier_points(self, num_points=100):
        """
        Generates points for the Efficient Frontier.
        Returns:
            list of (volatility, return) tuples.
        """
        ef = EfficientFrontier(self.mu, self.S)
        
        # Find min vol and max sharpe to set bounds
        min_vol_ef = EfficientFrontier(self.mu, self.S)
        min_vol_ef.min_volatility()
        min_vol_ret = min_vol_ef.portfolio_performance()[0]
        
        max_sharpe_ef = EfficientFrontier(self.mu, self.S)
        max_sharpe_ef.max_sharpe()
        max_sharpe_ret = max_sharpe_ef.portfolio_performance()[0]
        
        # Generate target returns from min vol to max return (approx)
        # We can go a bit higher than max_sharpe_ret if possible, up to max asset return
        max_asset_ret = self.mu.max()
        target_returns = np.linspace(min_vol_ret, max_asset_ret, num_points)
        
        frontier_volatility = []
        frontier_returns = []
        
        for r in target_returns:
            try:
                ef = EfficientFrontier(self.mu, self.S)
                ef.efficient_return(target_return=r)
                perf = ef.portfolio_performance(verbose=False)
                frontier_volatility.append(perf[1])
                frontier_returns.append(perf[0])
            except:
                pass
                
        return frontier_volatility, frontier_returns

    def optimize_kelly(self, weight_bounds=(0, 1)):
        """
        Maximize Kelly Criterion (approximate as Maximize Geometric Mean Return).
        Obj = mu - 0.5 * sigma^2
        """
        from scipy.optimize import minimize
        
        num_assets = len(self.mu)
        args = (self.mu, self.S)
        
        def kelly_objective(weights, mu, S):
            # Maximize: r - 0.5 * variance (approx log utility)
            # We minimize negative of that
            ret = np.sum(mu * weights)
            var = np.dot(weights.T, np.dot(S, weights))
            return -(ret - 0.5 * var)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [weight_bounds for _ in range(num_assets)]
        initial_guess = num_assets * [1. / num_assets,]
        
        result = minimize(kelly_objective, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = dict(zip(self.prices.columns, result.x))
        performance = self._get_performance(result.x)
        return weights, performance

    def optimize_sortino(self, risk_free_rate=0.02, mar=0.0, weight_bounds=(0, 1)):
        """
        Maximize Sortino Ratio.
        """
        from scipy.optimize import minimize
        
        returns = self.prices.pct_change().dropna()
        num_assets = len(self.mu)
        
        def sortino_objective(weights, returns, mar):
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_daily_rets = returns.dot(weights)
            
            # Downside deviation
            downside_rets = portfolio_daily_rets[portfolio_daily_rets < mar/252] # Adjust MAR to daily? Or assume MAR is annual.
            # Let's assume MAR is annual, so daily MAR approx MAR/252
            # Actually, let's stick to annual metrics for consistency
            
            # Calculate daily downside deviation
            # Downside deviation = sqrt(mean(min(0, R - MAR)^2)) * sqrt(252)
            # Here we use a simpler version: std dev of negative returns relative to MAR
            
            downside_diff = portfolio_daily_rets - (mar/252)
            downside_diff[downside_diff > 0] = 0
            downside_deviation = np.sqrt(np.mean(downside_diff**2)) * np.sqrt(252)
            
            if downside_deviation == 0:
                return 1e6 # Avoid division by zero
                
            sortino = (portfolio_return - risk_free_rate) / downside_deviation
            return -sortino
            
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [weight_bounds for _ in range(num_assets)]
        initial_guess = num_assets * [1. / num_assets,]
        
        result = minimize(sortino_objective, initial_guess, args=(returns, mar), method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = dict(zip(self.prices.columns, result.x))
        performance = self._get_performance(result.x)
        return weights, performance

    def optimize_omega(self, mar=0.0, weight_bounds=(0, 1)):
        """
        Maximize Omega Ratio.
        Omega = Sum(P(Gain)) / Sum(P(Loss)) relative to MAR
        """
        from scipy.optimize import minimize
        
        returns = self.prices.pct_change().dropna()
        num_assets = len(self.mu)
        
        def omega_objective(weights, returns, mar):
            portfolio_daily_rets = returns.dot(weights)
            threshold = mar / 252
            
            gains = portfolio_daily_rets[portfolio_daily_rets > threshold] - threshold
            losses = threshold - portfolio_daily_rets[portfolio_daily_rets < threshold]
            
            sum_gains = np.sum(gains)
            sum_losses = np.sum(losses)
            
            if sum_losses == 0:
                return -1e6
            
            omega = sum_gains / sum_losses
            return -omega

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [weight_bounds for _ in range(num_assets)]
        initial_guess = num_assets * [1. / num_assets,]
        
        result = minimize(omega_objective, initial_guess, args=(returns, mar), method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = dict(zip(self.prices.columns, result.x))
        performance = self._get_performance(result.x)
        return weights, performance

    def optimize_max_drawdown(self, weight_bounds=(0, 1)):
        """
        Minimize Maximum Drawdown.
        """
        from scipy.optimize import minimize
        
        returns = self.prices.pct_change().dropna()
        num_assets = len(self.mu)
        
        def mdd_objective(weights, returns):
            portfolio_daily_rets = returns.dot(weights)
            cum_rets = (1 + portfolio_daily_rets).cumprod()
            peak = cum_rets.cummax()
            drawdown = (cum_rets - peak) / peak
            max_drawdown = drawdown.min() # Negative value
            return -max_drawdown # Minimize positive magnitude of MDD (which is minimizing -MDD? No, MDD is negative, so min MDD means closest to 0. Wait. MDD is usually expressed as positive % drop. Let's say MDD is 0.2 (20%). We want to minimize 0.2.
            # My calc: drawdown is negative e.g. -0.2. min() gives -0.2.
            # We want to maximize the (negative) drawdown value (closest to 0).
            # So we minimize -1 * (negative value) = positive value.
            # Or just minimize abs(min_drawdown).
            
            return abs(max_drawdown)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [weight_bounds for _ in range(num_assets)]
        initial_guess = num_assets * [1. / num_assets,]
        
        result = minimize(mdd_objective, initial_guess, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = dict(zip(self.prices.columns, result.x))
        performance = self._get_performance(result.x)
        return weights, performance

    def optimize_tracking_error(self, benchmark_prices, weight_bounds=(0, 1)):
        """
        Minimize Tracking Error against benchmark.
        """
        from scipy.optimize import minimize
        
        returns = self.prices.pct_change().dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()
        
        # Align
        common_dates = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates].iloc[:, 0] # Assume 1 col
        
        num_assets = len(self.mu)
        
        def te_objective(weights, returns, benchmark_returns):
            portfolio_rets = returns.dot(weights)
            diff = portfolio_rets - benchmark_returns
            te = np.std(diff) * np.sqrt(252)
            return te

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [weight_bounds for _ in range(num_assets)]
        initial_guess = num_assets * [1. / num_assets,]
        
        result = minimize(te_objective, initial_guess, args=(returns, benchmark_returns), method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = dict(zip(self.prices.columns, result.x))
        performance = self._get_performance(result.x)
        return weights, performance

    def optimize_information_ratio(self, benchmark_prices, weight_bounds=(0, 1)):
        """
        Maximize Information Ratio.
        IR = (Rp - Rb) / TE
        """
        from scipy.optimize import minimize
        
        returns = self.prices.pct_change().dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()
        
        # Align
        common_dates = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates].iloc[:, 0]
        
        num_assets = len(self.mu)
        
        def ir_objective(weights, returns, benchmark_returns):
            portfolio_rets = returns.dot(weights)
            diff = portfolio_rets - benchmark_returns
            te = np.std(diff) * np.sqrt(252)
            
            active_return = (portfolio_rets.mean() - benchmark_returns.mean()) * 252
            
            if te == 0:
                return 1e6
                
            ir = active_return / te
            return -ir

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [weight_bounds for _ in range(num_assets)]
        initial_guess = num_assets * [1. / num_assets,]
        
        result = minimize(ir_objective, initial_guess, args=(returns, benchmark_returns), method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = dict(zip(self.prices.columns, result.x))
        performance = self._get_performance(result.x)
        return weights, performance

    def optimize_target_volatility(self, target_volatility=0.10, weight_bounds=(0, 1)):
        """
        Maximize Return subject to a target volatility constraint.
        """
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=weight_bounds)
        weights = ef.efficient_risk(target_volatility)
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=0.02)
        return cleaned_weights, performance

    def optimize_equal_weighted(self):
        """
        Equal weighted portfolio.
        """
        num_assets = len(self.prices.columns)
        weights = {ticker: 1.0/num_assets for ticker in self.prices.columns}
        
        # Calculate performance
        w_array = np.array(list(weights.values()))
        performance = self._get_performance(w_array)
        
        return weights, performance

    def _get_performance(self, weights):
        """
        Calculate standard performance metrics for a given weight vector.
        Returns: (Expected Return, Volatility, Sharpe Ratio)
        """
        # Ensure weights is a numpy array
        if isinstance(weights, dict):
            weights = np.array([weights[ticker] for ticker in self.prices.columns])
            
        ret = np.sum(self.mu * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(self.S, weights)))
        sharpe = (ret - 0.02) / vol # Assuming 2% risk free for now
        return (ret, vol, sharpe)

    def calculate_comprehensive_metrics(self, weights, benchmark_prices, risk_free_rate=0.02, start_balance=10000):
        """
        Calculate a comprehensive set of portfolio metrics.
        """
        # Portfolio Returns
        portfolio_daily_rets = (self.prices.pct_change().dropna() * weights).sum(axis=1)
        
        # Benchmark Returns
        benchmark_daily_rets = benchmark_prices.pct_change().dropna().iloc[:, 0]
        
        # Align Dates
        common_dates = portfolio_daily_rets.index.intersection(benchmark_daily_rets.index)
        portfolio_daily_rets = portfolio_daily_rets.loc[common_dates]
        benchmark_daily_rets = benchmark_daily_rets.loc[common_dates]
        
        # Basic Calculations
        total_return = (1 + portfolio_daily_rets).prod() - 1
        n_years = len(portfolio_daily_rets) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1
        
        end_balance = start_balance * (1 + total_return)
        
        annual_vol = portfolio_daily_rets.std() * np.sqrt(252)
        expected_ret = portfolio_daily_rets.mean() * 252
        
        # Drawdown
        cum_rets = (1 + portfolio_daily_rets).cumprod()
        peak = cum_rets.cummax()
        drawdown = (cum_rets - peak) / peak
        max_drawdown = drawdown.min()
        
        # Best/Worst Year
        portfolio_annual_rets = portfolio_daily_rets.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        best_year = portfolio_annual_rets.max()
        worst_year = portfolio_annual_rets.min()
        
        # Risk Metrics
        sharpe = (cagr - risk_free_rate) / annual_vol
        
        downside_rets = portfolio_daily_rets[portfolio_daily_rets < 0]
        downside_std = downside_rets.std() * np.sqrt(252)
        sortino = (cagr - risk_free_rate) / downside_std if downside_std != 0 else 0
        
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Regression Metrics (Alpha, Beta, R2)
        # y = alpha + beta * x
        slope, intercept, r_value, p_value, std_err = stats.linregress(benchmark_daily_rets, portfolio_daily_rets)
        beta = slope
        alpha = intercept * 252 # Annualized Alpha
        r_squared = r_value ** 2
        
        treynor = (cagr - risk_free_rate) / beta if beta != 0 else 0
        
        # Active Metrics
        active_rets = portfolio_daily_rets - benchmark_daily_rets
        active_return = active_rets.mean() * 252
        tracking_error = active_rets.std() * np.sqrt(252)
        information_ratio = active_return / tracking_error if tracking_error != 0 else 0
        
        # VaR / CVaR (Historical 95%)
        var_95 = np.percentile(portfolio_daily_rets, 5)
        cvar_95 = portfolio_daily_rets[portfolio_daily_rets <= var_95].mean()
        
        # Skew / Kurtosis
        skew = stats.skew(portfolio_daily_rets)
        kurtosis = stats.kurtosis(portfolio_daily_rets)
        
        # Capture Ratios
        up_market = benchmark_daily_rets[benchmark_daily_rets > 0]
        down_market = benchmark_daily_rets[benchmark_daily_rets <= 0]
        
        up_portfolio = portfolio_daily_rets[benchmark_daily_rets > 0]
        down_portfolio = portfolio_daily_rets[benchmark_daily_rets <= 0]
        
        up_capture = (up_portfolio.mean() / up_market.mean()) * 100 if up_market.mean() != 0 else 0
        down_capture = (down_portfolio.mean() / down_market.mean()) * 100 if down_market.mean() != 0 else 0
        
        # Gain/Loss Ratio
        avg_gain = portfolio_daily_rets[portfolio_daily_rets > 0].mean()
        avg_loss = portfolio_daily_rets[portfolio_daily_rets < 0].mean()
        gain_loss_ratio = avg_gain / abs(avg_loss) if avg_loss != 0 else 0
        
        metrics = {
            "Start Balance": start_balance,
            "End Balance": end_balance,
            "Annualized Return (CAGR)": cagr,
            "Expected Return": expected_ret,
            "Standard Deviation": annual_vol,
            "Downside Deviation": downside_std,
            "Best Year": best_year,
            "Worst Year": worst_year,
            "Maximum Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Treynor Ratio": treynor,
            "Calmar Ratio": calmar,
            "Gain/Loss Ratio": gain_loss_ratio,
            "Alpha": alpha,
            "Beta": beta,
            "R-squared": r_squared,
            "Active Return": active_return,
            "Tracking Error": tracking_error,
            "Information Ratio": information_ratio,
            "VaR (95%)": var_95,
            "CVaR (95%)": cvar_95,
            "Skewness": skew,
            "Excess Kurtosis": kurtosis,
            "Upside Capture Ratio": up_capture,
            "Downside Capture Ratio": down_capture
        }
        
        return metrics


