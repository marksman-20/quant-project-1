import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

# API URL
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Portfolio Optimization Tool", layout="wide")

# Sidebar Inputs
st.sidebar.header("Portfolio Optimization Configuration")

# Portfolio Type (Tickers)
st.sidebar.subheader("Portfolio Type")
tickers_input = st.sidebar.text_area("Tickers (comma separated)", "AAPL, MSFT, GOOG, AMZN, RELIANCE.NS")
tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]

# Time Period
st.sidebar.subheader("Time Period")
start_year = st.sidebar.number_input("Start Year", min_value=1985, max_value=2025, value=2015)
end_year = st.sidebar.number_input("End Year", min_value=1985, max_value=2025, value=2025)

start_date = f"{start_year}-01-01"
end_date = f"{end_year}-12-31"

# Optimization Goal
st.sidebar.subheader("Optimization Goal")
goal = st.sidebar.selectbox("Goal", [
    "Mean Variance - Maximize Sharpe Ratio",
    "Mean Variance - Minimize Volatility",
    "Mean Variance - Maximize Return at Target Volatility",
    "Conditional Value-at-Risk (CVaR)",
    "Risk Parity",
    "Kelly Criterion",
    "Sortino Ratio",
    "Omega Ratio",
    "Maximum Drawdown",
    "Tracking Error",
    "Information Ratio"
])

# Strategy Specific Inputs
risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.02, step=0.01)
mar = 0.0
if goal in ["Sortino Ratio", "Omega Ratio"]:
    mar = st.sidebar.number_input("Minimum Acceptable Return (MAR)", value=0.0, step=0.01)

target_volatility = 0.10
if goal == "Mean Variance - Maximize Return at Target Volatility":
    target_volatility = st.sidebar.number_input("Target Volatility", value=0.10, step=0.01)

benchmark_ticker = "SPY"
if goal in ["Tracking Error", "Information Ratio"]:
    benchmark_ticker = st.sidebar.text_input("Benchmark Ticker", "SPY")

# Constraints
st.sidebar.subheader("Asset Constraints")
use_constraints = st.sidebar.checkbox("Use Constraints", value=True)
min_weight = 0.0
max_weight = 1.0
if use_constraints:
    min_weight = st.sidebar.slider("Min Weight per Asset", 0.0, 1.0, 0.0, 0.01)
    max_weight = st.sidebar.slider("Max Weight per Asset", 0.0, 1.0, 1.0, 0.01)

# Run Button
if st.sidebar.button("Optimize Portfolio"):
    with st.spinner("Optimizing..."):
        # Construct Request Payload
        payload = {
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "strategy": goal,
            "risk_free_rate": risk_free_rate,
            "mar": mar,
            "target_volatility": target_volatility,
            "benchmark_ticker": benchmark_ticker,
            "constraints": {
                "min_weight": min_weight if use_constraints else 0.0,
                "max_weight": max_weight if use_constraints else 1.0
            }
        }
        
        try:
            response = requests.post(f"{API_URL}/optimize", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                weights = result["weights"]
                performance = result["performance"]
                frontier_vol = result.get("frontier_volatility", [])
                frontier_ret = result.get("frontier_returns", [])
                
                # Display Results
                st.header("Optimization Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Portfolio Allocation")
                    
                    # Pie Chart
                    labels = list(weights.keys())
                    values = list(weights.values())
                    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
                    fig_pie.update_layout(title_text="Portfolio Allocation")
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    st.subheader("Weights")
                    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
                    st.dataframe(weights_df.style.format("{:.2%}"))

                with col2:
                    st.subheader("Performance Metrics")
                    
                    metrics = result.get("metrics", {})
                    
                    # Format metrics for display
                    def fmt_pct(val): return f"{val:.2%}" if val is not None else "N/A"
                    def fmt_num(val): return f"{val:.2f}" if val is not None else "N/A"
                    def fmt_curr(val): return f"${val:,.2f}" if val is not None else "N/A"
                    
                    display_data = {
                        "Start Balance": fmt_curr(metrics.get("Start Balance")),
                        "End Balance": fmt_curr(metrics.get("End Balance")),
                        "Annualized Return (CAGR)": fmt_pct(metrics.get("Annualized Return (CAGR)")),
                        "Expected Return": fmt_pct(metrics.get("Expected Return")),
                        "Standard Deviation": fmt_pct(metrics.get("Standard Deviation")),
                        "Downside Deviation": fmt_pct(metrics.get("Downside Deviation")),
                        "Best Year": fmt_pct(metrics.get("Best Year")),
                        "Worst Year": fmt_pct(metrics.get("Worst Year")),
                        "Maximum Drawdown": fmt_pct(metrics.get("Maximum Drawdown")),
                        "Sharpe Ratio": fmt_num(metrics.get("Sharpe Ratio")),
                        "Sortino Ratio": fmt_num(metrics.get("Sortino Ratio")),
                        "Treynor Ratio": fmt_num(metrics.get("Treynor Ratio")),
                        "Calmar Ratio": fmt_num(metrics.get("Calmar Ratio")),
                        "Gain/Loss Ratio": fmt_num(metrics.get("Gain/Loss Ratio")),
                        "Alpha": fmt_pct(metrics.get("Alpha")),
                        "Beta": fmt_num(metrics.get("Beta")),
                        "R-squared": fmt_num(metrics.get("R-squared")),
                        "Active Return": fmt_pct(metrics.get("Active Return")),
                        "Tracking Error": fmt_pct(metrics.get("Tracking Error")),
                        "Information Ratio": fmt_num(metrics.get("Information Ratio")),
                        "VaR (95%)": fmt_pct(metrics.get("VaR (95%)")),
                        "CVaR (95%)": fmt_pct(metrics.get("CVaR (95%)")),
                        "Skewness": fmt_num(metrics.get("Skewness")),
                        "Excess Kurtosis": fmt_num(metrics.get("Excess Kurtosis")),
                        "Upside Capture Ratio": fmt_num(metrics.get("Upside Capture Ratio")),
                        "Downside Capture Ratio": fmt_num(metrics.get("Downside Capture Ratio"))
                    }
                    
                    metrics_df = pd.DataFrame.from_dict(display_data, orient='index', columns=['Optimized Portfolio'])
                    st.table(metrics_df)
                    
                    # Efficient Frontier
                    if frontier_vol and frontier_ret:
                        st.subheader("Efficient Frontier")
                        fig_ef = go.Figure()
                        fig_ef.add_trace(go.Scatter(x=frontier_vol, y=frontier_ret, mode='lines', name='Efficient Frontier'))
                        fig_ef.add_trace(go.Scatter(x=[performance.get('Volatility')], y=[performance.get('Expected Return')], mode='markers', name='Optimized Portfolio', marker=dict(color='red', size=12, symbol='star')))
                        fig_ef.update_layout(title='Efficient Frontier', xaxis_title='Volatility', yaxis_title='Return')
                        st.plotly_chart(fig_ef, use_container_width=True)

                    # --- Report Generation Section ---
                    st.markdown("---")
                    st.header("Report Generation")
                    
                    if st.button("Generate Detailed Report"):
                        with st.spinner("Generating Report..."):
                            # 1. Fetch Equal Weighted Benchmark
                            payload_eq = payload.copy()
                            payload_eq["strategy"] = "Equal Weighted"
                            resp_eq = requests.post(f"{API_URL}/optimize", json=payload_eq)
                            
                            if resp_eq.status_code == 200:
                                res_eq = resp_eq.json()
                                weights_eq = res_eq["weights"]
                                perf_eq = res_eq["performance"]
                                
                                # 2. Create Comparison Charts
                                
                                # Pie Charts
                                fig_pie_opt = go.Figure(data=[go.Pie(labels=list(weights.keys()), values=list(weights.values()), hole=.3)])
                                fig_pie_opt.update_layout(title_text=f"Optimized: {goal}")
                                
                                fig_pie_eq = go.Figure(data=[go.Pie(labels=list(weights_eq.keys()), values=list(weights_eq.values()), hole=.3)])
                                fig_pie_eq.update_layout(title_text="Equal Weighted")
                                
                                col_r1, col_r2 = st.columns(2)
                                col_r1.plotly_chart(fig_pie_opt, use_container_width=True)
                                col_r2.plotly_chart(fig_pie_eq, use_container_width=True)
                                
                                # Growth Chart (Cumulative Returns)
                                # We need daily returns for this. The backend calculates metrics but doesn't return the full series.
                                # For now, we'll approximate or fetch data again here to plot.
                                # Ideally, backend should return equity curve.
                                # Let's fetch data here to plot.
                                data = fetch_data(tickers, start_date, end_date)
                                if not data.empty:
                                    data = data.dropna()
                                    rets = data.pct_change().dropna()
                                    
                                    # Optimized Equity Curve
                                    w_opt = pd.Series(weights)
                                    port_ret_opt = (rets * w_opt).sum(axis=1)
                                    cum_ret_opt = (1 + port_ret_opt).cumprod() * 10000 # Start $10k
                                    
                                    # Equal Weighted Equity Curve
                                    w_eq = pd.Series(weights_eq)
                                    port_ret_eq = (rets * w_eq).sum(axis=1)
                                    cum_ret_eq = (1 + port_ret_eq).cumprod() * 10000
                                    
                                    # Plot Growth
                                    fig_growth = go.Figure()
                                    fig_growth.add_trace(go.Scatter(x=cum_ret_opt.index, y=cum_ret_opt, mode='lines', name='Optimized', line=dict(color='blue')))
                                    fig_growth.add_trace(go.Scatter(x=cum_ret_eq.index, y=cum_ret_eq, mode='lines', name='Equal Weighted', line=dict(color='black')))
                                    fig_growth.update_layout(title="Portfolio Growth (Start $10,000)", xaxis_title="Date", yaxis_title="Value ($)")
                                    st.plotly_chart(fig_growth, use_container_width=True)
                                    
                                    # Annual Returns Bar Chart
                                    annual_ret_opt = port_ret_opt.resample('Y').apply(lambda x: (1 + x).prod() - 1)
                                    annual_ret_eq = port_ret_eq.resample('Y').apply(lambda x: (1 + x).prod() - 1)
                                    years = annual_ret_opt.index.year
                                    
                                    fig_bar = go.Figure()
                                    fig_bar.add_trace(go.Bar(x=years, y=annual_ret_opt, name='Optimized', marker_color='blue'))
                                    fig_bar.add_trace(go.Bar(x=years, y=annual_ret_eq, name='Equal Weighted', marker_color='black'))
                                    fig_bar.update_layout(title="Annual Returns", xaxis_title="Year", yaxis_title="Return", barmode='group')
                                    st.plotly_chart(fig_bar, use_container_width=True)
                                    
                                    # Dataset Download
                                    csv = data.to_csv().encode('utf-8')
                                    st.download_button(
                                        label="Download Dataset (CSV)",
                                        data=csv,
                                        file_name='portfolio_data.csv',
                                        mime='text/csv',
                                    )
                                    
                                    # HTML Report Download (Simple version)
                                    html_content = f"""
                                    <h1>Portfolio Optimization Report</h1>
                                    <h2>Strategy: {goal}</h2>
                                    <h3>Performance Metrics</h3>
                                    {metrics_df.to_html()}
                                    <br>
                                    <h3>Allocations</h3>
                                    <p>See interactive charts in app.</p>
                                    """
                                    st.download_button(
                                        label="Download Report (HTML)",
                                        data=html_content,
                                        file_name='report.html',
                                        mime='text/html',
                                    )
                                    
                                    # PDF Report Download
                                    try:
                                        resp_pdf = requests.post(f"{API_URL}/report", json=payload)
                                        if resp_pdf.status_code == 200:
                                            st.download_button(
                                                label="Download Report (PDF)",
                                                data=resp_pdf.content,
                                                file_name='report.pdf',
                                                mime='application/pdf',
                                            )
                                        else:
                                            st.error(f"Failed to generate PDF: {resp_pdf.text}")
                                    except Exception as e:
                                        st.error(f"Error fetching PDF: {e}")

                            else:
                                st.error("Failed to generate Equal Weighted benchmark.")
            else:
                st.error(f"Optimization failed: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend. Please ensure the FastAPI server is running.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Enter parameters and click 'Optimize Portfolio' to start. Ensure backend is running on port 8000.")


