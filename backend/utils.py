import plotly.graph_objects as go
import pandas as pd

def plot_efficient_frontier(frontier_volatility, frontier_returns, portfolio_vol, portfolio_ret):
    """
    Plots the Efficient Frontier and the optimized portfolio.
    """
    fig = go.Figure()
    
    # Frontier
    fig.add_trace(go.Scatter(
        x=frontier_volatility, 
        y=frontier_returns, 
        mode='lines', 
        name='Efficient Frontier',
        line=dict(color='blue', width=2)
    ))
    
    # Optimized Portfolio
    fig.add_trace(go.Scatter(
        x=[portfolio_vol], 
        y=[portfolio_ret], 
        mode='markers', 
        name='Optimized Portfolio',
        marker=dict(color='red', size=12, symbol='star')
    ))
    
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility (Std. Dev)',
        yaxis_title='Expected Return',
        showlegend=True,
        width=800,
        height=500
    )
    
    return fig

def plot_weights(weights):
    """
    Plots the portfolio weights as a pie chart.
    """
    labels = list(weights.keys())
    values = list(weights.values())
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title_text="Portfolio Allocation")
    return fig

