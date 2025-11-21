import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data
def fetch_data(tickers, start_date, end_date):
    """
    Fetches historical data for the given tickers using yfinance.
    
    Args:
        tickers (list): List of ticker symbols.
        start_date (datetime): Start date.
        end_date (datetime): End date.
        
    Returns:
        pd.DataFrame: Adjusted Close prices.
    """
    if not tickers:
        return pd.DataFrame()
    
    # Remove duplicates and whitespace
    tickers = list(set([t.strip().upper() for t in tickers]))
    
    try:
        # Download data
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # Check if data is empty
        if data.empty:
            st.error("No data found for the selected tickers. Please check the ticker symbols and date range.")
            return pd.DataFrame()

        # Handle 'Adj Close' vs 'Close'
        if 'Adj Close' in data:
            prices = data['Adj Close']
        elif 'Close' in data:
            st.warning("'Adj Close' not found, using 'Close' prices instead.")
            prices = data['Close']
        else:
            # If neither exists, it might be a single level DF if only one ticker and flattened?
            # Or maybe it's just the prices directly if auto_adjust=True?
            # Let's check if the columns are the tickers themselves (implying it's already the price series)
            # But yfinance usually returns OHLCV.
            st.error(f"Could not find price data in response. Columns: {data.columns}")
            return pd.DataFrame()
            
        # If prices is a Series (single ticker), convert to DataFrame
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        
        # Drop columns with all NaNs (failed downloads)
        prices = prices.dropna(axis=1, how='all')
        
        # Drop rows with any NaNs to ensure common history for all assets
        # This is important for optimization stability
        prices = prices.dropna(axis=0, how='any')
        
        if prices.empty:
            st.error("No common history found for the selected tickers. Try selecting a shorter time period or different tickers.")
            return pd.DataFrame()
            
        return prices
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()



