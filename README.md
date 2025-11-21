# Portfolio Optimization Engine

A comprehensive portfolio optimization tool with a FastAPI backend and Streamlit frontend.

## Features
- **Strategies**: Mean-Variance, CVaR, Risk Parity, Kelly, Sortino, Omega, Max Drawdown, Tracking Error, Information Ratio, Target Volatility, Equal Weighted.
- **Reporting**: Generate detailed PDF reports comparing optimized portfolios against an equal-weighted benchmark.
- **Visualization**: Interactive charts for allocation, efficient frontier, and performance.

## Running Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```
   This starts the backend on port 8000 and the frontend on port 8501.

## Running with Docker

1. **Build Image**:
   ```bash
   docker build -t portfolio-optimizer .
   ```

2. **Run Container**:
   ```bash
   docker run -p 8000:8000 -p 8501:8501 portfolio-optimizer
   ```

## Deployment (Streamlit Cloud)

1. Push this repository to GitHub.
2. Connect your GitHub account to Streamlit Cloud.
3. Deploy the `frontend/app.py` file.
4. **Note**: Streamlit Cloud only hosts the frontend. You will need to deploy the backend separately (e.g., on Render, Railway, or a VPS) and update the `API_URL` in `frontend/app.py` to point to your deployed backend.
   - Alternatively, for a simple demo, you can refactor the app to run monolithically (importing backend logic directly), but the current architecture separates concerns for scalability.

## API Documentation

Visit `http://localhost:8000/docs` for the interactive Swagger UI.
