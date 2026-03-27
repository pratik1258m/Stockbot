<p align="center">
  <img src="https://img.shields.io/badge/Status-Final%20Year%20Project%20Ready-success" alt="Status Badge"/>
  <img src="https://img.shields.io/badge/Version-3.0%20(Tailwind)-blue" alt="Version Badge"/>
  <img src="https://img.shields.io/badge/Python-3.8+-yellow" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/-Flask-000000?style=flat&logo=flask&logoColor=white" alt="Flask Badge"/>
  <img src="https://img.shields.io/badge/-TailwindCSS-38B2AC?style=flat&logo=tailwind-css&logoColor=white" alt="Tailwind Badge"/>
  <img src="https://img.shields.io/badge/-Three.js-000000?style=flat&logo=three.js&logoColor=white" alt="Three.js Badge"/>
</p>

# StockBot AI Vision Pro (v3.0)

This is a simple, interactive command-line chatbot built with Python and the `yfinance` module. It allows users to fetch real-time stock data by entering the ticker symbol (e.g., AAPL, TSLA, MSFT). The bot retrieves and displays the current price, day’s high, and day’s low for the selected stock.

## Ultimate Features

*   **Interactive 3D Interface**: Features an immersive WebGL particle background powered by **Three.js** that responds to user mouse movement.
*   **Holographic Charting Engine**: Integrates **Chart.js** mapped with dynamic bezier curves, gradient fills, and customized tooltips.
*   **Tailwind CSS Overhaul**: Fully responsive, "Apple-tier" frosted glass dashboard (`backdrop-blur-xl`) with staggered animations.
*   **Advanced Technical Indicators Engine**:
    *   Simple Moving Averages (SMA 20 & 50 Crossovers)
    *   Relative Strength Index (RSI 14)
    *   MACD (Moving Average Convergence Divergence Histogram)
    *   Bollinger Bands (Volatility Boundaries)
*   **Neural Verdict System**: A proprietary AI scoring engine that automatically grades data points to generate actionable `STRONG BUY` / `HOLD` / `STRONG SELL` signals alongside risk assessments.
*   **Global Market Support**: Handles both NSE/BSE (e.g., "RELIANCE.NS", "TATASTEEL.NS") and international markets (e.g., "TSLA", "AAPL").

## Architecture Stack

The project has been scaled to a full MVT architectural pattern for maximum maintainability:

1.  **Frontend (Jinja2 + Tailwind + Three.js)**: 
    *   `layout.html`, `index.html`, `dashboard.html`. Handles WebGL rendering, CSS grids, and chart population.
2.  **Backend (Flask Server)**:
    *   `app.py`: The robust routing and controller layer running asynchronously via Python.
3.  **Data Core (`data_engine.py`)**: 
    *   Extensive integration with the `yfinance` API for instant market history fetching and caching.
4.  **Math Engine (`analysis_engine.py`)**: 
    *   Calculates all statistical formulas via NumPy & Pandas.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Internet connection (required for CDN styling and real-time market polling)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/pratik1258m/Stockbot.git
    cd Stockbot
    ```

2.  Install required backend dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Execute the Application Server:
    ```bash
    python3 app.py
    ```
    *Access the breathtaking dashboard at `http://127.0.0.1:5001`.*

## Evaluation Methodology

The internal "Verdict AI" formulates signals using a weighted scoring matrix:
*   **Trend Factor**: Evaluates the relation of the Close price natively against the SMA-50 baseline.
*   **Momentum Factor**: Assigns gravity based on RSI extremities (Oversold < 30 = Bullish; Overbought > 70 = Bearish).
*   **Convergence Factor**: Interprets positive/negative histogram divergence emitted by the MACD.
*   **Volatility Factor**: Monitors Bollinger Band expansion tests and standard deviation breakouts.

These raw scores are mapped to a human-readable professional diagnostic scale.

## Academic Context
Designed explicitly to fulfill requirements for a Computer Science Final Year thesis/project. Code structure is strictly self-documented, highly modular, and production-ready.
