# My Trading Strategy Implementation

This repository contains an implementation of a quantitative trading strategy.

## Important Disclaimer & Attribution

**I did not invent this strategy.** The core concepts, methodology, and logic implemented here are entirely derived from existing academic research.

Specifically, this project is an implementation of the strategy detailed in the following article:

* **Beat the Market An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)**
* **Carlo Zarattini, Andrew Aziz, Andrea Barbon
  

My contribution to this project is limited to the **software implementation of the strategy**, focusing particularly on the section *[ Further Investigations]* as described in the aforementioned article.

All intellectual property and original ideas belong to the author(s) of the cited research. This implementation serves as a practical application and study of their work.
## Project Overview

This project backtests the described trading strategy **using intraday data at 30-minute intervals**. 

## Setup and Usage

[Provide instructions on how to set up the project and run your code, e.g.:]

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-link]
    cd [your-repo-name]
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll need to create a `requirements.txt` file listing libraries like `pandas`, `numpy`, `matplotlib`, `seaborn`)*
3.  **Run the backtest:**
    ```bash
    python your_main_script_name.py
    ```

## Results & Analysis

The script generates performance metrics (e.g., Total Return, Sharpe Ratio, Max Drawdown, Alpha, Beta) and an equity curve plot.

---
