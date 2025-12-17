#!/usr/bin/env python3
# AOTI LEVEL 39: THE GEOMETRIC OPTIMIZER
# "Machine Learning" approach to trading.
# It runs the simulation 50+ times to find the perfect 'Window Size'
# that would have beaten Buy & Hold for a specific asset.

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
TICKER = "BTC-USD"  # Change to NVDA, TSLA, etc.
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
START_CASH = 10000.0

class Optimizer:
    def __init__(self, ticker):
        print(f"--- TUNING AOTI ENGINE FOR {ticker} ---")
        self.ticker = ticker
        self.data = self.fetch_data()
        
    def fetch_data(self):
        print("Downloading Data...")
        df = yf.download(self.ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        # Use Log Price for Geometry
        df['LogPrice'] = np.log(df['Close'])
        return df

    def run_simulation(self, window_size):
        """
        Runs the standard AOTI Logic with a specific 'Manifold Window'.
        Returns: Final Portfolio Value
        """
        prices_raw = self.data['Close'].values.flatten()
        prices_log = self.data['LogPrice'].values.flatten()
        
        cash = START_CASH
        shares = 0
        
        # We need enough data to start
        if len(prices_raw) < window_size + 1: return START_CASH

        for t in range(window_size, len(prices_raw)):
            current_price = float(prices_raw[t])
            
            # GEOMETRY
            # We fit a line to the log-prices
            segment = prices_log[t-window_size : t]
            x = np.arange(window_size)
            slope, _ = np.polyfit(x, segment, 1)
            
            # STRATEGY (Simplified for Speed)
            # Bull Regime (> 0.2% daily trend) -> BUY/HOLD
            # Bear Regime (< -0.2% daily trend) -> SELL/CASH
            
            THRESHOLD = 0.002
            
            if slope > THRESHOLD:
                # BUY signal
                if shares == 0:
                    buy_price = current_price * 1.001 # Slippage
                    if cash > buy_price:
                        shares = int(cash / buy_price)
                        cash -= shares * buy_price
            
            elif slope < -THRESHOLD:
                # SELL signal
                if shares > 0:
                    sell_price = current_price * 0.999 # Slippage
                    cash += shares * sell_price
                    shares = 0
                    
        final_value = cash + (shares * prices_raw[-1])
        return final_value

    def optimize(self):
        results = []
        windows = range(5, 100, 5) # Test 5, 10, 15 ... 95
        
        prices = self.data['Close'].values.flatten()
        bnh_return = (prices[-1] - prices[0]) / prices[0] * 100
        print(f"Benchmark (Buy & Hold) Return: {bnh_return:.2f}%")
        print("Scanning Geometric Frequencies...")
        
        best_win = 0
        best_return = -9999
        
        for w in windows:
            final_val = self.run_simulation(w)
            ret = (final_val - START_CASH) / START_CASH * 100
            results.append(ret)
            
            if ret > best_return:
                best_return = ret
                best_win = w
            
            # Optional: Print progress
            # print(f"Window {w}: {ret:.2f}%")

        print("-" * 30)
        print(f"OPTIMIZATION COMPLETE")
        print(f"Best Window Size: {best_win} days")
        print(f"Best AOTI Return: {best_return:.2f}%")
        print("-" * 30)
        
        if best_return > bnh_return:
            print(">> SUCCESS: Geometry beat the Market.")
        else:
            print(">> REALITY CHECK: Buy & Hold won. (Asset is too vertical).")

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.plot(windows, results, marker='o', label='AOTI Return')
        plt.axhline(bnh_return, color='gray', linestyle='--', label='Buy & Hold Benchmark')
        
        plt.title(f"Geometry Optimization: {self.ticker}")
        plt.xlabel("Manifold Window Size (Days)")
        plt.ylabel("Total Return %")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    try:
        opt = Optimizer(TICKER)
        opt.optimize()
    except Exception as e:
        import traceback
        traceback.print_exc()