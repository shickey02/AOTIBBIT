#!/usr/bin/env python3
# AOTI LEVEL 40: PRODUCTION TRADER (FIXED)
# The Final Artifact.
# Usage: Run this every morning to get your daily trading signal.
# Configuration: Update 'OPTIMIZED_WINDOW' based on your Optimizer results.

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- YOUR GOLDEN SETTINGS ---
TICKER = "BTC-USD"
OPTIMIZED_WINDOW = 45  # The number you found in Level 39
THRESHOLD = 0.002      # The slope threshold (0.2% daily trend)

class ProductionDroid:
    def __init__(self):
        print(f"--- AOTI PRODUCTION DROID ({TICKER}) ---")
        print(f"Optimized Window: {OPTIMIZED_WINDOW} Days")
        self.data = self.fetch_live_data()

    def fetch_live_data(self):
        print(">> Ping Yahoo Finance...")
        # Get enough data to calculate the window + buffer
        df = yf.download(TICKER, period="1y", interval="1d", progress=False, auto_adjust=True)
        # Handle cases where columns might be MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(TICKER, axis=1, level=0) if TICKER in df.columns else df
            
        df['LogPrice'] = np.log(df['Close'])
        return df

    def analyze_geometry(self):
        # We only care about the curve RIGHT NOW (The tip of the spear)
        prices_log = self.data['LogPrice'].values
        # Explicit float conversion to fix formatting bug
        current_price = self.data['Close'].values[-1].item()
        dates = self.data.index
        
        if len(prices_log) < OPTIMIZED_WINDOW:
            print("Error: Not enough data history.")
            return

        # Extract the specific window for today
        window_log = prices_log[-OPTIMIZED_WINDOW:]
        
        # FIT THE MANIFOLD
        x = np.arange(OPTIMIZED_WINDOW)
        
        # 1. Slope (Trend)
        slope, _ = np.polyfit(x, window_log, 1)
        slope = float(slope) # Force scalar
        
        # 2. Curvature (Acceleration)
        # We use a smaller sub-window (e.g., half the trend window) for curvature
        curve_window = int(OPTIMIZED_WINDOW / 2)
        curve_log = prices_log[-curve_window:]
        cx = np.arange(curve_window)
        curvature, _, _ = np.polyfit(cx, curve_log, 2)
        
        # Scale curvature for readability
        curvature = float(curvature * 100) # Force scalar

        self.report(dates[-1], current_price, slope, curvature)

    def report(self, date, price, slope, curvature):
        print("\n" + "="*40)
        print(f"DATE: {date.strftime('%Y-%m-%d')}")
        print(f"ASSET PRICE: ${price:,.2f}")
        print("-" * 40)
        print(f"GEOMETRIC SLOPE:     {slope:.6f}")
        print(f"GEOMETRIC CURVATURE: {curvature:.6f}")
        print("-" * 40)
        
        # --- THE DECISION MATRIX ---
        signal = "HOLD / CASH"
        reason = "Neutral"
        
        # LOGIC: Matches your Backtest
        if slope > THRESHOLD:
            signal = "BUY / LONG"
            reason = "Bull Market Regime (Strong Trend)"
            
        elif slope < -THRESHOLD:
            signal = "SELL / CASH"
            reason = "Bear Market Regime (Crash Detected)"
            
        else:
            # Chop Mode - Check Geometry
            if curvature > 0.05:
                signal = "SNIPE ENTRY (BUY)"
                reason = "Chop Market + Positive Curvature (Smile)"
            elif curvature < -0.05:
                signal = "EXIT POSITION (SELL)"
                reason = "Chop Market + Negative Curvature (Frown)"
            else:
                signal = "WAIT"
                reason = "No distinct geometry. Stay Out."

        print(f"SIGNAL: {signal}")
        print(f"REASON: {reason}")
        print("="*40 + "\n")
        
        self.visualize_context()

    def visualize_context(self):
        # Show the user the chart so they trust the math
        prices = self.data['Close'][-100:] # Last 100 days
        
        plt.figure(figsize=(10, 5))
        plt.plot(prices.index, prices.values, color='black', label='Price')
        
        # Highlight the window used for calculation
        window_prices = prices[-OPTIMIZED_WINDOW:]
        plt.plot(window_prices.index, window_prices.values, color='orange', linewidth=2, label='AOTI Scan Window')
        
        plt.title(f"AOTI Visual Context: {TICKER}")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    bot = ProductionDroid()
    bot.analyze_geometry()