#!/usr/bin/env python3
# AOTI LEVEL 37: THE HYBRID QUANT ENGINE (FIXED)
# Real-World Backtesting Framework.
# Fixes: float/scalar conversion errors with numpy arrays.

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
TICKER = "CVNA"       # Try "BTC-USD", "NVDA", "MSTR"
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
START_CASH = 10000.0

# GEOMETRY SETTINGS
MANIFOLD_WINDOW = 50   # Trend definition
GEOMETRY_WINDOW = 20   # Curvature definition

class QuantEngine:
    def __init__(self, ticker):
        print(f"--- INITIALIZING AOTI QUANT ENGINE ({ticker}) ---")
        self.ticker = ticker
        self.data = self.fetch_data()
        self.cash = START_CASH
        self.shares = 0
        self.equity_curve = []
        self.regime_history = [] 

    def fetch_data(self):
        print(f"Downloading Data for {self.ticker}...")
        # auto_adjust=True fixes the Future Warning and gives simpler data
        df = yf.download(self.ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        return df

    def fit_manifold(self, prices):
        x = np.arange(len(prices))
        
        # 1. Global Trend (Linear)
        slope, intercept = np.polyfit(x, prices, 1)
        
        # 2. Local Geometry (Quadratic)
        if len(prices) > GEOMETRY_WINDOW:
            recent = prices[-GEOMETRY_WINDOW:]
            rx = np.arange(GEOMETRY_WINDOW)
            a, b, c = np.polyfit(rx, recent, 2)
            curvature = a
        else:
            curvature = 0
            
        return slope, curvature

    def run_backtest(self):
        # Flatten the data to a 1D array to avoid shape errors
        prices = self.data['Close'].values.flatten()
        dates = self.data.index
        
        print(f"Running simulation on {len(prices)} days...")
        
        for t in range(MANIFOLD_WINDOW, len(prices)):
            # Force conversion to simple python float (Fixes formatting error)
            price = float(prices[t])
            
            # 1. ANALYZE GEOMETRY
            window = prices[t-MANIFOLD_WINDOW : t]
            slope, curvature = self.fit_manifold(window)
            
            # 2. DETERMINE REGIME
            if slope > 0.1:
                regime = "BULL"
                regime_val = 1
            elif slope < -0.1:
                regime = "BEAR"
                regime_val = -1
            else:
                regime = "CHOP"
                regime_val = 0
            
            self.regime_history.append(regime_val)
            
            # 3. HYBRID STRATEGY
            
            # --- SCENARIO A: BULL MARKET (Chaser) ---
            if regime == "BULL":
                if self.shares == 0:
                    self.buy(price, t, "CHASER ENTRY")
                    
            # --- SCENARIO B: BEAR/CHOP (AOTI Geometry) ---
            else:
                if regime == "BEAR" and self.shares > 0:
                    self.sell_all(price, t, "DOOMSDAY EXIT")
                
                # AOTI SNIPER ENTRY (Buy the Smile)
                if self.shares == 0 and curvature > 0.15:
                    self.buy(price, t, "AOTI SNIPER ENTRY")
                
                # AOTI SNIPER EXIT (Sell the Frown)
                if self.shares > 0 and curvature < -0.1:
                    self.sell_all(price, t, "AOTI PROFIT TAKE")

            # Track Performance
            val = self.cash + (self.shares * price)
            self.equity_curve.append(val)
            
        self.visualize(prices, dates)

    def buy(self, price, t, reason):
        if self.cash > price:
            count = int(self.cash / price)
            self.shares += count
            self.cash -= count * price

    def sell_all(self, price, t, reason):
        if self.shares > 0:
            self.cash += self.shares * price
            self.shares = 0

    def visualize(self, prices, dates):
        # Convert all benchmarks to scalars using float()
        start_price = float(prices[MANIFOLD_WINDOW])
        end_price = float(prices[-1])
        
        # Avoid divide by zero check
        if start_price == 0: start_price = 1.0
            
        bnh_return = ((end_price - start_price) / start_price) * 100
        
        my_end = float(self.equity_curve[-1])
        my_return = ((my_end - START_CASH) / START_CASH) * 100
        
        print("-" * 30)
        print(f"FINAL RESULTS ({self.ticker}):")
        print(f"Buy & Hold Return: {bnh_return:.2f}%")
        print(f"AOTI Hybrid Return: {my_return:.2f}%")
        print("-" * 30)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. Equity Curve
        ax1.plot(self.equity_curve, color='green', linewidth=2, label='AOTI Hybrid Strategy')
        
        # Create Buy/Hold line
        # Use numpy broadcasting but ensure dimensions match
        relevant_prices = prices[MANIFOLD_WINDOW:]
        bnh_line = (relevant_prices / start_price) * START_CASH
        
        ax1.plot(bnh_line, color='gray', linestyle='--', label='Buy & Hold (Benchmark)')
        
        ax1.set_title(f"Performance: AOTI vs Market ({self.ticker})")
        ax1.legend()
        ax1.grid(True)
        
        # 2. Regime Map
        ax2.plot(self.regime_history, color='orange', alpha=0.6, label='Market Regime')
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['Bear (AOTI Only)', 'Chop', 'Bull (Chaser Active)'])
        ax2.fill_between(range(len(self.regime_history)), 0, self.regime_history, color='orange', alpha=0.2)
        ax2.set_title("The Manifold State")
        ax2.grid(True)
        
        plt.show()

if __name__ == "__main__":
    try:
        engine = QuantEngine(TICKER)
        engine.run_backtest()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()