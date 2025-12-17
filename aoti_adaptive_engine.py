#!/usr/bin/env python3
# AOTI LEVEL 38: THE ADAPTIVE ENGINE (FIXED)
# Fixes the "Scale Bug" by using Log-Returns for Geometry.
# Works on ANY asset (Penny stocks, BTC, Forex) without changing thresholds.

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
TICKER = "NVDA"   # Now works for NVDA, BTC, ETH, etc.
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
START_CASH = 10000.0

# GEOMETRY SETTINGS
MANIFOLD_WINDOW = 50
GEOMETRY_WINDOW = 20

class AdaptiveEngine:
    def __init__(self, ticker):
        print(f"--- INITIALIZING ADAPTIVE ENGINE ({ticker}) ---")
        self.ticker = ticker
        self.data = self.fetch_data()
        self.cash = START_CASH
        self.shares = 0
        self.equity_curve = []
        self.regime_history = [] 

    def fetch_data(self):
        print(f"Downloading Data for {self.ticker}...")
        df = yf.download(self.ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        
        # *** THE FIX: LOGARITHMIC NORMALIZATION ***
        # We perform geometry on log(price) so $100 move and $10000 move are treated by %
        # We add a new column for internal calculations
        df['LogPrice'] = np.log(df['Close'])
        return df

    def fit_manifold(self, log_prices):
        x = np.arange(len(log_prices))
        
        # 1. Global Trend (Linear on Log Data = Exponential on Real Data)
        slope, intercept = np.polyfit(x, log_prices, 1)
        
        # 2. Local Geometry (Quadratic on Log Data)
        if len(log_prices) > GEOMETRY_WINDOW:
            recent = log_prices[-GEOMETRY_WINDOW:]
            rx = np.arange(GEOMETRY_WINDOW)
            a, b, c = np.polyfit(rx, recent, 2)
            curvature = a * 100 # Scale up for readability
        else:
            curvature = 0
            
        return slope, curvature

    def run_backtest(self):
        # We iterate through the RAW prices for trading, but use LOG prices for logic
        prices_raw = self.data['Close'].values.flatten()
        prices_log = self.data['LogPrice'].values.flatten()
        dates = self.data.index
        
        print(f"Running simulation on {len(prices_raw)} days...")
        
        for t in range(MANIFOLD_WINDOW, len(prices_raw)):
            current_price = float(prices_raw[t])
            
            # 1. ANALYZE GEOMETRY (Using LOG prices)
            window_log = prices_log[t-MANIFOLD_WINDOW : t]
            slope, curvature = self.fit_manifold(window_log)
            
            # 2. DETERMINE REGIME (Normalized Thresholds)
            # Slope of 0.001 in Log Space approx 0.1% daily growth (~30% APY)
            # This threshold now applies equally to BTC and SPY.
            
            BULL_THRESHOLD = 0.002  # Aggressive Bull (>0.2% daily trend)
            BEAR_THRESHOLD = -0.002 # Aggressive Bear (<-0.2% daily trend)
            
            if slope > BULL_THRESHOLD:
                regime = "BULL"
                regime_val = 1
            elif slope < BEAR_THRESHOLD:
                regime = "BEAR"
                regime_val = -1
            else:
                regime = "CHOP"
                regime_val = 0
            
            self.regime_history.append(regime_val)
            
            # 3. HYBRID STRATEGY
            
            # --- BULL MARKET: MOMENTUM ---
            # If the Manifold is pointing UP, stay fully invested.
            if regime == "BULL":
                if self.shares == 0:
                    self.buy(current_price, t)
            
            # --- CHOP/BEAR: GEOMETRY ---
            else:
                # If we detect a crash (Bear Regime), exit immediately to preserve capital
                if regime == "BEAR" and self.shares > 0:
                    self.sell_all(current_price, t)
                
                # If we are in "Chop" (Sideways), try to snipe the bottom
                # We buy if Curvature is strongly positive (Smile)
                if regime == "CHOP":
                     if self.shares == 0 and curvature > 0.05:
                         self.buy(current_price, t)
                     elif self.shares > 0 and curvature < -0.05:
                         self.sell_all(current_price, t)

            # Track Performance
            val = self.cash + (self.shares * current_price)
            self.equity_curve.append(val)
            
        self.visualize(prices_raw)

    def buy(self, price, t):
        if self.cash > price:
            # Slippage Model: We buy at Price * 1.001 (0.1% fee/slip)
            buy_price = price * 1.001
            count = int(self.cash / buy_price)
            if count > 0:
                self.shares += count
                self.cash -= count * buy_price

    def sell_all(self, price, t):
        if self.shares > 0:
            # Slippage Model: We sell at Price * 0.999
            sell_price = price * 0.999
            self.cash += self.shares * sell_price
            self.shares = 0

    def visualize(self, prices):
        start_price = float(prices[MANIFOLD_WINDOW])
        end_price = float(prices[-1])
        if start_price == 0: start_price = 1.0
            
        bnh_return = ((end_price - start_price) / start_price) * 100
        my_end = float(self.equity_curve[-1])
        my_return = ((my_end - START_CASH) / START_CASH) * 100
        
        print("-" * 30)
        print(f"FINAL RESULTS ({self.ticker}):")
        print(f"Buy & Hold Return: {bnh_return:.2f}%")
        print(f"AOTI Adaptive Return: {my_return:.2f}%")
        print("-" * 30)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot Logic
        ax1.semilogy(self.equity_curve, color='green', linewidth=2, label='AOTI Adaptive (Log Scale)')
        
        # Benchmark Line
        relevant_prices = prices[MANIFOLD_WINDOW:]
        bnh_line = (relevant_prices / start_price) * START_CASH
        ax1.semilogy(bnh_line, color='gray', linestyle='--', label='Buy & Hold (Benchmark)')
        
        ax1.set_title(f"AOTI vs Market ({self.ticker}) - Logarithmic Scale")
        ax1.legend()
        ax1.grid(True, which="both", alpha=0.3)
        
        # Regime Map
        ax2.plot(self.regime_history, color='orange', alpha=0.6, label='Market Regime')
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['Bear', 'Chop', 'Bull'])
        ax2.fill_between(range(len(self.regime_history)), 0, self.regime_history, color='orange', alpha=0.2)
        ax2.set_title("The Adaptive Manifold State")
        ax2.grid(True)
        
        plt.show()

if __name__ == "__main__":
    try:
        engine = AdaptiveEngine(TICKER)
        engine.run_backtest()
    except Exception as e:
        import traceback
        traceback.print_exc()