#!/usr/bin/env python3
# AOTI LEVEL 36: THE REAL WORLD BRIDGE
# Loads real market data (CSV) and applies AOTI Geometric Reasoning.
# Usage: Download a CSV from Yahoo Finance (e.g., BTC-USD.csv) and run this.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def load_data(filepath):
    """
    Expects a standard CSV with a 'Close' column.
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        print("Please download a CSV (e.g. NVDA.csv) from Yahoo Finance.")
        return None
    
    df = pd.read_csv(filepath)
    # Ensure we have data
    return df['Close'].values

class GeometricReasoning:
    def __init__(self, prices):
        self.prices = prices
        self.decisions = [] # (Index, Type, Price)
        
    def analyze(self):
        print("--- AOTI GEOMETRIC SCANNING ---")
        window = 20
        
        for t in range(window, len(self.prices)):
            # 1. Extract Local Geometry
            segment = self.prices[t-window:t]
            x = np.arange(window)
            
            # 2. Fit Parabola (Curvature)
            # y = ax^2 + bx + c
            a, b, c = np.polyfit(x, segment, 2)
            
            # 3. Fit Line (Trend)
            slope, intercept = np.polyfit(x, segment, 1)
            
            current_price = self.prices[t]
            
            # LOGIC:
            # We want Positive Curvature (Smile) AND Price below the Linear Manifold
            # Linear Manifold Value at t = slope*(window) + intercept
            manifold_val = (slope * window) + intercept
            gap = current_price - manifold_val
            
            # Signal Thresholds
            if a > 0.05 and gap < -1.0:
                self.decisions.append((t, 'BUY', current_price))
            elif a < -0.05 and gap > 1.0:
                self.decisions.append((t, 'SELL', current_price))

    def visualize(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.prices, color='black', alpha=0.6, label='Asset Price')
        
        buys_x, buys_y = [], []
        sells_x, sells_y = [], []
        
        for t, action, price in self.decisions:
            if action == 'BUY':
                buys_x.append(t); buys_y.append(price)
            else:
                sells_x.append(t); sells_y.append(price)
                
        plt.scatter(buys_x, buys_y, color='green', marker='^', s=80, label='AOTI Buy Signal', zorder=5)
        plt.scatter(sells_x, sells_y, color='red', marker='v', s=80, label='AOTI Sell Signal', zorder=5)
        
        plt.title(f"AOTI Analysis on Real Data ({len(self.decisions)} Signals)")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Create a dummy file for demonstration if none exists
    if not os.path.exists("market_data.csv"):
        print(">> No Data Found. Generating sample 'market_data.csv'...")
        # Create a fake realistic pattern
        t = np.linspace(0, 100, 300)
        p = 100 + 2*t + 20*np.sin(t) + np.random.randn(300)*2
        df = pd.DataFrame({'Close': p})
        df.to_csv("market_data.csv")
        target_file = "market_data.csv"
    else:
        # You can change this to your downloaded file
        target_file = "market_data.csv"
        
    data = load_data(target_file)
    if data is not None:
        ai = GeometricReasoning(data)
        ai.analyze()
        ai.visualize()