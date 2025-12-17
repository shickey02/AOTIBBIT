#!/usr/bin/env python3
# AOTI LEVEL 34: THE QUANT LAB
# A modular trading environment for testing AI strategies.
# Features:
# 1. Buy/Sell Visualization (Arrows on chart)
# 2. Polynomial Manifolding (Detecting Curves vs Lines)
# 3. Regime Switching Market (Trends + Chop)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- CONFIGURATION ---
SIM_LENGTH = 600
START_CASH = 10000.0
COMMISSION = 2.0 

class MarketEnv:
    def __init__(self):
        # Generate a complex market: Trend + Waves + Noise
        t = np.linspace(0, 60, SIM_LENGTH)
        
        # 1. The Underlying Trend (The Manifold)
        # A slow rise, then a crash, then a recovery
        trend = 100 + (t * 1.5) 
        trend[300:] -= (t[300:] - 30) * 3 # The Crash
        trend[450:] += (t[450:] - 45) * 5 # The V-Shape Recovery
        
        # 2. The Cycle
        cycle = 20 * np.sin(t)
        
        # 3. The Noise
        noise = np.random.randn(SIM_LENGTH) * 2.0
        
        self.price = trend + cycle + noise
        
    def get_price(self, t):
        if t >= SIM_LENGTH: return None
        return self.price[t]

class Trader:
    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.cash = START_CASH
        self.shares = 0
        self.history = [START_CASH]
        # For visualization
        self.buys_x = []
        self.buys_y = []
        self.sells_x = []
        self.sells_y = []

    def get_value(self, price):
        return self.cash + (self.shares * price)

    def buy(self, t, price):
        # Bet size: 10% of cash
        bet = self.cash * 0.1
        if bet < price: return # Too poor
        
        count = int(bet / price)
        cost = (count * price) + COMMISSION
        self.cash -= cost
        self.shares += count
        
        # Record for plot
        self.buys_x.append(t)
        self.buys_y.append(price)

    def sell(self, t, price):
        if self.shares == 0: return
        
        # Sell half position
        count = max(1, int(self.shares * 0.5))
        gain = (count * price) - COMMISSION
        self.cash += gain
        self.shares -= count
        
        # Record for plot
        self.sells_x.append(t)
        self.sells_y.append(price)

    def strategy(self, t, prices):
        pass # Override me

# --- STRATEGY 1: THE CHASER (Momentum) ---
class MomentumBot(Trader):
    def strategy(self, t, prices):
        if t < 10: return
        # Simple Logic: Current Price vs Price 5 days ago
        delta = prices[t] - prices[t-5]
        
        if delta > 2.0: self.buy(t, prices[t])
        elif delta < -2.0: self.sell(t, prices[t])

# --- STRATEGY 2: THE POLYNOMIAL GEOMETER (AOTI V2) ---
class PolyBot(Trader):
    """
    Instead of Linear Regression (Straight Lines),
    this bot fits a PARABOLA (Curve) to recent prices.
    It looks for the 'Vertex' (The bottom of the dip).
    """
    def strategy(self, t, prices):
        window = 25
        if t < window: return
        
        # 1. Extract Geometry
        recent = prices[t-window : t]
        x = np.arange(window)
        
        # 2. Fit Polynomial (Degree 2: y = ax^2 + bx + c)
        # a describes curvature. 
        # a > 0: Happy Face (Dip)
        # a < 0: Sad Face (Peak)
        coeffs = np.polyfit(x, recent, 2)
        a, b, c = coeffs
        
        # 3. Calculate Vertex (Turning Point)
        # Vertex x = -b / 2a
        # We want to know if the turn is happening NOW (near end of window)
        vertex_x = -b / (2 * a) if a != 0 else -999
        
        is_turning = (window - 5) < vertex_x < (window + 5)
        
        # 4. DECISION
        # If curvature is Positive (U-shape) and we are near the bottom -> BUY
        if a > 0.05 and is_turning:
            self.buy(t, prices[t])
            
        # If curvature is Negative (n-shape) and we are near the top -> SELL
        elif a < -0.05 and is_turning:
            self.sell(t, prices[t])

# --- THE LAB ---
def run_lab():
    market = MarketEnv()
    bots = [
        MomentumBot("Chaser (Blue)", "blue"),
        PolyBot("AOTI Poly (Green)", "green")
    ]
    
    price_history = []
    
    # Run Simulation (Instant, no animation, focusing on final analysis)
    print("--- RUNNING QUANT SIMULATION ---")
    for t in range(SIM_LENGTH):
        price = market.get_price(t)
        price_history.append(price)
        
        for bot in bots:
            bot.strategy(t, price_history)
            bot.history.append(bot.get_value(price))

    # --- ANALYSIS PLOT ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # CHART 1: Price Action & Trades
    ax1.plot(price_history, color='black', alpha=0.5, label='Market')
    
    # Plot Trades
    for bot in bots:
        # Buy Markers (Up Triangle)
        ax1.scatter(bot.buys_x, bot.buys_y, color=bot.color, marker='^', s=50, label=f'{bot.name} Buys', alpha=0.8)
        # Sell Markers (Down Triangle)
        ax1.scatter(bot.sells_x, bot.sells_y, color=bot.color, marker='v', s=50, label=f'{bot.name} Sells', alpha=0.8)

    ax1.set_title("Trade Execution Map (Did they buy the dip?)")
    ax1.legend()
    ax1.grid(True)
    
    # CHART 2: Performance
    for bot in bots:
        ax2.plot(bot.history, color=bot.color, linewidth=2, label=f"{bot.name}: ${int(bot.history[-1])}")
    
    ax2.set_title("Portfolio Growth")
    ax2.legend()
    ax2.grid(True)
    
    plt.show()

if __name__ == "__main__":
    run_lab()