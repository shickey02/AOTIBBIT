#!/usr/bin/env python3
# AOTI LEVEL 33: THE MARKET ARENA
# A Proof of Concept for AOTI in Finance.
# Simulates a stock market and pits 3 AI Traders against each other.
# Demonstrates "Geometric Value Finding" vs. Momentum/Random.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- CONFIGURATION ---
SIM_LENGTH = 500
START_CASH = 10000.0
TRANSACTION_COST = 5.0 # Slippage/Fees

class MarketMaker:
    """
    Generates the 'Reality'.
    The Market consists of:
    1. The Manifold (True Value): A smooth, predictable curve (e.g., Earnings).
    2. The Noise (Sentiment): Chaotic fluctuations around the value.
    """
    def __init__(self):
        self.time = 0
        # Create a "True Value" trend (Manifold)
        # It's a sine wave + linear growth
        t = np.linspace(0, 50, SIM_LENGTH)
        self.manifold = 100 + (t * 2) + (20 * np.sin(t))
        
        # Create "Market Price" (True Value + Chaos)
        # Noise is a random walk to simulate market irrationality
        noise = np.random.randn(SIM_LENGTH) * 2.0
        self.price_history = self.manifold + np.cumsum(noise)
        
    def get_data(self, t):
        if t >= SIM_LENGTH: return None, None
        return self.price_history[t], self.manifold[t]

class TraderBase:
    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.cash = START_CASH
        self.shares = 0
        self.history = [START_CASH] # Track portfolio value
        
    def get_portfolio_value(self, current_price):
        return self.cash + (self.shares * current_price)
    
    def buy(self, price, amount=10):
        cost = (price * amount) + TRANSACTION_COST
        if self.cash >= cost:
            self.cash -= cost
            self.shares += amount
            
    def sell(self, price, amount=10):
        if self.shares >= amount:
            gain = (price * amount) - TRANSACTION_COST
            self.cash += gain
            self.shares -= amount
            
    def think(self, price, t, history):
        pass # Overwritten by subclasses

# --- TRADER 1: THE GAMBLER ---
class RandomTrader(TraderBase):
    def think(self, price, t, history):
        action = np.random.choice(['buy', 'sell', 'hold'])
        if action == 'buy': self.buy(price)
        elif action == 'sell': self.sell(price)

# --- TRADER 2: THE CHASER (Momentum) ---
class MomentumTrader(TraderBase):
    def think(self, price, t, history):
        if t < 5: return
        # Simple Derivative: Is it going up?
        momentum = price - history[t-1]
        
        if momentum > 0.5: # It's mooning! FOMO Buy!
            self.buy(price)
        elif momentum < -0.5: # It's crashing! Panic Sell!
            self.sell(price)

# --- TRADER 3: THE GEOMETER (AOTI) ---
class AOTITrader(TraderBase):
    """
    Uses Geometric Manifold Projection.
    It calculates the 'Geometric Gap' between Price and Estimated Value.
    """
    def __init__(self, name, color):
        super().__init__(name, color)
        self.window_size = 20 # How far back to look to build the manifold
        
    def think(self, price, t, history):
        if t < self.window_size: return
        
        # 1. CONSTRUCT MANIFOLD (Find True Value)
        # We use Linear Regression on the recent window to find the "Center Line"
        # This is the AOTI "Crystal" logic applied to data.
        recent_prices = history[t-self.window_size : t]
        x = np.arange(self.window_size)
        
        # Fit a line (y = mx + b)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, recent_prices, rcond=None)[0]
        
        # Project the Manifold to current time (t)
        estimated_value = (m * self.window_size) + c
        
        # 2. MEASURE THE GAP
        # Geometric Gap = Price - True Value
        gap = price - estimated_value
        
        # 3. REASONING
        # If Gap is positive, Price is "floating above reality" (Overvalued) -> SELL
        # If Gap is negative, Price is "below reality" (Undervalued) -> BUY
        
        threshold = 2.0 # Volatility tolerance
        
        if gap < -threshold:
            # "The asset is geometrically undervalued."
            self.buy(price, amount=20) # Aggressive Buy
            
        elif gap > threshold:
            # "The asset is detached from reality."
            self.sell(price, amount=20) # Aggressive Sell

# --- THE ARENA ---
def run_arena():
    market = MarketMaker()
    
    traders = [
        RandomTrader("Gambler", "gray"),
        MomentumTrader("Chaser", "blue"),
        AOTITrader("AOTI (Geometer)", "green")
    ]
    
    # Visualization Setup
    fig, (ax_price, ax_pnl) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    print("--- AOTI MARKET ARENA ---")
    print("Simulating 500 trading days.")
    print("Green Line = The AOTI Geometer (Value Finding)")
    print("Blue Line  = The Momentum Chaser")
    print("Gray Line  = The Random Gambler")
    
    # Store history for plotting
    price_hist = []
    manifold_hist = []
    
    def update(frame):
        # 1. Get Market Data
        price, true_value = market.get_data(frame)
        if price is None: return
        
        price_hist.append(price)
        manifold_hist.append(true_value)
        
        # 2. Traders Think & Act
        for trader in traders:
            trader.think(price, frame, price_hist)
            val = trader.get_portfolio_value(price)
            trader.history.append(val)
        
        # 3. Visualization
        ax_price.clear(); ax_pnl.clear()
        
        # Top Chart: Stock Price
        ax_price.plot(price_hist, color='black', alpha=0.6, label='Market Price')
        ax_price.plot(manifold_hist, color='orange', linestyle='--', alpha=0.8, label='True Value Manifold (Hidden)')
        ax_price.set_title(f"Market Day {frame}")
        ax_price.legend(loc='upper left')
        ax_price.grid(True)
        
        # Bottom Chart: Trader Performance
        for trader in traders:
            ax_pnl.plot(trader.history, color=trader.color, label=f"{trader.name}: ${int(trader.history[-1])}")
            
        ax_pnl.set_title("Trader Performance (Portfolio Value)")
        ax_pnl.legend(loc='upper left')
        ax_pnl.grid(True)
        
    ani = animation.FuncAnimation(fig, update, frames=SIM_LENGTH, interval=1, repeat=False)
    plt.show()

if __name__ == "__main__":
    run_arena()