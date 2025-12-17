#!/usr/bin/env python3
# AOTI LEVEL 35: THE CRASH TEST
# Pits Momentum vs. Geometry in a Bear Market.
# Introduces the "Hybrid" strategy (Momentum Trend + Geometric Entry).

import numpy as np
import matplotlib.pyplot as plt

# --- MARKET CONFIGURATION ---
SIM_LENGTH = 800
START_CASH = 10000.0
COMMISSION = 2.0 

class CrashMarket:
    def __init__(self):
        t = np.linspace(0, 80, SIM_LENGTH)
        
        # 1. THE BULL RUN (Days 0-500)
        trend = 100 + (t * 2.5)
        
        # 2. THE CRASH (Days 500-800)
        # A massive 50% drop
        crash_start = 500
        trend[crash_start:] -= (np.arange(SIM_LENGTH - crash_start) ** 1.5) * 0.5
        
        # 3. Volatility
        noise = np.random.randn(SIM_LENGTH) * 3.0
        cycle = 15 * np.sin(t)
        
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

    def get_value(self, price):
        return self.cash + (self.shares * price)

    def buy(self, price, percentage=0.1):
        bet = self.cash * percentage
        if bet < price: return
        count = int(bet / price)
        self.cash -= (count * price) + COMMISSION
        self.shares += count

    def sell(self, price, percentage=0.5):
        if self.shares == 0: return
        count = max(1, int(self.shares * percentage))
        self.cash += (count * price) - COMMISSION
        self.shares -= count
        
    def sell_all(self, price):
        if self.shares == 0: return
        self.cash += (self.shares * price) - COMMISSION
        self.shares = 0

    def strategy(self, t, prices): pass

# --- STRATEGY 1: PURE MOMENTUM (The Chaser) ---
class MomentumBot(Trader):
    def strategy(self, t, prices):
        if t < 10: return
        # If price is up over last 5 days, BUY.
        if prices[t] > prices[t-5]:
            self.buy(prices[t], 0.2) # Aggressive
        # If price drops, SELL (Stop Loss)
        elif prices[t] < prices[t-5]:
            self.sell(prices[t], 0.2)

# --- STRATEGY 2: PURE GEOMETRY (The Sniper) ---
class PolyBot(Trader):
    def strategy(self, t, prices):
        window = 20
        if t < window: return
        
        # Fit Parabola
        y = prices[t-window:t]
        x = np.arange(window)
        a, b, c = np.polyfit(x, y, 2)
        
        # Buy the "Smile" (U-turn)
        if a > 0.1: self.buy(prices[t], 0.2)
        # Sell the "Frown" (n-turn)
        elif a < -0.1: self.sell(prices[t], 0.2)

# --- STRATEGY 3: THE HYBRID GRANDMASTER ---
class HybridBot(Trader):
    """
    1. Trend Filter: Only Buy if Long-Term Trend (50 days) is UP.
    2. Geometric Entry: Only Buy if Short-Term Curvature (20 days) is a "Smile".
    3. Crash Detection: If Price falls below Manifold floor, PANIC SELL.
    """
    def strategy(self, t, prices):
        if t < 50: return
        
        # 1. Global Trend (Linear Regression)
        # Are we in a Bull or Bear market?
        long_term = prices[t-50:t]
        slope, _ = np.polyfit(np.arange(50), long_term, 1)
        
        # 2. Local Geometry (Parabolic)
        # Is the dip turning around?
        short_term = prices[t-20:t]
        curvature, b, c = np.polyfit(np.arange(20), short_term, 2)
        
        price = prices[t]
        
        # LOGIC:
        if slope < -0.5:
            # BEAR MARKET DETECTED!
            # Don't buy dips. Just cash out.
            self.sell_all(price)
            
        elif slope > 0:
            # BULL MARKET
            # Use Geometry to time the entry
            if curvature > 0.05: # "The Smile"
                self.buy(price, 0.2)
            elif curvature < -0.05: # "The Frown"
                self.sell(price, 0.2) # Take profit

def run_crash_test():
    market = CrashMarket()
    
    bots = [
        MomentumBot("Chaser (Blue)", "blue"),
        PolyBot("Geometer (Green)", "green"),
        HybridBot("Hybrid (Gold)", "orange")
    ]
    
    price_hist = []
    
    print("--- AOTI CRASH TEST ---")
    print("Phase 1: Bull Market (Chaser wins)")
    print("Phase 2: The Crash (Chaser dies)")
    print("Watch the Orange (Hybrid) Bot.")
    
    for t in range(SIM_LENGTH):
        price = market.get_price(t)
        price_hist.append(price)
        
        for bot in bots:
            bot.strategy(t, price_hist)
            bot.history.append(bot.get_value(price))
            
    # VISUALIZATION
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(price_hist, color='black', alpha=0.5, label="Market")
    ax1.set_title("Market Price (The Crash)")
    ax1.grid(True)
    
    for bot in bots:
        ax2.plot(bot.history, color=bot.color, linewidth=2, label=f"{bot.name}: ${int(bot.history[-1])}")
        
    ax2.set_title("Net Worth (Survival of the Fittest)")
    ax2.legend()
    ax2.grid(True)
    
    plt.show()

if __name__ == "__main__":
    run_crash_test()