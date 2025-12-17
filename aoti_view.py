#!/usr/bin/env python3
# AOTI LEVEL 44: THE BANKROLL MANAGER (FINAL)
# Upgrades:
# 1. Dynamic Position Sizing (% of Cash).
# 2. Honest Cash Tracking (Simulated or Real).
# 3. Prevents "Insufficient Funds" errors.

import time
import json
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
import robin_stocks.robinhood as r

# --- CONFIGURATION ---
CONFIG_FILE = 'fund_config.json'
DB_FILE = 'aoti_portfolio.db'

# --- INFRASTRUCTURE ---
def load_config():
    if not os.path.exists(CONFIG_FILE):
        # NEW DNA: "allocation_pct" instead of "risk_per_trade"
        # 0.10 means "Use 10% of available cash per trade"
        default_conf = {
            "slope_threshold": 0.002, 
            "allocation_pct": 0.10, 
            "sim_cash": 10000.0
        }
        with open(CONFIG_FILE, 'w') as f: json.dump(default_conf, f)
        return default_conf
    with open(CONFIG_FILE, 'r') as f: return json.load(f)

def save_config(conf):
    with open(CONFIG_FILE, 'w') as f: json.dump(conf, f, indent=4)

engine = create_engine(f'sqlite:///{DB_FILE}')
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

class TradeLog(Base):
    __tablename__ = 'trade_memory'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.now)
    ticker = Column(String)
    action = Column(String)
    price = Column(Float)
    is_closed = Column(Boolean, default=False)
    entry_time = Column(DateTime, default=datetime.now) 
    sell_price = Column(Float, nullable=True)
    profit = Column(Float, nullable=True)
    # Track how much we actually bought (Qty)
    qty = Column(Float, default=0.0)

Base.metadata.create_all(engine)

# --- MODULE 1: THE HUNTER ---
class MarketHunter:
    def __init__(self, config):
        self.config = config
        self.universe = ['BTC-USD', 'ETH-USD', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMD', 'COIN']
        
    def scan(self, current_holdings):
        print(f"--- PHASE 1: HUNTING (Threshold: {self.config['slope_threshold']:.5f}) ---")
        candidates = []
        
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        data = yf.download(self.universe, period="3mo", progress=False, auto_adjust=True)
        if 'Close' in data: data = data['Close']

        for ticker in self.universe:
            if ticker in current_holdings: continue

            try:
                if ticker not in data.columns: continue
                prices = data[ticker].dropna().values
                if len(prices) < 50: continue
                
                log_prices = np.log(prices)
                x = np.arange(50)
                slope, _ = np.polyfit(x, log_prices[-50:], 1)
                
                if slope > self.config['slope_threshold']:
                    print(f">> Candidate Found: {ticker} (Slope: {slope:.5f})")
                    candidates.append(ticker)
            except Exception: continue
        return candidates

# --- MODULE 2: THE BANKER (Execution + Wallet) ---
class ExecutionBridge:
    def __init__(self, config):
        self.config = config
        self.connected = False # Set to True if using Real Robinhood

    def connect(self):
        # UNCOMMENT FOR REAL TRADING
        # r.login(username="...", password="...", mfa_code=...)
        # self.connected = True
        pass

    def get_buying_power(self):
        """Returns available cash to spend."""
        if self.connected:
            try:
                profile = r.profiles.load_account_profile()
                return float(profile['buying_power'])
            except:
                print("xx Error reading Robinhood Profile")
                return 0.0
        else:
            # SIMULATION MODE: Read from JSON memory
            return self.config.get('sim_cash', 10000.0)

    def update_sim_wallet(self, amount_change):
        """Updates the simulated cash balance after a trade."""
        if not self.connected:
            current = self.config.get('sim_cash', 10000.0)
            new_bal = current + amount_change
            self.config['sim_cash'] = new_bal
            save_config(self.config)
            print(f">> WALLET UPDATE: ${current:.2f} -> ${new_bal:.2f}")

    def execute_buy(self, ticker):
        # 1. Get Price
        df = yf.download(ticker, period="1d", progress=False, auto_adjust=True)
        if df.empty: return 0.0, 0.0
        price = float(df['Close'].values[-1].item())
        
        # 2. Calculate Sizing (The Upgrade)
        cash = self.get_buying_power()
        alloc_pct = self.config.get('allocation_pct', 0.10)
        
        # Calculate bet size: Cash * %
        bet_size = cash * alloc_pct
        
        # Safety check: Don't bet less than $10 or more than we have
        if bet_size < 10:
            print(f"xx Insufficient funds to buy {ticker} (Cash: ${cash:.2f})")
            return 0.0, 0.0
            
        qty = bet_size / price
        
        print(f">> ORDER: BUY ${bet_size:.2f} of {ticker} ({qty:.4f} shares)")
        
        if self.connected:
            # REAL ORDER
            # r.order_buy_fractional_by_price(ticker, bet_size)
            pass
        else:
            # SIM ORDER: Deduct cash
            self.update_sim_wallet(-bet_size)

        return price, qty

    def execute_sell(self, ticker, qty):
        df = yf.download(ticker, period="1d", progress=False, auto_adjust=True)
        if df.empty: return 0.0
        price = float(df['Close'].values[-1].item())
        
        sale_value = price * qty
        print(f">> ORDER: SELL {ticker} ({qty:.4f} shares) for ${sale_value:.2f}")
        
        if self.connected:
            # REAL ORDER
            # r.order_sell_fractional_by_price(ticker, sale_value)
            pass
        else:
            # SIM ORDER: Add cash back
            self.update_sim_wallet(sale_value)

        return price

# --- MODULE 3: THE PORTFOLIO MANAGER ---
class PortfolioManager:
    def manage(self, executor):
        print("--- PHASE 2: MANAGING HOLDINGS ---")
        holdings = session.query(TradeLog).filter(TradeLog.is_closed == False).all()
        held_tickers = [t.ticker for t in holdings]
        
        if not holdings:
            print("No active positions.")
            return []

        for trade in holdings:
            df = yf.download(trade.ticker, period="3mo", progress=False, auto_adjust=True)
            if df.empty: continue
            
            prices = df['Close'].values
            if len(prices) < 50: continue
            
            log_prices = np.log(prices)
            x = np.arange(50)
            slope, _ = np.polyfit(x, log_prices[-50:], 1)
            
            sell_threshold = -0.001 
            
            if slope < sell_threshold:
                print(f">> EXIT SIGNAL: {trade.ticker} (Slope dropped to {slope:.5f})")
                
                # Use stored Quantity if available, else estimate (legacy support)
                qty_to_sell = trade.qty if trade.qty else (trade.price / trade.price) # Fallback 
                
                price = executor.execute_sell(trade.ticker, qty_to_sell)
                
                trade.is_closed = True
                trade.sell_price = price
                trade.profit = (price - trade.price) / trade.price
                session.commit()
            else:
                print(f"   > Holding {trade.ticker} (Slope: {slope:.5f})")
                
        return held_tickers 

# --- MODULE 4: THE STUDENT ---
class SelfReflector:
    def review(self, current_config):
        print("--- PHASE 3: SELF REFLECTION ---")
        cutoff = datetime.now() - timedelta(days=1)
        trades = session.query(TradeLog).filter(TradeLog.is_closed == False, TradeLog.entry_time < cutoff).all()
        
        if not trades:
            print("No mature trades (>24h) to review.")
            return current_config

        wins = 0; total = 0
        for t in trades:
            df = yf.download(t.ticker, period="1d", progress=False, auto_adjust=True)
            if df.empty: continue
            curr_price = df['Close'].values[-1].item()
            if (curr_price - t.price) > 0: wins += 1
            total += 1

        if total == 0: return current_config
        
        win_rate = wins / total
        print(f">> Mature Win Rate: {win_rate*100:.1f}%")

        new_config = current_config.copy()
        if win_rate < 0.40:
            print("!! PERFORMANCE LOW. Tightening standards.")
            new_config['slope_threshold'] *= 1.10
            save_config(new_config)
        elif win_rate > 0.70:
            print("++ PERFORMANCE HIGH. Expanding hunt.")
            new_config['slope_threshold'] *= 0.95
            save_config(new_config)
            
        return new_config

def run_fund():
    print(f"=== AOTI FUND CYCLE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    config = load_config()
    
    executor = ExecutionBridge(config)
    executor.connect() # Try to connect real broker
    
    hunter = MarketHunter(config)
    manager = PortfolioManager()
    student = SelfReflector()
    
    # 1. Manage
    current_holdings = manager.manage(executor)
    
    # 2. Hunt
    targets = hunter.scan(current_holdings)
    
    for ticker in targets:
        # Pass the executor so it checks cash balance
        price, qty = executor.execute_buy(ticker)
        
        if price > 0 and qty > 0:
            log = TradeLog(ticker=ticker, action="BUY", price=price, qty=qty, is_closed=False)
            session.add(log)
            session.commit()
            
    # 3. Learn
    student.review(config)
    print("=== CYCLE COMPLETE ===")

if __name__ == "__main__":
    run_fund()