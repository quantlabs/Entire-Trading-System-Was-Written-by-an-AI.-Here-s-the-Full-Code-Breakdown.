# Import necessary libraries
import asyncio
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os
import sys
from scipy.optimize import minimize
from scipy.stats import norm
import argparse
import json
from pathlib import Path
import colorama
from colorama import Fore, Back, Style
from tabulate import tabulate

# Initialize colorama for cross-platform colored terminal output
colorama.init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MicroFuturesTrader")

# Create and set event loop for the current thread
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Now it's safe to import ib_insync
from ib_insync import *

# ----------------------------------
# TERMINAL DISPLAY UTILITIES
# ----------------------------------

class TerminalDisplay:
    """Utility class for formatted terminal output"""
    
    @staticmethod
    def clear_screen():
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def print_header(text):
        """Print a formatted header"""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'-' * len(text)}")
    
    @staticmethod
    def print_status(status, details=""):
        """Print a status message with optional details"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if status.lower() == "success":
            print(f"{Fore.GREEN}[✓] {timestamp} - {details}")
        elif status.lower() == "error":
            print(f"{Fore.RED}[✗] {timestamp} - {details}")
        elif status.lower() == "warning":
            print(f"{Fore.YELLOW}[!] {timestamp} - {details}")
        elif status.lower() == "info":
            print(f"{Fore.BLUE}[i] {timestamp} - {details}")
        else:
            print(f"{timestamp} - {details}")
    
    @staticmethod
    def print_connection_status(connected, host, port, last_connected=None):
        """Print connection status"""
        if connected:
            status = f"{Fore.GREEN}Connected to TWS at {host}:{port}"
            if last_connected:
                duration = datetime.now() - last_connected
                hours, remainder = divmod(duration.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                status += f" for {hours}h {minutes}m {seconds}s"
        else:
            status = f"{Fore.RED}Disconnected from TWS"
        
        print(status)
    
    @staticmethod
    def print_market_data(market_data, symbols=None):
        """Print formatted market data table"""
        if not market_data:
            print(f"{Fore.YELLOW}No market data available")
            return
            
        if not symbols:
            symbols = list(market_data.keys())
            
        table_data = []
        headers = ["Symbol", "Last", "Bid", "Ask", "Volume", "Updated"]
        
        for symbol in symbols:
            if symbol in market_data:
                data = market_data[symbol]
                updated = data.get('time', 'N/A')
                if isinstance(updated, datetime):
                    updated = updated.strftime("%H:%M:%S")
                    
                table_data.append([
                    symbol,
                    f"{data.get('last', 'N/A'):.2f}" if data.get('last') is not None else 'N/A',
                    f"{data.get('bid', 'N/A'):.2f}" if data.get('bid') is not None else 'N/A',
                    f"{data.get('ask', 'N/A'):.2f}" if data.get('ask') is not None else 'N/A',
                    data.get('volume', 'N/A'),
                    updated
                ])
        
        if table_data:
            print(tabulate(table_data, headers=headers, tablefmt="simple"))
        else:
            print(f"{Fore.YELLOW}No data for specified symbols")
    
    @staticmethod
    def print_portfolio_status(portfolio):
        """Print portfolio status"""
        print(f"\n{Fore.GREEN}Portfolio Value: ${portfolio.current_capital:.2f}")
        print(f"{Fore.BLUE}P&L: ${(portfolio.current_capital - portfolio.initial_capital):.2f}")
        
        # Print strategy performance
        table_data = []
        headers = ["Strategy", "Capital", "P&L", "Status"]
        
        for name, strategy in portfolio.strategies.items():
            status = f"{Fore.GREEN}Active" if strategy.active else f"{Fore.RED}Inactive"
            table_data.append([
                name,
                f"${strategy.capital:.2f}",
                f"${strategy.pl:.2f}",
                status
            ])
        
        print("\n" + tabulate(table_data, headers=headers, tablefmt="simple"))
    
    @staticmethod
    def print_positions(positions):
        """Print current positions"""
        if not positions:
            print(f"{Fore.YELLOW}No open positions")
            return
            
        table_data = []
        headers = ["Symbol", "Position", "Avg Cost", "Value"]
        
        for symbol, data in positions.items():
            table_data.append([
                symbol,
                data.get('position', 'N/A'),
                f"${data.get('avgCost', 'N/A'):.2f}" if data.get('avgCost') is not None else 'N/A',
                f"${data.get('position', 0) * data.get('avgCost', 0):.2f}" if data.get('position') is not None and data.get('avgCost') is not None else 'N/A'
            ])
        
        print("\n" + tabulate(table_data, headers=headers, tablefmt="simple"))
    
    @staticmethod
    def print_recent_trades(trades, limit=5):
        """Print recent trades"""
        if not trades:
            print(f"{Fore.YELLOW}No recent trades")
            return
            
        # Sort by time, most recent first
        sorted_trades = sorted(trades, key=lambda x: x.get('time', datetime.min), reverse=True)
        
        table_data = []
        headers = ["Time", "Symbol", "Action", "Quantity", "Price", "Value"]
        
        for trade in sorted_trades[:limit]:
            time_str = trade.get('time', 'N/A')
            if isinstance(time_str, datetime):
                time_str = time_str.strftime("%H:%M:%S")
                
            table_data.append([
                time_str,
                trade.get('symbol', 'N/A'),
                trade.get('action', 'N/A'),
                trade.get('quantity', 'N/A'),
                f"${trade.get('price', 'N/A'):.2f}" if trade.get('price') is not None else 'N/A',
                f"${trade.get('value', 'N/A'):.2f}" if trade.get('value') is not None else 'N/A'
            ])
        
        print("\n" + tabulate(table_data, headers=headers, tablefmt="simple"))
    
    @staticmethod
    def print_system_status(ib_manager):
        """Print overall system status"""
        connection_time = ib_manager.last_connection_time
        connection_status = "Connected" if ib_manager.connected else "Disconnected"
        status_color = Fore.GREEN if ib_manager.connected else Fore.RED
        
        print(f"\n{Fore.CYAN}System Status: {status_color}{connection_status}")
        
        if connection_time:
            duration = datetime.now() - connection_time
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"{Fore.CYAN}Connection Duration: {hours}h {minutes}m {seconds}s")
        
        # Print last few log entries
        print(f"\n{Fore.CYAN}Recent Events:")
        
        # Combine all logs and sort by time
        all_logs = []
        for event in ib_manager.connection_status_log[-3:]:
            all_logs.append((event['time'], 'CONNECTION', event['event']))
        
        for event in ib_manager.market_data_log[-3:]:
            all_logs.append((event['time'], 'MARKET DATA', event['event']))
            
        for event in ib_manager.order_execution_log[-3:]:
            all_logs.append((event['time'], 'ORDER', event['event']))
        
        # Sort by time (most recent first) and take the 5 most recent
        all_logs.sort(key=lambda x: x[0], reverse=True)
        
        for time, category, event in all_logs[:5]:
            if isinstance(time, datetime):
                time_str = time.strftime("%H:%M:%S")
            else:
                time_str = str(time)
                
            if category == 'CONNECTION':
                color = Fore.BLUE
            elif category == 'MARKET DATA':
                color = Fore.GREEN
            else:  # ORDER
                color = Fore.YELLOW
                
            print(f"{time_str} - {color}{category}{Fore.RESET}: {event}")

# ----------------------------------
# IB API INTEGRATION
# ----------------------------------

class IBManager:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1, config_path=None):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.config_path = config_path
        self.ib = IB()
        self.connected = False
        self.market_data = {}
        self.positions = {}
        self.account_summary = {}
        self.data_lock = threading.Lock()
        self.order_history = []
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.last_connection_time = None
        self.connection_status_log = []
        self.market_data_log = []
        self.order_execution_log = []
        
    def connect(self):
        """Connect to IB TWS with enhanced logging and retry logic"""
        if not self.connected:
            self.connection_attempts += 1
            try:
                logger.info(f"Connecting to IB TWS at {self.host}:{self.port} (Attempt {self.connection_attempts})")
                TerminalDisplay.print_status("info", f"Connecting to TWS at {self.host}:{self.port} (Attempt {self.connection_attempts})")
                
                self.connection_status_log.append({
                    'time': datetime.now(),
                    'event': 'connection_attempt',
                    'attempt': self.connection_attempts,
                    'host': self.host,
                    'port': self.port
                })
                
                # Use the synchronous connect method
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                self.connected = True
                self.last_connection_time = datetime.now()
                
                connection_msg = f"Connected to IB TWS at {self.host}:{self.port}"
                logger.info(connection_msg)
                TerminalDisplay.print_status("success", connection_msg)
                
                self.connection_status_log.append({
                    'time': datetime.now(),
                    'event': 'connected',
                    'host': self.host,
                    'port': self.port
                })
                
                # Set up event handlers
                self.ib.pendingTickersEvent += self.on_ticker_update
                self.ib.positionEvent += self.on_position_update
                self.ib.accountSummaryEvent += self.on_account_update
                self.ib.execDetailsEvent += self.on_execution
                self.ib.errorEvent += self.on_error
                self.ib.disconnectedEvent += self.on_disconnect
                
                # Request account updates - FIXED: removed incorrect parameters
                self.ib.reqAccountUpdates()
                logger.info("Requested account updates")
                TerminalDisplay.print_status("info", "Requested account updates")
                
                # Start the event loop
                self.ib_thread = threading.Thread(target=self.run_event_loop, daemon=True)
                self.ib_thread.start()
                logger.info("IB event loop started")
                
                # Reset connection attempts on successful connection
                self.connection_attempts = 0
                
                return True
            except Exception as e:
                error_msg = f"Failed to connect to IB TWS: {str(e)}"
                logger.error(error_msg)
                TerminalDisplay.print_status("error", error_msg)
                
                self.connection_status_log.append({
                    'time': datetime.now(),
                    'event': 'connection_error',
                    'error': str(e),
                    'attempt': self.connection_attempts
                })
                
                # If exceeded max attempts, wait longer before next try
                if self.connection_attempts >= self.max_connection_attempts:
                    wait_msg = f"Max connection attempts ({self.max_connection_attempts}) reached. Waiting 60 seconds before retry."
                    logger.warning(wait_msg)
                    TerminalDisplay.print_status("warning", wait_msg)
                    time.sleep(60)
                    self.connection_attempts = 0
                else:
                    time.sleep(5)  # Wait 5 seconds before retry
                    
                return False
        return True
        
    def disconnect(self):
        """Disconnect from IB TWS with logging"""
        if self.connected:
            try:
                self.ib.disconnect()
                self.connected = False
                logger.info("Disconnected from IB TWS")
                TerminalDisplay.print_status("info", "Disconnected from IB TWS")
                
                self.connection_status_log.append({
                    'time': datetime.now(),
                    'event': 'disconnected',
                    'reason': 'user_request'
                })
            except Exception as e:
                error_msg = f"Error during disconnect: {str(e)}"
                logger.error(error_msg)
                TerminalDisplay.print_status("error", error_msg)
                
                self.connection_status_log.append({
                    'time': datetime.now(),
                    'event': 'disconnect_error',
                    'error': str(e)
                })
            
    def on_disconnect(self):
        """Handle unexpected disconnections"""
        self.connected = False
        logger.warning("Unexpectedly disconnected from IB TWS")
        TerminalDisplay.print_status("warning", "Unexpectedly disconnected from IB TWS")
        
        self.connection_status_log.append({
            'time': datetime.now(),
            'event': 'disconnected',
            'reason': 'unexpected'
        })
        
        # Attempt to reconnect
        time.sleep(5)  # Wait 5 seconds before reconnecting
        self.connect()
            
    def run_event_loop(self):
        """Run the IB event loop with health checks"""
        last_health_check = datetime.now()
        
        while self.connected:
            self.ib.sleep(0.1)
            
            # Perform health check every 60 seconds
            now = datetime.now()
            if (now - last_health_check).total_seconds() > 60:
                self.check_connection_health()
                last_health_check = now
    
    def check_connection_health(self):
        """Verify the connection is healthy and functioning"""
        try:
            # Request current time as a simple health check
            current_time = self.ib.reqCurrentTime()
            time_diff = abs((datetime.now() - current_time).total_seconds())
            
            logger.debug(f"Connection health check: IB server time differs by {time_diff:.2f} seconds")
            
            # If time difference is large, log a warning
            if time_diff > 30:
                warning_msg = f"Large time difference with IB server: {time_diff:.2f} seconds"
                logger.warning(warning_msg)
                TerminalDisplay.print_status("warning", warning_msg)
                
        except Exception as e:
            error_msg = f"Connection health check failed: {str(e)}"
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            
            # Connection might be dead, mark as disconnected and try to reconnect
            self.connected = False
            self.connect()
    
    def on_error(self, reqId, errorCode, errorString, contract):
        """Handle IB API errors"""
        error_msg = f"IB Error {errorCode}: {errorString}"
        if contract:
            error_msg += f" for contract: {contract.symbol}"
            
        if errorCode in [1100, 1101, 1102, 2110]:  # Connection-related errors
            logger.critical(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            
            self.connection_status_log.append({
                'time': datetime.now(),
                'event': 'connection_error',
                'error_code': errorCode,
                'error_message': errorString
            })
        elif errorCode in [2104, 2106, 2158]:  # Market data errors
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            
            self.market_data_log.append({
                'time': datetime.now(),
                'event': 'market_data_error',
                'error_code': errorCode,
                'error_message': errorString,
                'contract': contract.symbol if contract else None
            })
        elif errorCode in [201, 202, 203, 399]:  # Order errors
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            
            self.order_execution_log.append({
                'time': datetime.now(),
                'event': 'order_error',
                'error_code': errorCode,
                'error_message': errorString,
                'contract': contract.symbol if contract else None
            })
        else:
            logger.warning(error_msg)
            
    def on_ticker_update(self, tickers):
        """Handle market data updates with enhanced logging"""
        with self.data_lock:
            for ticker in tickers:
                symbol = ticker.contract.symbol
                
                if symbol not in self.market_data:
                    self.market_data[symbol] = {}
                    logger.info(f"Started receiving market data for {symbol}")
                    TerminalDisplay.print_status("info", f"Started receiving market data for {symbol}")
                    
                    self.market_data_log.append({
                        'time': datetime.now(),
                        'event': 'market_data_start',
                        'symbol': symbol
                    })
                    
                # Check if price has changed significantly (more than 0.1%)
                old_price = self.market_data.get(symbol, {}).get('last', None)
                new_price = ticker.last if ticker.last > 0 else None
                
                if old_price and new_price and abs((new_price - old_price) / old_price) > 0.001:
                    logger.info(f"Significant price change for {symbol}: {old_price} -> {new_price}")
                    
                    self.market_data_log.append({
                        'time': datetime.now(),
                        'event': 'significant_price_change',
                        'symbol': symbol,
                        'old_price': old_price,
                        'new_price': new_price,
                        'change_pct': ((new_price - old_price) / old_price) * 100
                    })
                
                self.market_data[symbol]['bid'] = ticker.bid if ticker.bid > 0 else None
                self.market_data[symbol]['ask'] = ticker.ask if ticker.ask > 0 else None
                self.market_data[symbol]['last'] = new_price
                self.market_data[symbol]['close'] = ticker.close if ticker.close > 0 else None
                self.market_data[symbol]['volume'] = ticker.volume if ticker.volume > 0 else None
                self.market_data[symbol]['time'] = datetime.now()
                
    def on_position_update(self, position):
        """Handle position updates with logging"""
        with self.data_lock:
            symbol = position.contract.symbol
            
            # Check if this is a new position or position change
            old_position = self.positions.get(symbol, {}).get('position', 0)
            new_position = position.position
            
            if old_position != new_position:
                position_msg = f"Position changed for {symbol}: {old_position} -> {new_position}"
                logger.info(position_msg)
                TerminalDisplay.print_status("info", position_msg)
                
                self.market_data_log.append({
                    'time': datetime.now(),
                    'event': 'position_change',
                    'symbol': symbol,
                    'old_position': old_position,
                    'new_position': new_position
                })
            
            self.positions[symbol] = {
                'position': position.position,
                'avgCost': position.avgCost,
                'contract': position.contract
            }
            
    def on_account_update(self, account_value):
        """Handle account updates with logging"""
        with self.data_lock:
            key = account_value.tag
            
            # Log important account values
            important_keys = ['NetLiquidation', 'AvailableFunds', 'MaintMarginReq', 'ExcessLiquidity']
            if key in important_keys:
                old_value = self.account_summary.get(key)
                new_value = account_value.value
                
                if old_value != new_value:
                    logger.info(f"Account {key} changed: {old_value} -> {new_value}")
            
            self.account_summary[key] = account_value.value
            
    def on_execution(self, trade, fill):
        """Handle execution events with detailed logging"""
        order_details = {
            'time': datetime.now(),
            'symbol': trade.contract.symbol,
            'action': fill.execution.side,
            'quantity': fill.execution.shares,
            'price': fill.execution.price,
            'value': fill.execution.shares * fill.execution.price,
            'commission': fill.commissionReport.commission if hasattr(fill, 'commissionReport') else 0,
            'execution_id': fill.execution.execId,
            'order_id': trade.order.orderId,
            'account': fill.execution.acctNumber
        }
        
        self.order_history.append(order_details)
        
        # Log execution details
        execution_msg = f"Trade executed: {order_details['action']} {order_details['quantity']} {order_details['symbol']} @ {order_details['price']}"
        logger.info(execution_msg)
        TerminalDisplay.print_status("success", execution_msg)
        
        # Add to execution log
        self.order_execution_log.append({
            'time': datetime.now(),
            'event': 'trade_executed',
            'details': order_details
        })
            
    def get_market_data(self):
        """Get current market data"""
        with self.data_lock:
            return self.market_data.copy()
            
    def get_positions(self):
        """Get current positions"""
        with self.data_lock:
            return self.positions.copy()
            
    def get_account_summary(self):
        """Get account summary"""
        with self.data_lock:
            return self.account_summary.copy()
            
    def get_order_history(self):
        """Get order history"""
        with self.data_lock:
            return self.order_history.copy()
            
    def subscribe_market_data(self, contracts):
        """Subscribe to real-time market data for the given contracts with logging"""
        if not self.connected:
            error_msg = "Not connected to IB TWS - cannot subscribe to market data"
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            return False
            
        try:
            for contract in contracts:
                # Request market data
                self.ib.reqMktData(contract)
                subscribe_msg = f"Subscribed to market data for {contract.symbol}"
                logger.info(subscribe_msg)
                TerminalDisplay.print_status("success", subscribe_msg)
                
                self.market_data_log.append({
                    'time': datetime.now(),
                    'event': 'market_data_subscription',
                    'symbol': contract.symbol,
                    'exchange': contract.exchange,
                    'contract_type': contract.__class__.__name__
                })
                
            return True
        except Exception as e:
            error_msg = f"Failed to subscribe to market data: {str(e)}"
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            
            self.market_data_log.append({
                'time': datetime.now(),
                'event': 'market_data_subscription_error',
                'error': str(e),
                'contracts': [c.symbol for c in contracts]
            })
            return False
            
    def create_micro_future_contract(self, symbol, exchange='GLOBEX', expiry=None):
        """Create a micro futures contract with logging"""
        logger.info(f"Creating micro future contract for {symbol} on {exchange}")
        TerminalDisplay.print_status("info", f"Creating contract for {symbol}")
        
        try:
            contract = Future(symbol=symbol, exchange=exchange)
            
            if expiry:
                contract.lastTradeDateOrContractMonth = expiry
                
            # Qualify the contract to get additional details
            qualified_contracts = self.ib.qualifyContracts(contract)
            
            if qualified_contracts:
                success_msg = f"Successfully created contract for {symbol}"
                logger.info(success_msg)
                TerminalDisplay.print_status("success", success_msg)
                return qualified_contracts[0]
            else:
                error_msg = f"Could not qualify contract for {symbol}"
                logger.error(error_msg)
                TerminalDisplay.print_status("error", error_msg)
                return None
        except Exception as e:
            error_msg = f"Error creating future contract for {symbol}: {str(e)}"
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            return None
            
    def create_micro_option_contract(self, symbol, right, strike, expiry, exchange='GLOBEX'):
        """Create a micro option contract with logging"""
        logger.info(f"Creating micro option contract for {symbol} {strike} {right} {expiry}")
        TerminalDisplay.print_status("info", f"Creating option for {symbol} {strike} {right}")
        
        try:
            # First get the underlying future contract
            underlying = self.create_micro_future_contract(symbol, exchange, expiry)
            
            if not underlying:
                return None
                
            # Create the option contract
            option = FuturesOption(
                symbol=symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=strike,
                right=right,  # 'C' for Call, 'P' for Put
                exchange=exchange
            )
            
            # Qualify the contract
            qualified_options = self.ib.qualifyContracts(option)
            
            if qualified_options:
                success_msg = f"Successfully created option contract for {symbol} {strike} {right}"
                logger.info(success_msg)
                TerminalDisplay.print_status("success", success_msg)
                return qualified_options[0]
            else:
                error_msg = f"Could not qualify option contract for {symbol} {strike} {right}"
                logger.error(error_msg)
                TerminalDisplay.print_status("error", error_msg)
                return None
        except Exception as e:
            error_msg = f"Error creating option contract: {str(e)}"
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            return None
            
    def place_market_order(self, contract, action, quantity):
        """Place a market order with detailed logging"""
        if not self.connected:
            error_msg = "Not connected to IB TWS - cannot place market order"
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            return None
            
        try:
            # Log the order attempt
            attempt_msg = f"Placing {action} market order for {quantity} {contract.symbol}"
            logger.info(attempt_msg)
            TerminalDisplay.print_status("info", attempt_msg)
            
            self.order_execution_log.append({
                'time': datetime.now(),
                'event': 'order_attempt',
                'type': 'market',
                'action': action,
                'quantity': quantity,
                'symbol': contract.symbol
            })
            
            # Create order
            order = MarketOrder(action, quantity)
            
            # Place order
            #trade = self.ib.placeOrder(contract, order)
            trade = None
            
            # Log order placement
            placed_msg = f"Placed {action} market order for {quantity} {contract.symbol} (Order ID: {order.orderId})"
            logger.info(placed_msg)
            TerminalDisplay.print_status("success", placed_msg)
            
            self.order_execution_log.append({
                'time': datetime.now(),
                'event': 'order_placed',
                'type': 'market',
                'action': action,
                'quantity': quantity,
                'symbol': contract.symbol,
                'order_id': order.orderId
            })
            
            return trade
        except Exception as e:
            error_msg = f"Failed to place market order: {str(e)}"
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            
            self.order_execution_log.append({
                'time': datetime.now(),
                'event': 'order_error',
                'type': 'market',
                'action': action,
                'quantity': quantity,
                'symbol': contract.symbol,
                'error': str(e)
            })
            return None
            
    def place_limit_order(self, contract, action, quantity, price):
        """Place a limit order with detailed logging"""
        if not self.connected:
            error_msg = "Not connected to IB TWS - cannot place limit order"
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            return None
            
        try:
            # Log the order attempt
            attempt_msg = f"Placing {action} limit order for {quantity} {contract.symbol} at {price}"
            logger.info(attempt_msg)
            TerminalDisplay.print_status("info", attempt_msg)
            
            self.order_execution_log.append({
                'time': datetime.now(),
                'event': 'order_attempt',
                'type': 'limit',
                'action': action,
                'quantity': quantity,
                'symbol': contract.symbol,
                'price': price
            })
            
            # Create order
            order = LimitOrder(action, quantity, price)
            
            # Place order
            #trade = self.ib.placeOrder(contract, order)
            trade = None
            
            # Log order placement
            placed_msg = f"Placed {action} limit order for {quantity} {contract.symbol} at {price} (Order ID: {order.orderId})"
            logger.info(placed_msg)
            TerminalDisplay.print_status("success", placed_msg)
            
            self.order_execution_log.append({
                'time': datetime.now(),
                'event': 'order_placed',
                'type': 'limit',
                'action': action,
                'quantity': quantity,
                'symbol': contract.symbol,
                'price': price,
                'order_id': order.orderId
            })
            
            return trade
        except Exception as e:
            error_msg = f"Failed to place limit order: {str(e)}"
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            
            self.order_execution_log.append({
                'time': datetime.now(),
                'event': 'order_error',
                'type': 'limit',
                'action': action,
                'quantity': quantity,
                'symbol': contract.symbol,
                'price': price,
                'error': str(e)
            })
            return None
            
    def get_historical_data(self, contract, duration='1 D', bar_size='1 min'):
        """Get historical data for the given contract with logging"""
        if not self.connected:
            error_msg = "Not connected to IB TWS - cannot get historical data"
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            return None
            
        try:
            request_msg = f"Requesting historical data for {contract.symbol}: {duration} {bar_size}"
            logger.info(request_msg)
            TerminalDisplay.print_status("info", request_msg)
            
            self.market_data_log.append({
                'time': datetime.now(),
                'event': 'historical_data_request',
                'symbol': contract.symbol,
                'duration': duration,
                'bar_size': bar_size
            })
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract=contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True
            )
            
            # Convert to dataframe
            df = util.df(bars)
            
            received_msg = f"Received {len(df) if df is not None else 0} bars for {contract.symbol}"
            logger.info(received_msg)
            TerminalDisplay.print_status("success", received_msg)
            
            self.market_data_log.append({
                'time': datetime.now(),
                'event': 'historical_data_received',
                'symbol': contract.symbol,
                'bars_count': len(df) if df is not None else 0
            })
            
            return df
        except Exception as e:
            error_msg = f"Failed to get historical data: {str(e)}"
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            
            self.market_data_log.append({
                'time': datetime.now(),
                'event': 'historical_data_error',
                'symbol': contract.symbol,
                'error': str(e)
            })
            return None
    
    def save_logs(self, directory="logs"):
        """Save all logs to files"""
        try:
            # Ensure directory exists
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            # Get current timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save connection logs
            connection_file = f"{directory}/connection_log_{timestamp}.json"
            with open(connection_file, 'w') as f:
                json.dump(self.connection_status_log, f, default=str, indent=2)
                
            # Save market data logs
            market_data_file = f"{directory}/market_data_log_{timestamp}.json"
            with open(market_data_file, 'w') as f:
                json.dump(self.market_data_log, f, default=str, indent=2)
                
            # Save order execution logs
            order_file = f"{directory}/order_execution_log_{timestamp}.json"
            with open(order_file, 'w') as f:
                json.dump(self.order_execution_log, f, default=str, indent=2)
                
            save_msg = f"Logs saved to {directory}"
            logger.info(save_msg)
            TerminalDisplay.print_status("success", save_msg)
            return True
        except Exception as e:
            error_msg = f"Failed to save logs: {str(e)}"
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            return False

# ----------------------------------
# STRATEGY IMPLEMENTATIONS
# ----------------------------------

class Strategy:
    def __init__(self, name, capital, ib_manager):
        self.name = name
        self.capital = capital
        self.active = False
        self.ib = ib_manager
        self.positions = {}
        self.pl = 0.0
        self.daily_pl = []
        self.cumulative_pl = []
        self.timestamps = []
        self.last_update = datetime.now()
        self.forecast_return = 0.0
        self.signal_count = 0
        self.trade_count = 0
        
    def activate(self):
        """Activate the strategy"""
        self.active = True
        logger.info(f"Strategy {self.name} activated")
        TerminalDisplay.print_status("success", f"Strategy {self.name} activated")
        
    def deactivate(self):
        """Deactivate the strategy"""
        self.active = False
        logger.info(f"Strategy {self.name} deactivated")
        TerminalDisplay.print_status("info", f"Strategy {self.name} deactivated")
        
    def update_capital(self, new_capital):
        """Update the strategy capital"""
        self.capital = new_capital
        logger.info(f"Strategy {self.name} capital updated to ${new_capital}")
        
    def get_pl(self):
        """Get the strategy P&L"""
        return self.pl
        
    def update_pl(self, new_pl):
        """Update the strategy P&L"""
        self.pl = new_pl
        self.daily_pl.append(new_pl)
        self.timestamps.append(datetime.now())
        
        if not self.cumulative_pl:
            self.cumulative_pl.append(new_pl)
        else:
            self.cumulative_pl.append(self.cumulative_pl[-1] + new_pl)
            
    def check_signals(self, market_data):
        """Check for trading signals based on market data"""
        # To be implemented by subclasses
        pass
        
    def execute_trades(self, signals):
        """Execute trades based on signals"""
        # To be implemented by subclasses
        pass
        
    def close_positions(self):
        """Close all positions"""
        # To be implemented by subclasses
        pass
        
    def get_forecast_return(self):
        """Get forecasted return for the strategy"""
        return self.forecast_return
    
    def get_status_summary(self):
        """Get a summary of the strategy status"""
        return {
            "name": self.name,
            "active": self.active,
            "capital": self.capital,
            "pl": self.pl,
            "signals_generated": self.signal_count,
            "trades_executed": self.trade_count,
            "positions": self.positions
        }

class MicroCrudeOilArbitrageStrategy(Strategy):
    def __init__(self, name, capital, ib_manager, 
                 primary_symbol='MCL', secondary_symbol='USO',
                 z_entry=2.0, z_exit=0.5, lookback=20):
        super().__init__(name, capital, ib_manager)
        self.primary_symbol = primary_symbol  # Micro Crude Oil futures
        self.secondary_symbol = secondary_symbol  # Oil ETF for arbitrage
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.lookback = lookback
        self.price_history = {
            primary_symbol: pd.DataFrame(),
            secondary_symbol: pd.DataFrame()
        }
        self.spread_mean = None
        self.spread_std = None
        self.position_direction = None  # 'long' or 'short'
        self.entry_prices = {}
        self.primary_contract = None
        self.secondary_contract = None
        self.forecast_return = 0.08  # Initial forecast return
        self.current_z_score = None
        self.last_spread = None
        
        logger.info(f"Initialized {name} strategy with primary={primary_symbol}, secondary={secondary_symbol}")
        TerminalDisplay.print_status("info", f"Initialized {name} strategy")
        
    def initialize_contracts(self):
        """Initialize contracts for trading"""
        logger.info(f"Initializing contracts for {self.name}")
        TerminalDisplay.print_status("info", f"Initializing contracts for {self.name}")
        
        # Get the front-month micro crude oil contract
        self.primary_contract = self.ib.create_micro_future_contract(self.primary_symbol)
        
        # For the ETF, create a stock contract
        self.secondary_contract = Stock(symbol=self.secondary_symbol, exchange='SMART', currency='USD')
        
        # Subscribe to market data
        if self.primary_contract and self.secondary_contract:
            success = self.ib.subscribe_market_data([self.primary_contract, self.secondary_contract])
            status_msg = f"Contract initialization for {self.name}: {'Success' if success else 'Failed'}"
            logger.info(status_msg)
            TerminalDisplay.print_status("success" if success else "error", status_msg)
            return success
            
        logger.error(f"Failed to initialize contracts for {self.name}")
        TerminalDisplay.print_status("error", f"Failed to initialize contracts for {self.name}")
        return False
        
    def check_signals(self, market_data):
        if not self.active or not self.primary_contract or not self.secondary_contract:
            return None
            
        # Update price history with latest data
        for symbol in [self.primary_symbol, self.secondary_symbol]:
            if symbol in market_data and 'last' in market_data[symbol]:
                # Append new data point
                new_data = pd.DataFrame({
                    'timestamp': [datetime.now()],
                    'price': [market_data[symbol]['last']]
                })
                self.price_history[symbol] = pd.concat([self.price_history[symbol], new_data])
                
                # Keep only lookback period
                self.price_history[symbol] = self.price_history[symbol].tail(self.lookback)
        
        # Check if we have enough data for all symbols
        for symbol in [self.primary_symbol, self.secondary_symbol]:
            if len(self.price_history[symbol]) < self.lookback:
                return None
                
        # Calculate spread ratio
        primary_prices = self.price_history[self.primary_symbol]['price'].values
        secondary_prices = self.price_history[self.secondary_symbol]['price'].values
        
        # For micro crude oil vs ETF, need a scaling factor to account for contract size
        # MCL is 100 barrels, so adjust based on the relative price of the ETF
        scaling_factor = primary_prices[-1] / secondary_prices[-1] * 0.1  # Example adjustment
        
        # Calculate adjusted spread
        spread = primary_prices / (secondary_prices * scaling_factor)
        self.last_spread = spread[-1]
        
        # Calculate z-score if we have historical data
        if self.spread_mean is None or self.spread_std is None:
            self.spread_mean = np.mean(spread)
            self.spread_std = np.std(spread)
            baseline_msg = f"{self.name}: Calculated baseline spread metrics - mean={self.spread_mean:.4f}, std={self.spread_std:.4f}"
            logger.info(baseline_msg)
            TerminalDisplay.print_status("info", baseline_msg)
            return None
            
        current_spread = spread[-1]
        z_score = (current_spread - self.spread_mean) / self.spread_std
        self.current_z_score = z_score
        
        # Log z-score every 10 minutes
        if (datetime.now() - self.last_update).total_seconds() > 600:
            z_score_msg = f"{self.name}: Current z-score = {z_score:.4f}"
            logger.info(z_score_msg)
            self.last_update = datetime.now()
        
        signals = []
        
        # If no position, check for entry
        if self.position_direction is None:
            if z_score > self.z_entry:
                # Spread is too wide, expect convergence
                signal_msg = f"{self.name}: Entry signal - SHORT with z-score {z_score:.4f}"
                logger.info(signal_msg)
                TerminalDisplay.print_status("info", signal_msg)
                
                signals.append({
                    'action': 'entry',
                    'direction': 'short',
                    'z_score': z_score
                })
                self.signal_count += 1
            elif z_score < -self.z_entry:
                # Spread is too tight, expect divergence
                signal_msg = f"{self.name}: Entry signal - LONG with z-score {z_score:.4f}"
                logger.info(signal_msg)
                TerminalDisplay.print_status("info", signal_msg)
                
                signals.append({
                    'action': 'entry',
                    'direction': 'long',
                    'z_score': z_score
                })
                self.signal_count += 1
        # If we have a position, check for exit
        else:
            if (self.position_direction == 'long' and z_score > -self.z_exit) or \
               (self.position_direction == 'short' and z_score < self.z_exit):
                signal_msg = f"{self.name}: Exit signal with z-score {z_score:.4f}"
                logger.info(signal_msg)
                TerminalDisplay.print_status("info", signal_msg)
                
                signals.append({
                    'action': 'exit',
                    'z_score': z_score
                })
                self.signal_count += 1
                
        return signals if signals else None
        
    def execute_trades(self, signals):
        if not signals or not self.active:
            return
            
        for signal in signals:
            if signal['action'] == 'entry':
                # Enter new position
                direction = signal['direction']
                
                # Calculate position sizes based on current prices and capital
                primary_price = self.price_history[self.primary_symbol]['price'].values[-1]
                secondary_price = self.price_history[self.secondary_symbol]['price'].values[-1]
                
                # Calculate appropriate quantities for each instrument
                # For micro crude oil, contract value is current_price * 100 barrels
                primary_contract_value = primary_price * 100
                primary_qty = max(1, int((self.capital * 0.4) / primary_contract_value))
                
                # For ETF, straightforward calculation
                secondary_qty = max(1, int((self.capital * 0.4) / secondary_price))
                
                entry_msg = f"{self.name}: Executing {direction} spread entry"
                logger.info(entry_msg)
                TerminalDisplay.print_status("info", entry_msg)
                TerminalDisplay.print_status("info", f"  Primary: {primary_qty} {self.primary_symbol} @ ~{primary_price:.2f}")
                TerminalDisplay.print_status("info", f"  Secondary: {secondary_qty} {self.secondary_symbol} @ ~{secondary_price:.2f}")
                
                if direction == 'long':
                    # Long spread: Buy primary, Sell secondary
                    self.ib.place_market_order(self.primary_contract, "BUY", primary_qty)
                    self.ib.place_market_order(self.secondary_contract, "SELL", secondary_qty)
                    trade_msg = f"Entered LONG spread: Buy {primary_qty} {self.primary_symbol}, Sell {secondary_qty} {self.secondary_symbol}"
                    logger.info(trade_msg)
                    TerminalDisplay.print_status("success", trade_msg)
                else:
                    # Short spread: Sell primary, Buy secondary
                    self.ib.place_market_order(self.primary_contract, "SELL", primary_qty)
                    self.ib.place_market_order(self.secondary_contract, "BUY", secondary_qty)
                    trade_msg = f"Entered SHORT spread: Sell {primary_qty} {self.primary_symbol}, Buy {secondary_qty} {self.secondary_symbol}"
                    logger.info(trade_msg)
                    TerminalDisplay.print_status("success", trade_msg)
                    
                # Update state
                self.position_direction = direction
                self.entry_prices = {
                    self.primary_symbol: primary_price,
                    self.secondary_symbol: secondary_price
                }
                
                # Update positions dict for tracking
                self.positions = {
                    self.primary_symbol: primary_qty if direction == 'long' else -primary_qty,
                    self.secondary_symbol: -secondary_qty if direction == 'long' else secondary_qty
                }
                
                self.trade_count += 1
                
            elif signal['action'] == 'exit':
                # Exit existing position
                exit_msg = f"{self.name}: Executing position exit"
                logger.info(exit_msg)
                TerminalDisplay.print_status("info", exit_msg)
                self.close_positions()
                
                # Calculate P&L
                primary_price = self.price_history[self.primary_symbol]['price'].values[-1]
                secondary_price = self.price_history[self.secondary_symbol]['price'].values[-1]
                
                entry_price_primary = self.entry_prices[self.primary_symbol]
                entry_price_secondary = self.entry_prices[self.secondary_symbol]
                
                # Calculate P&L for each leg
                if self.position_direction == 'long':
                    # Long spread: Bought primary, Sold secondary
                    primary_pl = (primary_price - entry_price_primary) / entry_price_primary
                    secondary_pl = (entry_price_secondary - secondary_price) / entry_price_secondary
                else:
                    # Short spread: Sold primary, Bought secondary
                    primary_pl = (entry_price_primary - primary_price) / entry_price_primary
                    secondary_pl = (secondary_price - entry_price_secondary) / entry_price_secondary
                    
                # Primary has a larger impact due to contract size
                primary_weight = 0.6
                secondary_weight = 0.4
                
                # Total P&L
                total_pl = (primary_pl * primary_weight + secondary_pl * secondary_weight) * self.capital
                
                # Update P&L
                self.update_pl(total_pl)
                pl_msg = f"Exited {self.position_direction} spread with P&L: ${total_pl:.2f}"
                logger.info(pl_msg)
                TerminalDisplay.print_status("success", pl_msg)
                TerminalDisplay.print_status("info", f"  Primary: {primary_pl*100:.2f}%, Secondary: {secondary_pl*100:.2f}%")
                
                # Reset state
                self.position_direction = None
                self.entry_prices = {}
                self.positions = {}
                
                self.trade_count += 1
                
    def close_positions(self):
        if self.position_direction is None:
            logger.info(f"{self.name}: No positions to close")
            return
            
        positions = self.ib.get_positions()
        logger.info(f"{self.name}: Closing positions: {self.positions}")
        TerminalDisplay.print_status("info", f"{self.name}: Closing positions")
        
        # Close primary position
        if self.primary_symbol in positions:
            qty = abs(positions[self.primary_symbol]['position'])
            action = "SELL" if positions[self.primary_symbol]['position'] > 0 else "BUY"
            self.ib.place_market_order(self.primary_contract, action, qty)
            logger.info(f"Closing {self.primary_symbol} position: {action} {qty}")
            TerminalDisplay.print_status("info", f"Closing {self.primary_symbol} position: {action} {qty}")
            
        # Close secondary position
        if self.secondary_symbol in positions:
            qty = abs(positions[self.secondary_symbol]['position'])
            action = "SELL" if positions[self.secondary_symbol]['position'] > 0 else "BUY"
            self.ib.place_market_order(self.secondary_contract, action, qty)
            logger.info(f"Closing {self.secondary_symbol} position: {action} {qty}")
            TerminalDisplay.print_status("info", f"Closing {self.secondary_symbol} position: {action} {qty}")
            
        logger.info(f"Closed all positions for {self.name}")
        TerminalDisplay.print_status("success", f"Closed all positions for {self.name}")
    
    def get_status_summary(self):
        """Get a summary of the strategy status"""
        base_summary = super().get_status_summary()
        
        # Add strategy-specific details
        strategy_summary = {
            "primary_symbol": self.primary_symbol,
            "secondary_symbol": self.secondary_symbol,
            "z_entry": self.z_entry,
            "z_exit": self.z_exit,
            "current_z_score": self.current_z_score,
            "current_spread": self.last_spread,
            "position_direction": self.position_direction,
            "entry_prices": self.entry_prices
        }
        
        return {**base_summary, **strategy_summary}

# ----------------------------------
# PORTFOLIO MANAGEMENT
# ----------------------------------

class Portfolio:
    def __init__(self, initial_capital=2000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategies = {}
        self.strategy_capital = {}
        self.pl_history = []
        self.capital_history = [initial_capital]
        self.timestamps = [datetime.now()]
        
        logger.info(f"Portfolio initialized with ${initial_capital} capital")
        TerminalDisplay.print_status("info", f"Portfolio initialized with ${initial_capital} capital")
        
    def add_strategy(self, strategy):
        """Add a strategy to the portfolio"""
        self.strategies[strategy.name] = strategy
        self.strategy_capital[strategy.name] = strategy.capital
        
        add_msg = f"Added strategy {strategy.name} to portfolio with ${strategy.capital} capital"
        logger.info(add_msg)
        TerminalDisplay.print_status("success", add_msg)
        
    def remove_strategy(self, strategy_name):
        """Remove a strategy from the portfolio"""
        if strategy_name in self.strategies:
            # Close all positions first
            self.strategies[strategy_name].close_positions()
            del self.strategies[strategy_name]
            del self.strategy_capital[strategy_name]
            
            remove_msg = f"Removed strategy {strategy_name} from portfolio"
            logger.info(remove_msg)
            TerminalDisplay.print_status("info", remove_msg)
            
    def allocate_capital(self, allocations):
        """Allocate capital to strategies based on percentage allocations"""
        total_allocation = sum(allocations.values())
        if abs(total_allocation - 100) > 0.01:
            error_msg = f"Total allocation ({total_allocation}%) must sum to 100%"
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            raise ValueError("Total allocation must sum to 100%")
            
        logger.info(f"Allocating capital: {allocations}")
        TerminalDisplay.print_status("info", "Allocating capital across strategies")
            
        # Calculate new capital for each strategy
        for strategy_name, allocation_pct in allocations.items():
            if strategy_name in self.strategies:
                capital = self.current_capital * (allocation_pct / 100)
                self.strategies[strategy_name].update_capital(capital)
                self.strategy_capital[strategy_name] = capital
                
                allocation_msg = f"Allocated ${capital:.2f} ({allocation_pct:.1f}%) to {strategy_name}"
                logger.info(allocation_msg)
                TerminalDisplay.print_status("info", allocation_msg)
                
    def update_pl(self):
        """Update portfolio P&L based on strategies"""
        total_pl = sum(strategy.get_pl() for strategy in self.strategies.values())
        self.current_capital = self.initial_capital + total_pl
        
        self.pl_history.append(total_pl)
        self.capital_history.append(self.current_capital)
        self.timestamps.append(datetime.now())
        
        # Log portfolio update every hour
        if len(self.timestamps) <= 1 or (datetime.now() - self.timestamps[-2]).total_seconds() > 3600:
            strategy_pls = {name: strategy.get_pl() for name, strategy in self.strategies.items()}
            
            update_msg = f"Portfolio update: Capital=${self.current_capital:.2f}, P&L=${total_pl:.2f}"
            logger.info(update_msg)
            
            for name, pl in strategy_pls.items():
                logger.info(f"  {name}: ${pl:.2f}")
        
    def get_strategy_pl(self):
        """Get P&L for each strategy"""
        return {name: strategy.get_pl() for name, strategy in self.strategies.items()}
        
    def optimize_allocation(self, risk_tolerance=0.5):
        """Optimize capital allocation based on forecasted returns"""
        strategy_names = list(self.strategies.keys())
        
        # Get forecasted returns from each strategy
        returns = [self.strategies[name].get_forecast_return() for name in strategy_names]
        
        # Simple risk estimate based on return magnitude
        risks = [max(0.01, abs(r) * 0.5) for r in returns]
        
        logger.info(f"Portfolio optimization: risk_tolerance={risk_tolerance}")
        logger.info(f"Strategy forecasted returns: {dict(zip(strategy_names, returns))}")
        
        TerminalDisplay.print_status("info", f"Optimizing portfolio allocation (risk={risk_tolerance})")
        
        # Define objective function to maximize return
        def objective(weights):
            portfolio_return = sum(w * r for w, r in zip(weights, returns))
            return -portfolio_return  # Negative because we want to maximize
            
        # Define risk constraint
        def risk_constraint(weights):
            portfolio_risk = sum(w * r for w, r in zip(weights, risks))
            return risk_tolerance - portfolio_risk  # Risk must be less than tolerance
            
        # Initial weights - equal allocation
        initial_weights = np.ones(len(strategy_names)) / len(strategy_names)
        
        # Bounds and constraints
        bounds = [(0.1, 0.8) for _ in range(len(strategy_names))]  # Min 10%, max 80% per strategy
        constraints = [
            {'type': 'eq', 'fun': lambda weights: sum(weights) - 1}  # Sum to 1
        ]
        
        # Run optimization
        try:
            result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            
            # Convert to percentages
            allocations = {name: weight * 100 for name, weight in zip(strategy_names, result.x)}
            
            optimize_msg = f"Optimization result: {allocations}"
            logger.info(optimize_msg)
            TerminalDisplay.print_status("success", optimize_msg)
            
            return allocations
        except Exception as e:
            error_msg = f"Portfolio optimization failed: {str(e)}"
            logger.error(error_msg)
            TerminalDisplay.print_status("error", error_msg)
            
            # Return equal allocation as fallback
            return {name: 100 / len(strategy_names) for name in strategy_names}

# ----------------------------------
# MAIN APPLICATION
# ----------------------------------

def display_dashboard(ib_manager, portfolio, main_symbols=None):
    """Display a simple dashboard in the terminal"""
    TerminalDisplay.clear_screen()
    
    # Display header with current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TerminalDisplay.print_header(f"MICRO FUTURES TRADING SYSTEM - {current_time}")
    
    # Connection status
    TerminalDisplay.print_connection_status(
        ib_manager.connected, 
        ib_manager.host, 
        ib_manager.port,
        ib_manager.last_connection_time
    )
    
    # Portfolio status
    TerminalDisplay.print_header("PORTFOLIO STATUS")
    TerminalDisplay.print_portfolio_status(portfolio)
    
    # Current positions
    positions = ib_manager.get_positions()
    TerminalDisplay.print_header("CURRENT POSITIONS")
    TerminalDisplay.print_positions(positions)
    
    # Market data
    TerminalDisplay.print_header("MARKET DATA")
    market_data = ib_manager.get_market_data()
    TerminalDisplay.print_market_data(market_data, main_symbols)
    
    # Recent trades
    TerminalDisplay.print_header("RECENT TRADES")
    trades = ib_manager.get_order_history()
    TerminalDisplay.print_recent_trades(trades)
    
    # Strategy details
    for name, strategy in portfolio.strategies.items():
        TerminalDisplay.print_header(f"STRATEGY: {name}")
        status = strategy.get_status_summary()
        
        # Print strategy status
        status_color = Fore.GREEN if status['active'] else Fore.RED
        print(f"Status: {status_color}{'Active' if status['active'] else 'Inactive'}")
        print(f"Capital: ${status['capital']:.2f}")
        print(f"P&L: ${status['pl']:.2f}")
        print(f"Signals Generated: {status['signals_generated']}")
        print(f"Trades Executed: {status['trades_executed']}")
        
        # Print strategy-specific details
        if 'current_z_score' in status:
            z_score = status['current_z_score']
            z_color = Fore.GREEN
            if z_score:
                if abs(z_score) > status['z_entry']:
                    z_color = Fore.RED
                elif abs(z_score) > status['z_exit']:
                    z_color = Fore.YELLOW
                    
                print(f"Current Z-Score: {z_color}{z_score:.4f}")
            
        if status['position_direction']:
            print(f"Position: {Fore.CYAN}{status['position_direction'].upper()}")
            
            for symbol, price in status['entry_prices'].items():
                print(f"  {symbol} Entry: ${price:.2f}")
    
    # System status
    TerminalDisplay.print_header("SYSTEM STATUS")
    TerminalDisplay.print_system_status(ib_manager)

def run_headless_app(config_file=None, host='127.0.0.1', port=7497, client_id=1, initial_capital=2000, 
                    log_dir='logs', check_interval=5, save_logs_interval=3600, display_interval=15):
    """
    Run the trading application in headless mode with enhanced terminal output
    
    Parameters:
    - config_file: Path to configuration file (optional)
    - host: IB TWS host address
    - port: IB TWS port
    - client_id: IB client ID
    - initial_capital: Initial capital for the portfolio
    - log_dir: Directory to save logs
    - check_interval: How often to check for signals (seconds)
    - save_logs_interval: How often to save logs to files (seconds)
    - display_interval: How often to update the dashboard (seconds)
    """
    TerminalDisplay.clear_screen()
    TerminalDisplay.print_header("MICRO FUTURES TRADING SYSTEM STARTUP")
    TerminalDisplay.print_status("info", "Starting Micro Futures Trading System in headless mode")
    TerminalDisplay.print_status("info", f"Configuration: host={host}, port={port}, client_id={client_id}, capital=${initial_capital}")
    
    logger.info("Starting Micro Futures Trading System in headless mode")
    logger.info(f"Configuration: host={host}, port={port}, client_id={client_id}, capital=${initial_capital}")
    
    # Initialize IB Manager
    ib_manager = IBManager(host=host, port=port, client_id=client_id)
    
    # Initialize Portfolio
    portfolio = Portfolio(initial_capital=initial_capital)
    
    # Create strategies
    strategy_capital = initial_capital / 1  # Just one strategy for now
    
    # Create the strategy
    crude_strategy = MicroCrudeOilArbitrageStrategy(
        name="Micro Crude Oil Arbitrage",
        capital=strategy_capital,
        ib_manager=ib_manager
    )
    
    # Add strategy to portfolio
    portfolio.add_strategy(crude_strategy)
    
    # Connect to IB TWS
    connected = False
    connection_attempts = 0
    
    while not connected and connection_attempts < 5:
        TerminalDisplay.print_status("info", "Attempting to connect to IB TWS...")
        connection_attempts += 1
        connected = ib_manager.connect()
        if not connected:
            TerminalDisplay.print_status("warning", f"Connection failed (attempt {connection_attempts}/5), will retry in 10 seconds")
            time.sleep(10)
    
    if not connected:
        TerminalDisplay.print_status("error", "Failed to connect after multiple attempts. Check that TWS is running and properly configured.")
        logger.error("Failed to connect after multiple attempts")
        return
    
    # Initialize contracts for strategies
    for name, strategy in portfolio.strategies.items():
        success = strategy.initialize_contracts()
        if success:
            TerminalDisplay.print_status("success", f"Contracts initialized for {name}")
        else:
            TerminalDisplay.print_status("error", f"Failed to initialize contracts for {name}")
    
    # Activate strategies
    for name, strategy in portfolio.strategies.items():
        strategy.activate()
        TerminalDisplay.print_status("success", f"Strategy {name} activated")
    
    # Main loop
    last_log_save = datetime.now()
    last_display_update = datetime.now()
    main_symbols = [crude_strategy.primary_symbol, crude_strategy.secondary_symbol]
    
    TerminalDisplay.print_status("success", "Entering main trading loop")
    
    try:
        logger.info("Entering main trading loop")
        
        while True:
            # Check if still connected
            if not ib_manager.connected:
                TerminalDisplay.print_status("warning", "Connection to IB TWS lost, attempting to reconnect")
                connected = ib_manager.connect()
                if not connected:
                    TerminalDisplay.print_status("error", "Failed to reconnect, will retry")
                    time.sleep(10)
                    continue
            
            # Get latest market data
            market_data = ib_manager.get_market_data()
            
            # Check for signals and execute trades for each strategy
            for name, strategy in portfolio.strategies.items():
                if strategy.active:
                    signals = strategy.check_signals(market_data)
                    if signals:
                        TerminalDisplay.print_status("info", f"Strategy {name} generated signals: {signals}")
                        strategy.execute_trades(signals)
            
            # Update portfolio P&L
            portfolio.update_pl()
            
            # Save logs periodically
            if (datetime.now() - last_log_save).total_seconds() > save_logs_interval:
                TerminalDisplay.print_status("info", "Saving logs to files")
                ib_manager.save_logs(directory=log_dir)
                last_log_save = datetime.now()
            
            # Update dashboard display periodically
            if (datetime.now() - last_display_update).total_seconds() > display_interval:
                display_dashboard(ib_manager, portfolio, main_symbols)
                last_display_update = datetime.now()
            
            # Sleep for the check interval
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        TerminalDisplay.print_status("info", "Keyboard interrupt received, shutting down")
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        TerminalDisplay.print_status("error", f"Unexpected error: {str(e)}")
        logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        # Close all positions
        TerminalDisplay.print_status("info", "Closing all positions")
        for name, strategy in portfolio.strategies.items():
            strategy.close_positions()
        
        # Disconnect from IB TWS
        TerminalDisplay.print_status("info", "Disconnecting from IB TWS")
        ib_manager.disconnect()
        
        # Save final logs
        TerminalDisplay.print_status("info", "Saving final logs")
        ib_manager.save_logs(directory=log_dir)
        
        TerminalDisplay.print_status("success", "Trading system shutdown complete")
        logger.info("Trading system shutdown complete")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Micro Futures Trading System')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='IB TWS host address')
    parser.add_argument('--port', type=int, default=7497, help='IB TWS port')
    parser.add_argument('--client-id', type=int, default=1, help='IB client ID')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--capital', type=float, default=2000, help='Initial capital')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--check-interval', type=float, default=5, help='How often to check for signals (seconds)')
    parser.add_argument('--log-interval', type=int, default=3600, help='How often to save logs (seconds)')
    parser.add_argument('--display-interval', type=int, default=15, help='How often to update dashboard (seconds)')
    
    args = parser.parse_args()
    
    # Run the headless application
    run_headless_app(
        config_file=args.config,
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        initial_capital=args.capital,
        log_dir=args.log_dir,
        check_interval=args.check_interval,
        save_logs_interval=args.log_interval,
        display_interval=args.display_interval
    )