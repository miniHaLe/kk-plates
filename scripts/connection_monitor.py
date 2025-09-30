#!/usr/bin/env python3
"""
Connection Monitor for KichiKichi System
Monitors user connections and triggers system restarts for clean sessions
"""

import time
import subprocess
import logging
import threading
import signal
import sys
import os
from datetime import datetime
from typing import Set, Dict

class ConnectionMonitor:
    """
    Monitor user connections and automatically restart the system 
    when users connect/disconnect to ensure clean sessions
    """
    
    def __init__(self, restart_command: str = "make run-sync"):
        self.logger = self._setup_logging()
        self.restart_command = restart_command
        self.running = False
        self.current_process = None
        self.connection_check_interval = 5.0  # Check every 5 seconds
        self.restart_cooldown = 10.0  # 10 seconds between restarts
        self.last_restart_time = 0
        
        # Track dashboard port activity
        self.dashboard_port = 8050
        self.connected_ips: Set[str] = set()
        self.last_connection_time = 0
        
        # 30-minute minimum runtime from first user connection
        self.minimum_runtime_minutes = 30
        self.first_user_connect_time = 0  # When first user connected
        self.minimum_runtime_active = False  # Whether we're in minimum runtime period
        
        self.logger.info("🔄 Connection Monitor initialized")
        self.logger.info(f"📊 Monitoring dashboard on port {self.dashboard_port}")
        self.logger.info(f"🔄 Restart command: {restart_command}")
        self.logger.info(f"⏰ Minimum runtime: {self.minimum_runtime_minutes} minutes from first user connection")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the connection monitor"""
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/connection_monitor.log')
            ]
        )
        return logging.getLogger('ConnectionMonitor')
    
    def get_dashboard_connections(self) -> Set[str]:
        """
        Get current connections to the dashboard port
        
        Returns:
            Set of IP addresses currently connected
        """
        try:
            # Use netstat to find connections to dashboard port
            result = subprocess.run(
                ['netstat', '-tn'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            connected_ips = set()
            for line in result.stdout.split('\n'):
                if f':{self.dashboard_port}' in line and 'ESTABLISHED' in line:
                    # Extract the foreign IP address
                    parts = line.split()
                    if len(parts) >= 5:
                        foreign_addr = parts[4]  # Foreign address column
                        ip = foreign_addr.split(':')[0]
                        # Filter out localhost connections (system internal)
                        if ip not in ['127.0.0.1', '::1', '0.0.0.0']:
                            connected_ips.add(ip)
            
            return connected_ips
            
        except Exception as e:
            self.logger.debug(f"Error checking connections: {e}")
            return set()
    
    def should_restart(self, current_connections: Set[str]) -> bool:
        """
        Determine if system should restart based on connection changes
        Implements 30-minute minimum runtime from first user connection
        
        Args:
            current_connections: Currently connected IP addresses
            
        Returns:
            True if restart is needed
        """
        current_time = time.time()
        
        # Check restart cooldown
        if current_time - self.last_restart_time < self.restart_cooldown:
            return False
        
        # Track first user connection for minimum runtime
        if current_connections and not self.connected_ips and self.first_user_connect_time == 0:
            # First user just connected - start minimum runtime period
            self.first_user_connect_time = current_time
            self.minimum_runtime_active = True
            self.logger.info(f"👤 First user connected - starting {self.minimum_runtime_minutes} minute minimum runtime period")
            from datetime import datetime
            self.logger.info(f"⏰ No restarts allowed until: {datetime.fromtimestamp(current_time + self.minimum_runtime_minutes * 60).strftime('%H:%M:%S')}")
        
        # Check if we're still in minimum runtime period
        if self.minimum_runtime_active and self.first_user_connect_time > 0:
            minimum_runtime_seconds = self.minimum_runtime_minutes * 60
            time_since_first_user = current_time - self.first_user_connect_time
            
            if time_since_first_user < minimum_runtime_seconds:
                # Still in minimum runtime period - no restarts allowed
                remaining_minutes = (minimum_runtime_seconds - time_since_first_user) / 60
                if current_connections != self.connected_ips:
                    self.logger.info(f"⏰ Minimum runtime active - {remaining_minutes:.1f} minutes remaining before restart logic activates")
                return False
            else:
                # Minimum runtime period has passed
                if self.minimum_runtime_active:
                    self.logger.info(f"✅ Minimum runtime period ({self.minimum_runtime_minutes} minutes) completed - restart logic now active")
                    self.minimum_runtime_active = False
        
        # Only allow restarts if minimum runtime period has passed
        if not self.minimum_runtime_active:
            # New user connected - restart for clean session
            new_connections = current_connections - self.connected_ips
            if new_connections and self.connected_ips:  # Don't restart on first connection
                self.logger.info(f"👤 New user(s) connected: {new_connections}")
                return True
            
            # All users disconnected - restart to clean state
            if self.connected_ips and not current_connections:
                self.logger.info("👤 All users disconnected - cleaning state")
                return True
        
        return False
    
    def restart_system(self) -> bool:
        """
        Restart the KichiKichi system
        
        Returns:
            True if restart was successful
        """
        try:
            self.logger.info("🛑 Stopping current system...")
            
            # Stop current process if running
            if self.current_process and self.current_process.poll() is None:
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.current_process.kill()
                    self.current_process.wait()
            
            # Additional cleanup - kill any remaining processes
            subprocess.run(['pkill', '-f', 'main_app.py'], check=False)
            subprocess.run(['pkill', '-f', 'dashboard'], check=False)
            
            # Wait for cleanup
            time.sleep(2)
            
            self.logger.info(f"🔄 Starting fresh system: {self.restart_command}")
            
            # Start new process
            self.current_process = subprocess.Popen(
                self.restart_command.split(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid  # Create new process group
            )
            
            self.last_restart_time = time.time()
            # Reset minimum runtime tracking after restart
            self.first_user_connect_time = 0
            self.minimum_runtime_active = False
            self.logger.info("✅ System restart completed - minimum runtime tracking reset")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error restarting system: {e}")
            return False
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.logger.info("👁️  Starting connection monitoring loop...")
        
        try:
            while self.running:
                current_connections = self.get_dashboard_connections()
                
                # Log connection changes
                if current_connections != self.connected_ips:
                    if current_connections:
                        self.logger.info(f"👤 Active connections: {current_connections}")
                    else:
                        self.logger.info("👤 No active connections")
                    
                    # Update last connection time
                    if current_connections:
                        self.last_connection_time = time.time()
                
                # Check if restart is needed
                if self.should_restart(current_connections):
                    if self.restart_system():
                        # Clear connection tracking after restart
                        self.connected_ips.clear()
                    else:
                        self.logger.error("Failed to restart system")
                
                # Update tracked connections
                self.connected_ips = current_connections
                
                # Wait before next check
                time.sleep(self.connection_check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Connection monitor interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Error in monitoring loop: {e}")
        finally:
            self.cleanup()
    
    def start(self):
        """Start the connection monitor"""
        if self.running:
            self.logger.warning("Monitor is already running")
            return
        
        self.running = True
        
        # Start the initial system
        self.logger.info("🚀 Starting initial KichiKichi system...")
        if not self.restart_system():
            self.logger.error("Failed to start initial system")
            return
        
        # Start monitoring
        self.monitor_loop()
    
    def stop(self):
        """Stop the connection monitor"""
        self.logger.info("🛑 Stopping connection monitor...")
        self.running = False
    
    def cleanup(self):
        """Cleanup when stopping"""
        self.logger.info("🧹 Cleaning up...")
        
        # Stop current process
        if self.current_process and self.current_process.poll() is None:
            self.logger.info("Stopping KichiKichi system...")
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
                self.current_process.wait()
        
        self.logger.info("✅ Cleanup completed")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Received shutdown signal. Stopping monitor...")
    monitor.stop()
    sys.exit(0)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="KichiKichi Connection Monitor")
    parser.add_argument('--restart-command', default='make run-sync',
                       help='Command to restart the system (default: make run-sync)')
    parser.add_argument('--check-interval', type=float, default=5.0,
                       help='Connection check interval in seconds (default: 5.0)')
    parser.add_argument('--restart-cooldown', type=float, default=10.0,
                       help='Minimum time between restarts in seconds (default: 10.0)')
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    global monitor
    monitor = ConnectionMonitor(restart_command=args.restart_command)
    monitor.connection_check_interval = args.check_interval
    monitor.restart_cooldown = args.restart_cooldown
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        monitor.start()
    except Exception as e:
        print(f"❌ Error starting monitor: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
