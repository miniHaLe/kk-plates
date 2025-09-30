#!/usr/bin/env python3
"""
Dashboard Watcher for KichiKichi System
Monitors the 'python run.py --mode dashboard' command and handles user disconnections
"""

import time
import subprocess
import logging
import signal
import sys
import os
import threading
from datetime import datetime
from typing import Set, Optional
import psutil

class DashboardWatcher:
    """
    Watches the dashboard process and automatically restarts it when users disconnect
    to ensure clean sessions for each user connection
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        # Use bash to run the start_kichi.sh script which handles environment setup
        self.dashboard_command = "bash start_kichi.sh"
        self.running = False
        self.current_process: Optional[subprocess.Popen] = None
        self.connection_check_interval = 3.0  # Check every 3 seconds
        self.restart_cooldown = 5.0  # 5 seconds between restarts
        self.last_restart_time = 0
        
        # Track dashboard port activity
        self.dashboard_port = 8050
        self.connected_ips: Set[str] = set()
        self.last_connection_time = 0
        self.connection_lost_time = 0
        self.disconnect_grace_period = 10.0  # Wait 10 seconds after all users disconnect before restart
        
        # 30-minute minimum runtime from first user connection
        self.minimum_runtime_minutes = 30
        self.first_user_connect_time = 0  # When first user connected
        self.minimum_runtime_active = False  # Whether we're in minimum runtime period
        
        self.logger.info("🎯 Dashboard Watcher initialized")
        self.logger.info(f"📊 Monitoring dashboard on port {self.dashboard_port}")
        self.logger.info(f"🔄 Dashboard command: {self.dashboard_command}")
        self.logger.info("📝 Note: Using start_kichi.sh for proper environment setup")
        self.logger.info(f"⏰ Minimum runtime: {self.minimum_runtime_minutes} minutes from first user connection")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the dashboard watcher"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/dashboard_watcher.log')
            ]
        )
        return logging.getLogger('DashboardWatcher')
    
    def get_dashboard_connections(self) -> Set[str]:
        """
        Get current connections to the dashboard port using psutil for better accuracy
        
        Returns:
            Set of IP addresses currently connected
        """
        try:
            connected_ips = set()
            
            # Get all network connections
            connections = psutil.net_connections(kind='inet')
            
            for conn in connections:
                # Look for connections to our dashboard port
                if (conn.laddr and conn.laddr.port == self.dashboard_port and 
                    conn.status == psutil.CONN_ESTABLISHED and conn.raddr):
                    
                    remote_ip = conn.raddr.ip
                    # Filter out localhost connections (system internal)
                    if remote_ip not in ['127.0.0.1', '::1', '0.0.0.0', 'localhost']:
                        connected_ips.add(remote_ip)
            
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
        
        # Update connection tracking
        if current_connections != self.connected_ips:
            if current_connections:
                self.last_connection_time = current_time
                # Reset disconnection timer if users are connected
                self.connection_lost_time = 0
            elif self.connected_ips and not current_connections:
                # Users just disconnected
                if self.connection_lost_time == 0:
                    self.connection_lost_time = current_time
                    self.logger.info(f"👤 All users disconnected - starting grace period ({self.disconnect_grace_period}s)")
        
        # Only allow restarts if minimum runtime period has passed
        if not self.minimum_runtime_active:
            # Check if grace period has passed after all users disconnected
            if (self.connection_lost_time > 0 and 
                current_time - self.connection_lost_time >= self.disconnect_grace_period and
                not current_connections):
                self.logger.info("⏰ Grace period expired - restarting for clean state")
                return True
            
            # New user connected while others were already connected - restart for clean session
            new_connections = current_connections - self.connected_ips
            if new_connections and self.connected_ips:
                self.logger.info(f"👤 New user(s) connected: {new_connections} - restarting for clean session")
                return True
        
        return False
    
    def start_dashboard(self) -> bool:
        """
        Start the dashboard process
        
        Returns:
            True if started successfully
        """
        try:
            if self.current_process and self.current_process.poll() is None:
                self.logger.warning("Dashboard process already running")
                return True
            
            self.logger.info(f"🚀 Starting dashboard: {self.dashboard_command}")
            
            # Change to the correct directory
            os.chdir('/home/hale/hale')
            
            # Start the dashboard process
            self.current_process = subprocess.Popen(
                self.dashboard_command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,  # Create new process group
                cwd='/home/hale/hale'
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            # Check if process is still running
            if self.current_process.poll() is None:
                self.logger.info("✅ Dashboard started successfully")
                self.logger.info(f"🌐 Dashboard should be available at: http://localhost:{self.dashboard_port}")
                return True
            else:
                # Process died immediately - capture error
                stdout, stderr = self.current_process.communicate()
                self.logger.error(f"❌ Dashboard process died immediately")
                self.logger.error(f"STDOUT: {stdout.decode()}")
                self.logger.error(f"STDERR: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error starting dashboard: {e}")
            return False
    
    def stop_dashboard(self) -> bool:
        """
        Stop the current dashboard process
        
        Returns:
            True if stopped successfully
        """
        try:
            if self.current_process is None:
                return True
            
            if self.current_process.poll() is not None:
                # Process already terminated
                return True
            
            self.logger.info("🛑 Stopping dashboard process...")
            
            # First try graceful termination
            self.current_process.terminate()
            
            try:
                # Wait for graceful shutdown
                self.current_process.wait(timeout=5)
                self.logger.info("✅ Dashboard stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                self.logger.warning("⚠️ Graceful shutdown failed, force killing...")
                self.current_process.kill()
                self.current_process.wait()
                self.logger.info("✅ Dashboard force stopped")
            
            # Additional cleanup - kill any remaining dashboard processes
            try:
                subprocess.run(['pkill', '-f', 'run.py.*dashboard'], check=False, timeout=5)
                subprocess.run(['pkill', '-f', 'main_app.py'], check=False, timeout=5)
            except subprocess.TimeoutExpired:
                pass
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping dashboard: {e}")
            return False
    
    def restart_dashboard(self) -> bool:
        """
        Restart the dashboard process
        
        Returns:
            True if restart was successful
        """
        try:
            self.logger.info("🔄 Restarting dashboard...")
            
            # Stop current process
            if not self.stop_dashboard():
                self.logger.error("Failed to stop dashboard")
                return False
            
            # Wait a moment for cleanup
            time.sleep(2)
            
            # Start new process
            if not self.start_dashboard():
                self.logger.error("Failed to start dashboard")
                return False
            
            self.last_restart_time = time.time()
            self.connection_lost_time = 0  # Reset disconnection timer
            # Reset minimum runtime tracking after restart
            self.first_user_connect_time = 0
            self.minimum_runtime_active = False
            self.logger.info("✅ Dashboard restart completed - minimum runtime tracking reset")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error restarting dashboard: {e}")
            return False
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.logger.info("👁️ Starting dashboard monitoring loop...")
        
        try:
            while self.running:
                current_connections = self.get_dashboard_connections()
                
                # Log connection changes
                if current_connections != self.connected_ips:
                    if current_connections:
                        self.logger.info(f"👤 Active connections: {current_connections}")
                    else:
                        self.logger.info("👤 No active connections")
                
                # Check if restart is needed
                if self.should_restart(current_connections):
                    if self.restart_dashboard():
                        # Clear connection tracking after restart
                        self.connected_ips.clear()
                    else:
                        self.logger.error("Failed to restart dashboard")
                
                # Update tracked connections
                self.connected_ips = current_connections
                
                # Check if dashboard process is still running
                if self.current_process and self.current_process.poll() is not None:
                    self.logger.warning("⚠️ Dashboard process died unexpectedly, restarting...")
                    if not self.restart_dashboard():
                        self.logger.error("Failed to restart dead dashboard process")
                        break
                
                # Wait before next check
                time.sleep(self.connection_check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Dashboard watcher interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Error in monitoring loop: {e}")
        finally:
            self.cleanup()
    
    def start(self):
        """Start the dashboard watcher"""
        if self.running:
            self.logger.warning("Watcher is already running")
            return
        
        self.running = True
        
        # Start the initial dashboard
        self.logger.info("🚀 Starting initial dashboard...")
        if not self.start_dashboard():
            self.logger.error("Failed to start initial dashboard")
            return
        
        # Start monitoring
        self.monitor_loop()
    
    def stop(self):
        """Stop the dashboard watcher"""
        self.logger.info("🛑 Stopping dashboard watcher...")
        self.running = False
    
    def cleanup(self):
        """Cleanup when stopping"""
        self.logger.info("🧹 Cleaning up...")
        
        # Stop dashboard process
        self.stop_dashboard()
        
        self.logger.info("✅ Cleanup completed")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Received shutdown signal. Stopping dashboard watcher...")
    if 'watcher' in globals():
        watcher.stop()
    sys.exit(0)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="KichiKichi Dashboard Watcher")
    parser.add_argument('--check-interval', type=float, default=3.0,
                       help='Connection check interval in seconds (default: 3.0)')
    parser.add_argument('--restart-cooldown', type=float, default=5.0,
                       help='Minimum time between restarts in seconds (default: 5.0)')
    parser.add_argument('--grace-period', type=float, default=10.0,
                       help='Time to wait after disconnect before restart in seconds (default: 10.0)')
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    global watcher
    watcher = DashboardWatcher()
    watcher.connection_check_interval = args.check_interval
    watcher.restart_cooldown = args.restart_cooldown
    watcher.disconnect_grace_period = args.grace_period
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("🎯 KichiKichi Dashboard Watcher")
        print("=" * 40)
        print(f"📊 Monitoring: python run.py --mode dashboard")
        print(f"🌐 Dashboard URL: http://localhost:8050")
        print(f"⏱️  Check interval: {args.check_interval}s")
        print(f"⏰ Grace period: {args.grace_period}s")
        print(f"🔄 Restart cooldown: {args.restart_cooldown}s")
        print("=" * 40)
        print("🚀 Starting watcher...")
        print()
        
        watcher.start()
    except Exception as e:
        print(f"❌ Error starting dashboard watcher: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
