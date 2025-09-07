"""Power BI REST API sink for metrics and alerts."""

from typing import Dict, List, Optional
import time
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import structlog

logger = structlog.get_logger()


class PowerBISink:
    """Send metrics and alerts to Power BI streaming dataset."""
    
    def __init__(self, endpoint: str, api_key: str, timeout: int = 5):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        
        # Setup session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Add auth header
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        
        # Track send statistics
        self.total_sent = 0
        self.total_failed = 0
        self.last_send_time = 0
        
    def send_metrics(self, metrics: Dict) -> bool:
        """
        Send metrics snapshot to Power BI.
        
        Args:
            metrics: Metrics dictionary from MetricSnapshot.to_dict()
            
        Returns:
            True if successful
        """
        # Transform to Power BI format
        payload = [{
            "timestamp": metrics["timestamp"],
            "datetime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(metrics["timestamp"])),
            "total_in": metrics["total_in"],
            "total_out": metrics["total_out"],
            "current_on_belt": metrics["current_on_belt"],
            "plates_per_minute": metrics["plates_per_minute"],
            
            # Flatten color data
            "red_count": metrics["color_counts"]["red"],
            "yellow_count": metrics["color_counts"]["yellow"],
            "normal_count": metrics["color_counts"]["normal"],
            
            "red_frequency": metrics["color_frequencies"]["red"],
            "yellow_frequency": metrics["color_frequencies"]["yellow"],
            "normal_frequency": metrics["color_frequencies"]["normal"],
            
            "red_ratio": metrics["color_ratios"]["red"],
            "yellow_ratio": metrics["color_ratios"]["yellow"],
            "normal_ratio": metrics["color_ratios"]["normal"],
            
            "metric_type": "snapshot"
        }]
        
        return self._send_data(payload, "metrics")
    
    def send_alert(self, alert: Dict) -> bool:
        """
        Send alert to Power BI.
        
        Args:
            alert: Alert dictionary from Alert.to_dict()
            
        Returns:
            True if successful
        """
        # Transform to Power BI format
        payload = [{
            "timestamp": alert["timestamp"],
            "datetime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(alert["timestamp"])),
            "alert_type": alert["alert_type"],
            "severity": alert["severity"],
            "message": alert["message"],
            "color": alert["details"].get("color", ""),
            "target_ratio": alert["details"].get("target_ratio", 0),
            "actual_ratio": alert["details"].get("actual_ratio", 0),
            "deviation": alert["details"].get("relative_deviation", 0),
            "metric_type": "alert"
        }]
        
        return self._send_data(payload, "alert")
    
    def send_event(self, event: Dict) -> bool:
        """
        Send crossing event to Power BI.
        
        Args:
            event: Crossing event data
            
        Returns:
            True if successful
        """
        # Transform to Power BI format
        payload = [{
            "timestamp": event["timestamp"],
            "datetime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(event["timestamp"])),
            "track_id": event["track_id"],
            "direction": event["direction"],
            "color": event["color"],
            "position_x": event["position"][0],
            "position_y": event["position"][1],
            "metric_type": "event"
        }]
        
        return self._send_data(payload, "event")
    
    def _send_data(self, payload: List[Dict], data_type: str) -> bool:
        """Send data to Power BI endpoint."""
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_send_time < 0.1:  # Max 10 req/sec
                time.sleep(0.1)
            
            response = self.session.post(
                self.endpoint,
                data=json.dumps(payload),
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                self.total_sent += 1
                self.last_send_time = current_time
                logger.debug(f"Sent {data_type} to Power BI", 
                           status=response.status_code,
                           size=len(payload))
                return True
            else:
                self.total_failed += 1
                logger.warning(f"Failed to send {data_type} to Power BI",
                             status=response.status_code,
                             response=response.text[:200])
                return False
                
        except requests.exceptions.RequestException as e:
            self.total_failed += 1
            logger.error(f"Power BI send error for {data_type}", error=str(e))
            return False
        except Exception as e:
            self.total_failed += 1
            logger.error(f"Unexpected error sending {data_type}", error=str(e))
            return False
    
    def batch_send(self, items: List[Dict], item_type: str) -> int:
        """
        Send multiple items in batches.
        
        Args:
            items: List of items to send
            item_type: Type of items ("metrics", "alert", "event")
            
        Returns:
            Number of successfully sent items
        """
        if not items:
            return 0
        
        # Power BI typically supports up to 10,000 rows per request
        batch_size = 1000
        sent_count = 0
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Transform based on type
            if item_type == "metrics":
                success = all(self.send_metrics(item) for item in batch)
            elif item_type == "alert":
                success = all(self.send_alert(item) for item in batch)
            elif item_type == "event":
                success = all(self.send_event(item) for item in batch)
            else:
                logger.error(f"Unknown item type: {item_type}")
                continue
            
            if success:
                sent_count += len(batch)
        
        return sent_count
    
    def get_stats(self) -> Dict:
        """Get sink statistics."""
        return {
            "endpoint": self.endpoint,
            "total_sent": self.total_sent,
            "total_failed": self.total_failed,
            "success_rate": self.total_sent / (self.total_sent + self.total_failed) 
                           if (self.total_sent + self.total_failed) > 0 else 0,
            "last_send_time": self.last_send_time
        }
    
    def test_connection(self) -> bool:
        """Test connection to Power BI endpoint."""
        test_payload = [{
            "timestamp": time.time(),
            "datetime": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "test": True,
            "metric_type": "test"
        }]
        
        logger.info("Testing Power BI connection", endpoint=self.endpoint)
        success = self._send_data(test_payload, "test")
        
        if success:
            logger.info("Power BI connection test successful")
        else:
            logger.error("Power BI connection test failed")
        
        return success