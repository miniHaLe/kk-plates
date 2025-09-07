"""Preset checking and alert generation."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time
import structlog

logger = structlog.get_logger()


@dataclass
class Alert:
    """Alert for preset violation."""
    timestamp: float
    alert_type: str  # "color_ratio_deviation"
    severity: str  # "warning" or "critical"
    message: str
    details: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "details": self.details
        }


class PresetChecker:
    """Check metrics against preset targets and generate alerts."""
    
    def __init__(self, 
                 target_ratios: Dict[str, float],
                 tolerance: float = 0.20,
                 violation_threshold: int = 2,
                 check_interval_seconds: int = 60):
        """
        Initialize preset checker.
        
        Args:
            target_ratios: Target color ratios (must sum to 1.0)
            tolerance: Relative tolerance (e.g., 0.20 = ±20%)
            violation_threshold: Number of consecutive violations before alert
            check_interval_seconds: How often to check
        """
        self.target_ratios = target_ratios
        self.tolerance = tolerance
        self.violation_threshold = violation_threshold
        self.check_interval_seconds = check_interval_seconds
        
        # Validate target ratios
        total = sum(target_ratios.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Target ratios must sum to 1.0, got {total}")
        
        # Track violations
        self.violation_counts = {"red": 0, "yellow": 0, "normal": 0}
        self.last_check_time = 0
        self.check_history: deque = deque(maxlen=10)
        
        # Active alerts
        self.active_alerts: List[Alert] = []
        
    def check(self, snapshot: Dict, force: bool = False) -> List[Alert]:
        """
        Check snapshot against presets.
        
        Args:
            snapshot: Metrics snapshot dict
            force: Force check even if recently checked
            
        Returns:
            List of new alerts
        """
        current_time = time.time()
        
        # Rate limit checks
        if not force and current_time - self.last_check_time < self.check_interval_seconds:
            return []
        
        self.last_check_time = current_time
        new_alerts = []
        
        # Get actual ratios
        actual_ratios = snapshot.get("color_ratios", {})
        
        # Skip if no plates detected
        if sum(actual_ratios.values()) == 0:
            logger.debug("No plates in window, skipping preset check")
            return []
        
        # Check each color
        violations = {}
        for color in ["red", "yellow", "normal"]:
            target = self.target_ratios.get(color, 0)
            actual = actual_ratios.get(color, 0)
            
            # Calculate deviation
            if target > 0:
                relative_deviation = abs(actual - target) / target
            else:
                relative_deviation = 1.0 if actual > 0 else 0
            
            # Check if violating tolerance
            if relative_deviation > self.tolerance:
                violations[color] = {
                    "target": target,
                    "actual": actual,
                    "deviation": relative_deviation
                }
                self.violation_counts[color] += 1
            else:
                # Reset violation count if back in tolerance
                self.violation_counts[color] = 0
        
        # Record check
        self.check_history.append({
            "timestamp": current_time,
            "violations": violations,
            "actual_ratios": actual_ratios
        })
        
        # Generate alerts for persistent violations
        for color, violation_count in self.violation_counts.items():
            if violation_count >= self.violation_threshold and color in violations:
                # Check if we already have an active alert for this
                has_active = any(
                    a.details.get("color") == color 
                    for a in self.active_alerts
                    if current_time - a.timestamp < 300  # 5 min
                )
                
                if not has_active:
                    violation = violations[color]
                    alert = Alert(
                        timestamp=current_time,
                        alert_type="color_ratio_deviation",
                        severity="critical" if violation_count >= self.violation_threshold * 2 else "warning",
                        message=f"{color.capitalize()} plate ratio deviation: "
                               f"{violation['actual']:.1%} (target: {violation['target']:.1%})",
                        details={
                            "color": color,
                            "target_ratio": violation["target"],
                            "actual_ratio": violation["actual"],
                            "relative_deviation": violation["deviation"],
                            "consecutive_violations": violation_count,
                            "snapshot": snapshot
                        }
                    )
                    
                    new_alerts.append(alert)
                    self.active_alerts.append(alert)
                    
                    logger.warning("Preset violation alert generated",
                                 color=color,
                                 target=violation["target"],
                                 actual=violation["actual"],
                                 violations=violation_count)
        
        # Clean old alerts
        self.active_alerts = [
            a for a in self.active_alerts 
            if current_time - a.timestamp < 3600  # Keep for 1 hour
        ]
        
        return new_alerts
    
    def get_status(self) -> Dict:
        """Get current preset check status."""
        return {
            "target_ratios": self.target_ratios,
            "tolerance": self.tolerance,
            "violation_counts": self.violation_counts.copy(),
            "active_alerts": len(self.active_alerts),
            "last_check": self.last_check_time,
            "recent_checks": list(self.check_history)[-5:]  # Last 5 checks
        }
    
    def reset_violations(self) -> None:
        """Reset violation counters."""
        self.violation_counts = {"red": 0, "yellow": 0, "normal": 0}
        logger.info("Violation counters reset")
    
    def update_targets(self, new_ratios: Dict[str, float]) -> None:
        """Update target ratios."""
        total = sum(new_ratios.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Target ratios must sum to 1.0, got {total}")
        
        self.target_ratios = new_ratios
        self.reset_violations()
        logger.info("Target ratios updated", ratios=new_ratios)