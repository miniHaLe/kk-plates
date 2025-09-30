"""
Backend caching system for KichiKichi to ensure data accuracy over UI smoothness
Implements comprehensive caching for Stage Summary and Phase Details synchronization
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict
import copy

@dataclass
class CachedStageData:
    """Cached data for a specific stage"""
    stage_id: int
    phase_data: Dict[int, Dict[str, int]] = field(default_factory=dict)  # phase -> dish_counts
    stage_totals: Dict[str, int] = field(default_factory=dict)  # kitchen_total, returned_total
    stage_metrics: Dict[str, int] = field(default_factory=dict)  # taken_out, added_in
    last_updated: datetime = field(default_factory=datetime.now)
    is_complete: bool = False  # True when stage is finished and data is final
    
    def get_total_dishes(self) -> int:
        """Calculate accurate total dishes for this stage"""
        total = 0
        for phase_counts in self.phase_data.values():
            # Only count actual dish types, not 'total' key
            total += sum(count for key, count in phase_counts.items() 
                        if key in ['normal_dish', 'red_dish', 'yellow_dish'] and isinstance(count, int))
        return total


@dataclass 
class CachedPhaseData:
    """Cached data for a specific phase"""
    phase_id: int
    stage_id: int
    dish_counts: Dict[str, int] = field(default_factory=dict)  # normal_dish, red_dish, yellow_dish
    last_updated: datetime = field(default_factory=datetime.now)
    is_validated: bool = False  # True when counts are confirmed accurate
    
    def get_total_dishes(self) -> int:
        """Calculate accurate total dishes for this phase"""
        return sum(count for key, count in self.dish_counts.items() 
                  if key in ['normal_dish', 'red_dish', 'yellow_dish'] and isinstance(count, int))


class BackendCache:
    """
    Backend caching system that prioritizes data accuracy over UI smoothness
    Ensures Stage Summary and Phase Details are synchronized and correct
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.lock = threading.RLock()  # Reentrant lock for nested operations
        
        # Cache storage
        self.stage_cache: Dict[int, CachedStageData] = {}
        self.phase_cache: Dict[int, CachedPhaseData] = {}
        
        # Validation tracking
        self.last_validation: datetime = datetime.now()
        self.validation_interval = timedelta(seconds=1)  # Validate every second for accuracy
        
        # Error tracking
        self.validation_errors: List[str] = []
        self.max_error_history = 50
        
        self.logger.info("🏪 Backend Cache initialized - Prioritizing accuracy over smoothness")
    
    def cache_stage_data(self, stage_id: int, phase_data: Dict[int, Dict[str, int]], 
                        stage_totals: Dict[str, int], stage_metrics: Dict[str, int],
                        is_complete: bool = False) -> None:
        """
        Cache stage data with validation
        
        Args:
            stage_id: Stage identifier
            phase_data: Phase-specific dish counts
            stage_totals: Stage totals (kitchen_total, returned_total)
            stage_metrics: Stage metrics (taken_out, added_in)
            is_complete: Whether this stage is finished
        """
        with self.lock:
            # Validate input data
            validated_phase_data = self._validate_phase_data(phase_data)
            validated_totals = self._validate_totals(stage_totals)
            validated_metrics = self._validate_metrics(stage_metrics)
            
            # Create or update cached stage data
            if stage_id not in self.stage_cache:
                self.stage_cache[stage_id] = CachedStageData(stage_id=stage_id)
            
            stage_cache = self.stage_cache[stage_id]
            stage_cache.phase_data = validated_phase_data.copy()
            stage_cache.stage_totals = validated_totals.copy()
            stage_cache.stage_metrics = validated_metrics.copy()
            stage_cache.last_updated = datetime.now()
            stage_cache.is_complete = is_complete
            
            # Cache individual phases
            for phase_id, dish_counts in validated_phase_data.items():
                self.cache_phase_data(phase_id, stage_id, dish_counts, is_validated=True)
            
            # Log accuracy-focused update
            total_dishes = stage_cache.get_total_dishes()
            # self.logger.info(f"🏪 Cache Updated - Stage {stage_id}: {total_dishes} total dishes, "
            #                f"{len(validated_phase_data)} phases, Complete: {is_complete}")
    
    def cache_phase_data(self, phase_id: int, stage_id: int, dish_counts: Dict[str, int],
                        is_validated: bool = False) -> None:
        """
        Cache phase data with validation
        
        Args:
            phase_id: Phase identifier  
            stage_id: Parent stage identifier
            dish_counts: Dish counts by type
            is_validated: Whether data has been validated
        """
        with self.lock:
            # Validate dish counts
            validated_counts = self._validate_dish_counts(dish_counts)
            
            # Create or update cached phase data
            if phase_id not in self.phase_cache:
                self.phase_cache[phase_id] = CachedPhaseData(phase_id=phase_id, stage_id=stage_id)
            
            phase_cache = self.phase_cache[phase_id]
            phase_cache.dish_counts = validated_counts.copy()
            phase_cache.last_updated = datetime.now()
            phase_cache.is_validated = is_validated
            
            total_dishes = phase_cache.get_total_dishes()
            # self.logger.debug(f"🏪 Phase Cache Updated - Phase {phase_id}: {total_dishes} dishes, "
            #                 f"Validated: {is_validated}")
    
    def get_stage_summary(self, stage_id: int) -> Optional[Dict[str, Any]]:
        """
        Get cached stage summary with accuracy validation
        
        Args:
            stage_id: Stage identifier
            
        Returns:
            Stage summary data or None if not cached/invalid
        """
        with self.lock:
            if stage_id not in self.stage_cache:
                self.logger.warning(f"🏪 Stage {stage_id} not found in cache")
                return None
            
            stage_data = self.stage_cache[stage_id]
            
            # Validate cache freshness (accuracy over smoothness)
            if not self._is_cache_valid(stage_data.last_updated):
                self.logger.warning(f"🏪 Stage {stage_id} cache is stale, returning None for accuracy")
                return None
            
            # Build accurate summary
            total_dishes = stage_data.get_total_dishes()
            phase_count = len(stage_data.phase_data)
            
            summary = {
                'stage_id': stage_id,
                'total_dishes': total_dishes,
                'phase_count': phase_count,
                'phase_data': copy.deepcopy(stage_data.phase_data),
                'stage_totals': copy.deepcopy(stage_data.stage_totals),
                'stage_metrics': copy.deepcopy(stage_data.stage_metrics),
                'is_complete': stage_data.is_complete,
                'last_updated': stage_data.last_updated,
                'cache_status': 'valid'
            }
            
            self.logger.debug(f"🏪 Stage Summary Retrieved - Stage {stage_id}: {total_dishes} dishes")
            return summary
    
    def get_phase_details(self, phase_id: int) -> Optional[Dict[str, Any]]:
        """
        Get cached phase details with accuracy validation
        
        Args:
            phase_id: Phase identifier
            
        Returns:
            Phase details data or None if not cached/invalid
        """
        with self.lock:
            if phase_id not in self.phase_cache:
                # self.logger.warning(f"🏪 Phase {phase_id} not found in cache")
                return None
            
            phase_data = self.phase_cache[phase_id]
            
            # Validate cache freshness
            if not self._is_cache_valid(phase_data.last_updated):
                # self.logger.warning(f"🏪 Phase {phase_id} cache is stale, returning None for accuracy")
                return None
            
            # Build accurate details
            total_dishes = phase_data.get_total_dishes()
            
            details = {
                'phase_id': phase_id,
                'stage_id': phase_data.stage_id,
                'dish_counts': copy.deepcopy(phase_data.dish_counts),
                'total_dishes': total_dishes,
                'is_validated': phase_data.is_validated,
                'last_updated': phase_data.last_updated,
                'cache_status': 'valid'
            }
            
            # self.logger.debug(f"🏪 Phase Details Retrieved - Phase {phase_id}: {total_dishes} dishes")
            return details
    
    def get_all_cached_stages(self) -> Dict[int, Dict[str, Any]]:
        """Get all cached stage summaries for dashboard display"""
        with self.lock:
            result = {}
            for stage_id in self.stage_cache:
                summary = self.get_stage_summary(stage_id)
                if summary:  # Only include valid cached data
                    result[stage_id] = summary
            
            self.logger.debug(f"🏪 Retrieved {len(result)} valid cached stages")
            return result
    
    def get_all_cached_phases(self) -> Dict[int, Dict[str, Any]]:
        """Get all cached phase details for dashboard display"""
        with self.lock:
            result = {}
            for phase_id in self.phase_cache:
                details = self.get_phase_details(phase_id)
                if details:  # Only include valid cached data
                    result[phase_id] = details
            
            self.logger.debug(f"🏪 Retrieved {len(result)} valid cached phases")
            return result
    
    def validate_cache_consistency(self) -> bool:
        """
        Validate cache consistency for accuracy
        
        Returns:
            True if cache is consistent, False otherwise
        """
        with self.lock:
            errors = []
            
            # Validate each stage's internal consistency
            for stage_id, stage_data in self.stage_cache.items():
                # Check that phase totals match stage total
                stage_total_calculated = stage_data.get_total_dishes()
                
                # Validate phase data consistency
                for phase_id, phase_counts in stage_data.phase_data.items():
                    if phase_id in self.phase_cache:
                        cached_phase = self.phase_cache[phase_id]
                        if cached_phase.dish_counts != phase_counts:
                            errors.append(f"Phase {phase_id} inconsistency between stage and phase cache")
            
            # Update error tracking
            self.validation_errors.extend(errors)
            if len(self.validation_errors) > self.max_error_history:
                self.validation_errors = self.validation_errors[-self.max_error_history:]
            
            is_consistent = len(errors) == 0
            
            if not is_consistent:
                self.logger.error(f"🏪 Cache Validation Failed: {len(errors)} errors found")
                for error in errors:
                    self.logger.error(f"   - {error}")
            else:
                self.logger.debug("🏪 Cache Validation Passed - All data consistent")
            
            return is_consistent
    
    def _validate_phase_data(self, phase_data: Dict[int, Dict[str, int]]) -> Dict[int, Dict[str, int]]:
        """Validate and clean phase data"""
        validated = {}
        for phase_id, dish_counts in phase_data.items():
            if isinstance(phase_id, int) and isinstance(dish_counts, dict):
                validated[phase_id] = self._validate_dish_counts(dish_counts)
        return validated
    
    def _validate_dish_counts(self, dish_counts: Dict[str, int]) -> Dict[str, int]:
        """Validate and clean dish counts, ensuring no double-counting"""
        validated = {}
        valid_dish_types = ['normal_dish', 'red_dish', 'yellow_dish', 'advertisement_dish']
        
        for dish_type, count in dish_counts.items():
            if dish_type in valid_dish_types and isinstance(count, int) and count >= 0:
                validated[dish_type] = count
            elif dish_type == 'total':
                # Skip 'total' key to prevent double-counting - we calculate this ourselves
                continue
        
        return validated
    
    def _validate_totals(self, totals: Dict[str, int]) -> Dict[str, int]:
        """Validate stage totals"""
        validated = {}
        valid_keys = ['kitchen_total', 'returned_total']
        
        for key, value in totals.items():
            if key in valid_keys and isinstance(value, int) and value >= 0:
                validated[key] = value
        
        return validated
    
    def _validate_metrics(self, metrics: Dict[str, int]) -> Dict[str, int]:
        """Validate stage metrics"""
        validated = {}
        valid_keys = ['taken_out', 'added_in']
        
        for key, value in metrics.items():
            if key in valid_keys and isinstance(value, int) and value >= 0:
                validated[key] = value
        
        return validated
    
    def _is_cache_valid(self, last_updated: datetime) -> bool:
        """Check if cache data is still valid (accuracy-focused)"""
        return (datetime.now() - last_updated) < self.validation_interval
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        with self.lock:
            self.stage_cache.clear()
            self.phase_cache.clear()
            self.validation_errors.clear()
            self.logger.info("🏪 Cache cleared for fresh start")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status for monitoring"""
        with self.lock:
            return {
                'cached_stages': len(self.stage_cache),
                'cached_phases': len(self.phase_cache),
                'last_validation': self.last_validation,
                'validation_errors': len(self.validation_errors),
                'recent_errors': self.validation_errors[-5:] if self.validation_errors else []
            }
