"""Object tracking using ByteTrack algorithm."""

from typing import List, Tuple, Dict, Optional
import numpy as np
from collections import defaultdict
import lap
import structlog

logger = structlog.get_logger()


class Track:
    """Single object track."""
    
    track_id_counter = 0
    
    def __init__(self, bbox: np.ndarray, score: float):
        self.track_id = Track.track_id_counter
        Track.track_id_counter += 1
        
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.score = score
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        
        # Kalman filter state (simplified - just position and velocity)
        cx, cy = self._get_center()
        self.mean = np.array([cx, cy, 0, 0])  # [cx, cy, vx, vy]
        self.covariance = np.eye(4) * 10
        
        # Track history for crossing detection
        self.history: List[Tuple[float, float]] = [(cx, cy)]
        self.max_history = 30
        
    def _get_center(self) -> Tuple[float, float]:
        """Get center point of bbox."""
        return (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2
    
    def predict(self) -> None:
        """Predict next state using simple motion model."""
        # Update position with velocity
        self.mean[0] += self.mean[2]
        self.mean[1] += self.mean[3]
        
        # Increase uncertainty
        self.covariance *= 1.1
        
        # Update bbox based on predicted center
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        self.bbox = np.array([
            self.mean[0] - w/2,
            self.mean[1] - h/2,
            self.mean[0] + w/2,
            self.mean[1] + h/2
        ])
        
        self.age += 1
        self.time_since_update += 1
    
    def update(self, bbox: np.ndarray, score: float) -> None:
        """Update track with new detection."""
        # Update bbox
        self.bbox = bbox
        self.score = score
        
        # Update Kalman filter
        cx, cy = self._get_center()
        
        # Simple velocity estimation
        if self.hits > 0:
            self.mean[2] = 0.8 * self.mean[2] + 0.2 * (cx - self.mean[0])
            self.mean[3] = 0.8 * self.mean[3] + 0.2 * (cy - self.mean[1])
        
        self.mean[0] = cx
        self.mean[1] = cy
        
        # Reduce uncertainty
        self.covariance *= 0.8
        
        # Update history
        self.history.append((cx, cy))
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        self.hits += 1
        self.time_since_update = 0
    
    @property
    def center(self) -> Tuple[float, float]:
        """Current center position."""
        return self.mean[0], self.mean[1]


class ByteTracker:
    """ByteTrack multi-object tracker."""
    
    def __init__(self, 
                 track_thresh: float = 0.5,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []
        
        self.frame_id = 0
        
    def update(self, detections: List[Tuple[int, int, int, int, float, int]]) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (x1, y1, x2, y2, score, class_id)
            
        Returns:
            List of active tracks
        """
        self.frame_id += 1
        
        # Convert detections to numpy arrays
        if len(detections) == 0:
            dets = np.empty((0, 5))
        else:
            dets = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections])
        
        # Split detections by score
        remain_inds = dets[:, 4] > self.track_thresh
        inds_low = dets[:, 4] <= self.track_thresh
        inds_high = np.logical_not(inds_low)
        
        dets_high = dets[inds_high]
        dets_low = dets[inds_low]
        
        # Predict existing tracks
        for track in self.tracked_tracks:
            track.predict()
        
        # Associate high score detections with tracks
        matched, unmatched_dets, unmatched_trks = self._associate(
            dets_high, self.tracked_tracks, self.match_thresh
        )
        
        # Update matched tracks
        for m in matched:
            self.tracked_tracks[m[1]].update(dets_high[m[0], :4], dets_high[m[0], 4])
        
        # Process unmatched tracks
        for i in unmatched_trks:
            track = self.tracked_tracks[i]
            if track.time_since_update < self.track_buffer:
                self.lost_tracks.append(track)
            else:
                self.removed_tracks.append(track)
        
        # Match remaining detections with lost tracks
        self.tracked_tracks = [self.tracked_tracks[i] for i in range(len(self.tracked_tracks)) 
                              if i not in unmatched_trks]
        
        # Second association with low score detections
        if len(dets_low) > 0 and len(self.lost_tracks) > 0:
            matched, unmatched_dets_low, unmatched_lost = self._associate(
                dets_low, self.lost_tracks, 0.5
            )
            
            for m in matched:
                track = self.lost_tracks[m[1]]
                track.update(dets_low[m[0], :4], dets_low[m[0], 4])
                self.tracked_tracks.append(track)
            
            self.lost_tracks = [self.lost_tracks[i] for i in unmatched_lost]
        
        # Initialize new tracks from unmatched high detections
        for i in unmatched_dets:
            track = Track(dets_high[i, :4], dets_high[i, 4])
            self.tracked_tracks.append(track)
        
        # Remove old lost tracks
        self.lost_tracks = [t for t in self.lost_tracks 
                           if t.time_since_update < self.track_buffer]
        
        # Output active tracks
        output_tracks = [track for track in self.tracked_tracks if track.hits >= 2]
        
        logger.debug("Tracker update", 
                    n_detections=len(detections),
                    n_tracks=len(output_tracks),
                    n_lost=len(self.lost_tracks))
        
        return output_tracks
    
    def _associate(self, detections: np.ndarray, tracks: List[Track], 
                   thresh: float) -> Tuple[np.ndarray, List[int], List[int]]:
        """Associate detections to tracks using IoU."""
        if len(detections) == 0 or len(tracks) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), list(range(len(tracks)))
        
        # Compute IoU matrix
        track_bboxes = np.array([t.bbox for t in tracks])
        iou_matrix = self._compute_iou_matrix(detections[:, :4], track_bboxes)
        
        # Solve assignment problem
        cost_matrix = 1 - iou_matrix
        cost_matrix[iou_matrix < thresh] = 1e5
        
        try:
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            matches = [[x[i], i] for i in range(len(x)) if x[i] >= 0]
            matches = np.array(matches)
        except:
            matches = np.empty((0, 2), dtype=int)
        
        unmatched_dets = []
        for d in range(len(detections)):
            if len(matches) == 0 or d not in matches[:, 0]:
                unmatched_dets.append(d)
        
        unmatched_trks = []
        for t in range(len(tracks)):
            if len(matches) == 0 or t not in matches[:, 1]:
                unmatched_trks.append(t)
        
        # Filter out low IoU matches
        if len(matches) > 0:
            to_remove = []
            for m in range(len(matches)):
                if iou_matrix[matches[m, 0], matches[m, 1]] < thresh:
                    unmatched_dets.append(matches[m, 0])
                    unmatched_trks.append(matches[m, 1])
                    to_remove.append(m)
            matches = np.delete(matches, to_remove, axis=0)
        
        return matches, unmatched_dets, unmatched_trks
    
    def _compute_iou_matrix(self, bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
        """Compute IoU between all pairs of bboxes."""
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
        
        xA = np.maximum(x11, x21.T)
        yA = np.maximum(y11, y21.T)
        xB = np.minimum(x12, x22.T)
        yB = np.minimum(y12, y22.T)
        
        interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
        
        boxAArea = (x12 - x11) * (y12 - y11)
        boxBArea = (x22 - x21) * (y22 - y21)
        
        iou = interArea / (boxAArea + boxBArea.T - interArea)
        
        return iou