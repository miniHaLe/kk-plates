"""RTSP stream reader using GStreamer with OpenCV fallback."""

import time
import threading
import queue
from typing import Optional, Tuple, Generator
import cv2
import numpy as np
import structlog

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    GSTREAMER_AVAILABLE = True
except ImportError:
    GSTREAMER_AVAILABLE = False

logger = structlog.get_logger()


class RTSPReader:
    """RTSP stream reader with GStreamer preferred, OpenCV fallback."""
    
    def __init__(self, rtsp_url: str, latency_ms: int = 60):
        self.rtsp_url = rtsp_url
        self.latency_ms = latency_ms
        self.frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.pipeline = None
        self.use_gstreamer = GSTREAMER_AVAILABLE
        
    def start(self) -> None:
        """Start capturing frames."""
        if self.running:
            return
            
        self.running = True
        
        if self.use_gstreamer:
            try:
                self._start_gstreamer()
                logger.info("Started RTSP capture with GStreamer", url=self.rtsp_url)
            except Exception as e:
                logger.warning("GStreamer failed, falling back to OpenCV", error=str(e))
                self.use_gstreamer = False
                self._start_opencv()
        else:
            self._start_opencv()
    
    def _start_gstreamer(self) -> None:
        """Start GStreamer pipeline."""
        Gst.init(None)
        
        pipeline_str = (
            f"rtspsrc location={self.rtsp_url} latency={self.latency_ms} ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink name=sink"
        )
        
        self.pipeline = Gst.parse_launch(pipeline_str)
        sink = self.pipeline.get_by_name("sink")
        sink.set_property("emit-signals", True)
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 1)
        sink.set_property("drop", True)
        sink.connect("new-sample", self._on_new_sample)
        
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # Start GLib main loop in thread
        self.thread = threading.Thread(target=self._gst_thread)
        self.thread.daemon = True
        self.thread.start()
    
    def _gst_thread(self) -> None:
        """GStreamer main loop thread."""
        loop = GLib.MainLoop()
        try:
            loop.run()
        except Exception as e:
            logger.error("GStreamer loop error", error=str(e))
    
    def _on_new_sample(self, sink) -> bool:
        """Handle new GStreamer sample."""
        sample = sink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            # Extract frame dimensions
            struct = caps.get_structure(0)
            width = struct.get_value("width")
            height = struct.get_value("height")
            
            # Convert buffer to numpy array
            result, mapinfo = buffer.map(Gst.MapFlags.READ)
            if result:
                frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
                frame = frame.reshape((height, width, 3))
                buffer.unmap(mapinfo)
                
                # Add to queue (drop old frames if full)
                try:
                    self.frame_queue.put_nowait((time.time(), frame))
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((time.time(), frame))
                    except queue.Empty:
                        pass
        
        return True
    
    def _start_opencv(self) -> None:
        """Start OpenCV capture."""
        self.cap = cv2.VideoCapture(self.rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open RTSP stream: {self.rtsp_url}")
        
        logger.info("Started RTSP capture with OpenCV", url=self.rtsp_url)
        
        self.thread = threading.Thread(target=self._opencv_thread)
        self.thread.daemon = True
        self.thread.start()
    
    def _opencv_thread(self) -> None:
        """OpenCV capture thread."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                try:
                    self.frame_queue.put_nowait((time.time(), frame))
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((time.time(), frame))
                    except queue.Empty:
                        pass
            else:
                logger.warning("Failed to read frame from RTSP")
                time.sleep(0.1)
    
    def read_frame(self, timeout: float = 1.0) -> Optional[Tuple[float, np.ndarray]]:
        """Read a frame from the queue."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def frames(self) -> Generator[Tuple[float, np.ndarray], None, None]:
        """Generator yielding frames."""
        while self.running:
            frame_data = self.read_frame()
            if frame_data:
                yield frame_data
    
    def stop(self) -> None:
        """Stop capturing."""
        self.running = False
        
        if self.use_gstreamer and self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        if self.cap:
            self.cap.release()
        
        if self.thread:
            self.thread.join(timeout=2.0)
        
        logger.info("Stopped RTSP capture")
    
    def get_fps(self) -> float:
        """Get stream FPS."""
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 25.0  # Default for Hikvision
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get stream resolution (width, height)."""
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return 1920, 1080  # Default 1080p