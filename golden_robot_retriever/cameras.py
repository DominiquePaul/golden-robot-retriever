"""
This file contains the classes for different cameras. Camera images fetch frames that are sent to OpenAi.

Camera options:
- WebcamCamera: for webcam
- RealSenseCamera: for realsense camera
"""
import typing as t
import numpy as np

def get_camera(camera_type: t.Literal["webcam", "realsense"]):
    """Get a camera instance based on the camera type."""
    if camera_type == "webcam":
        return WebcamCamera()
    elif camera_type == "realsense":
        return RealSenseCamera()
    else:
        raise ValueError(f"Invalid camera type: {camera_type}")


class CameraWrapperBase:
    """Base class for camera implementations with common interface."""
    def __init__(self):
        self.camera = None

    def get_frame(self):
        """Return the current camera frame."""
        pass
    
class WebcamCamera:
    """Camera implementation using standard webcam via OpenCV."""
    def __init__(
        self,
        camera_id=0,  # Default to camera 0 instead of None
    ):
        import cv2
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"Warning: Could not open camera with ID {camera_id}")
            
    def get_frame(self):
        """Capture and return the current webcam frame."""
        if not self.cap.isOpened():
            print("Error: Camera is not available")
            return None
            
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            print("Warning: Failed to read frame from camera")
            return None
            
    def close(self):
        """Properly release the camera resources."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

class RealSenseCamera:
    """Camera implementation using Intel RealSense depth camera."""
    def __init__(self):
        try:
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            config = rs.config()

            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Start streaming
            profile = self.pipeline.start(config)
            sensor = self.pipeline.get_active_profile().get_device().query_sensors()[0]
            sensor.set_option(rs.option.exposure, 50000.000)
            self.rs = rs
        except ImportError:
            print("Warning: pyrealsense2 module not found. RealSenseCamera will not function.")
            self.pipeline = None
            self.rs = None

    def get_frame(self):
        """Capture and return the current color frame from the RealSense camera."""
        if self.pipeline is None:
            return None
            
        frames = self.pipeline.wait_for_frames()
        # frames = self.pipeline.poll_for_frames()
        if frames:
            color_frame = frames.get_color_frame()
            frame = np.asanyarray(color_frame.get_data())

            return frame

        return None
    
    def close(self):
        """Properly release the camera resources."""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            self.pipeline.stop()
    
