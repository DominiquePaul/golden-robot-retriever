import numpy as np
from golden_robot_retriever.cameras import WebcamCamera

def test_webcam_camera():
    camera = WebcamCamera()
    frame = camera.get_frame()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape[0] > 0 and frame.shape[1] > 0
    camera.close()

