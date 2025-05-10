import time

import httpx
import numpy as np
from phosphobot.am import ACT
from phosphobot.camera import AllCameras

# Initialize hardware interfaces
PHOSPHOBOT_API_URL = "http://0.0.0.0:80"
allcameras = AllCameras()
time.sleep(1)  # Camera warmup

# Connect to ACT server
model = ACT()

while True:
    # Capture multi-camera frames (adjust camera IDs and size as needed)
    images = [
        allcameras.get_rgb_frame(0, resize=(240, 320)),
        allcameras.get_rgb_frame(1, resize=(240, 320)),
    ]

    # Get current robot state
    state = httpx.post(f"{PHOSPHOBOT_API_URL}/joints/read").json()

    # Generate actions
    start = time.time()
    actions = model(
        {
            "observation.state": np.array(state["angles_rad"]),
            "observation.images.main": np.array(
                allcameras.get_rgb_frame(0, resize=(240, 320))
            ),
            "observation.images.secondary_0": np.array(
                allcameras.get_rgb_frame(1, resize=(240, 320))
            ),
        },
    )
    end = time.time()
    # print(f"Time taken for inference: {end - start:.4f} seconds")
    # actions = actions * 3

    # Execute actions at 30Hz
    for action in actions:
        httpx.post(
            f"{PHOSPHOBOT_API_URL}/joints/write", json={"angles": action.tolist()}
        )
        time.sleep(1 / 30)
