import time
import numpy as np

import httpx
from phosphobot.am import ACT
from phosphobot.camera import AllCameras


class GraspingPolicy:
    def __init__(self):
        # Initialize hardware interfaces
        self.PHOSPHOBOT_API_URL = "http://0.0.0.0:80"
        self.allcameras = AllCameras()
        time.sleep(1)  # Camera warmup

        # Connect to ACT server
        self.model = ACT()

    def run(self, max_runtime_s=15):
        start_time = time.time()

        # Initialize robot state
        httpx.post(
                f"{self.PHOSPHOBOT_API_URL}/joints/write",
                json={"angles": [0,-1.57,1.57,-2,0,0]},
        )
        time.sleep(1)

        while True:
            # Get current robot state
            state = httpx.post(f"{self.PHOSPHOBOT_API_URL}/joints/read").json()

            # Generate actions
            actions = self.model(
                {
                    "observation.state": np.array(state["angles_rad"]),
                    "observation.images.main": np.array(
                        self.allcameras.get_rgb_frame(0, resize=(240, 320))
                    ),
                    "observation.images.secondary_0": np.array(
                        self.allcameras.get_rgb_frame(1, resize=(240, 320))
                    ),
                },
            )

            # Execute actions at 30Hz
            for action in actions:
                httpx.post(
                    f"{self.PHOSPHOBOT_API_URL}/joints/write",
                    json={"angles": action.tolist()},
                )
                time.sleep(1 / 30)

            if time.time() - start_time > max_runtime_s:
                break

        return True


if __name__ == "__main__":
    policy = GraspingPolicy()
    policy.run()
