import httpx
import time
import numpy as np
import cv2  # OpenCV is more widely available than PyAV

# Define constants
PHOSPHOBOT_API_URL = "http://localhost:80"

class SimpleCamera:
    """A simple camera class to replace phosphobot's AllCameras"""
    def __init__(self):
        self.cameras = {}
        # Try to initialize camera 0
        try:
            self.cameras[0] = cv2.VideoCapture(0)
            if not self.cameras[0].isOpened():
                print("Warning: Could not open camera 0")
        except Exception as e:
            print(f"Error initializing camera: {e}")
    
    def get_rgb_frame(self, camera_id=0, resize=None):
        """Get a frame from the specified camera"""
        if camera_id not in self.cameras:
            return np.zeros((240, 320, 3), dtype=np.uint8)  # Return black frame if camera not available
        
        ret, frame = self.cameras[camera_id].read()
        if not ret:
            return np.zeros((240, 320, 3), dtype=np.uint8)  # Return black frame if read failed
        
        # Convert BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if specified
        if resize is not None:
            frame = cv2.resize(frame, (resize[1], resize[0]))
            
        return frame
    
    def release(self):
        """Release all camera resources"""
        for cam in self.cameras.values():
            cam.release()

class SimpleACT:
    """Simple replacement for phosphobot's ACT model"""
    def __init__(self):
        print("Initializing simple ACT model")
        
    def __call__(self, inputs):
        """Generate simple actions based on inputs"""
        # This is a placeholder that returns a simple action sequence
        # In a real implementation, this would contain your ML model
        state = inputs.get("state", np.zeros(6))
        
        # Create a simple oscillating motion for demo purposes
        time_val = time.time()
        actions = []
        for i in range(10):  # Generate 10 actions
            # Create sinusoidal movements with different phases
            action = state.copy()
            for j in range(len(action)):
                action[j] += 0.05 * np.sin(time_val + i*0.1 + j*0.5)
            actions.append(action)
            
        return actions

# Initialize camera and model
simple_camera = SimpleCamera()
time.sleep(1)  # Camera warmup

# Initialize simple ACT model
model = SimpleACT()

print("Starting control loop...")
try:
    while True:
        # Capture camera frames
        images = [simple_camera.get_rgb_frame(0, resize=(240, 320))]

        # Get current robot state (joint angles)
        try:
            state = httpx.post(f"{PHOSPHOBOT_API_URL}/joints/read").json()
        except Exception as e:
            print(f"Error getting robot state: {e}")
            state = {"angles_rad": [0.0] * 6}  # Default state with 6 joints at 0 position

        # Generate actions using the model
        actions = model(
            {"state": np.array(state["angles_rad"]), "images": np.array(images)}
        )

        # Execute actions at 30Hz
        for action in actions:
            try:
                httpx.post(
                    f"{PHOSPHOBOT_API_URL}/joints/write", json={"angles": action.tolist()}
                )
            except Exception as e:
                print(f"Error sending action to robot: {e}")
            
            time.sleep(1 / 30)
except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Clean up resources
    simple_camera.release()
    print("Resources released. Exiting.")