import json_numpy  
import numpy as np  
import requests  
import cv2  
import torch  
  
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot  
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
# If you have a real robot, you would use its API to get joint positions  
# For this example, I'll show how to use the ManipulatorRobot class from lerobot  
# or simulate the data if you don't have a robot connected  
  
def get_observation_from_robot():  
    """Get joint positions and camera images from a robot.  
      
    Returns:  
        dict: Observation dictionary with state and images  
    """  
    try:  
        # Option 1: If you have a robot connected using lerobot  
          
        # Initialize and connect to your robot (simplified example)  
        robot = ManipulatorRobot(config)  
        robot.connect()  
          
        # Capture observation - this returns a dict with observation.state and observation.images.*  
        obs_dict = robot.capture_observation()  
          
        # Convert torch tensors to numpy arrays  
        for key in obs_dict:  
            if isinstance(obs_dict[key], torch.Tensor):  
                obs_dict[key] = obs_dict[key].numpy()  
                  
        return obs_dict  
          
    except (ImportError, NameError):  
        # Option 2: Simulate data if you don't have a robot  
        print("No robot connected. Using simulated data.")  
          
        # Simulate joint positions (6-DOF robot arm)  
        joint_positions = np.array([0.0, -0.5, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)  
          
        # Simulate camera image (RGB)  
        image = np.zeros((480, 640, 3), dtype=np.uint8)  
        # Draw something on the image to make it non-empty  
        cv2.rectangle(image, (100, 100), (300, 300), (0, 255, 0), 2)  
          
        return {  
            "observation.state": joint_positions,  
            "observation.images.0": image  
        }  
  
def run_inference(observation, server_url="http://localhost:8080/act"):  
    """Send observation to inference server and get actions.  
      
    Args:  
        observation (dict): Dictionary with observation.state and observation.images.*  
        server_url (str): URL of the inference server  
          
    Returns:  
        np.ndarray: Action to execute  
    """  
    # Encode the observation using json_numpy  
    encoded_payload = json_numpy.dumps(observation)  
      
    # Prepare the request  
    request_data = {"encoded": encoded_payload}  
      
    # Send the request to the server  
    response = requests.post(server_url, json=request_data)  
      
    if response.status_code != 200:  
        raise Exception(f"Error from server: {response.text}")  
      
    # Decode the response  
    action = json_numpy.loads(response.text)  
    return action  
  
def main():  
    # 1. Get observation from robot (joint positions and camera images)  
    observation = get_observation_from_robot()  
      
    # 2. Run inference to get action  
    action = run_inference(observation)  
      
    # 3. Print or use the action  
    print(f"Received action: {action}")  
      
    # 4. If you have a robot, you would send this action to it  
    # For example:  
    # robot.send_action(torch.from_numpy(action))  
  
if __name__ == "__main__":  
    main()



class ExternalRobotInferenceClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def get_action(self, observation):
        return {"action.single_arm": (16, 5), "action.gripper": (16, 1)}


import time
import numpy as np
import torch

from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config

def main():
    inference_client = ExternalRobotInferenceClient(
        host="51.159.142.77", # VM instance adress
        port=5555
    )

    config = So100RobotConfig(calibration_dir="calibration/so100")
    robot = make_robot_from_config(config)
    robot.connect()

    # Wait for systems to initialize
    time.sleep(1)
    
    while True:
        obs = robot.capture_observation()


        # the keys have been changed

        observation = {
            "http://video.cam_wrist": np.expand_dims(
                obs["observation.images.side"].data.numpy(), axis=0
            ),
            "http://video.cam_context": np.expand_dims(
                obs["http://observation.images.phone"].data.numpy(), axis=0
            ),
            "state.single_arm": np.expand_dims(
                obs["observation.state"].data.numpy()[..., :5].astype(np.float64), axis=0
            ),
            "state.gripper": np.expand_dims(
                obs["observation.state"].data.numpy()[..., 5:6].astype(np.float64), axis=0
            ),
            "annotation.human.task_description": ["Pick a blue lego block and move to the left."]
        }
        
        # Get action from the inference server
        action = inference_client.get_action(observation)
        # output is in format :
        # { 
        #   "action.single_arm": (16, 5),
        #   "action.gripper": (16, 1)
        # }

        single_arm_action = action["action.single_arm"]
        gripper_action = action["action.gripper"]
        
        # horizion of 16 timesteps, we run inference only every 16 timesteps
        
        for i in range(16):
            step_single_arm = single_arm_action[i]
            step_gripper = gripper_action[i]
            step_action = torch.cat([torch.tensor(step_single_arm, dtype=torch.float32),
                                    torch.tensor([step_gripper], dtype=torch.float32)])
            robot.send_action(step_action)
            time.sleep(1/30) # 30Hz control frequency

        """
        step_action = http://torch.cat([torch.tensor(single_arm_action[0], dtype=torch.float32),
                                torch.tensor([gripper_action[0]], dtype=torch.float32)])
        robot.send_action(step_action)
        time.sleep(1/30)"""

if __name__ == "__main__":
    main()

