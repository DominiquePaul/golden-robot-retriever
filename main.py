import threading
import time

import cv2
import openai
from art import text2art
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt

load_dotenv()

from golden_robot_retriever.openai_interface import OpenAIInterface
from golden_robot_retriever.cameras import get_camera

space_pressed = False

latest_frame = None
frame_lock = threading.Lock()

DEBUG = True

if not DEBUG:
    from golden_robot_retriever.robot_code.client import GraspingPolicy


def display_frames(camera, stop_event):
    """Display camera frames and check for space bar presses in a separate thread."""
    global latest_frame
    while not stop_event.is_set():
        frame = camera.get_frame()

        if frame is None:
            print("Warning: Camera returned None frame")
            time.sleep(0.1)
            continue

        # Validate frame before displaying
        if frame.ndim != 3 or frame.shape[2] != 3:
            print(f"Warning: Invalid frame dimensions: {frame.shape}")
            time.sleep(0.1)
            continue

        # Ensure frame values are within valid range for display
        try:
            # Make a normalized copy for display
            display_frame = frame.copy()
            if display_frame.dtype != np.uint8:
                display_frame = np.clip(display_frame, 0, 255).astype(np.uint8)

            with frame_lock:
                latest_frame = frame.copy()

            # Display the frame in a window
            # Display with matplotlib instead of cv2
            # plt.figure("Camera Feed")
            # plt.imshow(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.draw()
            # plt.pause(0.001)
        except Exception as e:
            print(f"Error displaying frame: {e}")

        time.sleep(0.05)


def mock_run_policy(max_runtime_s=15):
    """Execute the robot's grasping policy and return success status."""
    print("Attempting to execute policy")
    return True


class GoldenRetriever:
    def __init__(self):
        client = openai.OpenAI()
        self.ai = OpenAIInterface(client)
        self.last_img_checking_time = time.time()
        self.img_check_delta_in_s = 0.5
        self.last_asking_time = 0
        self.asking_delta_in_s = 5

        if not DEBUG:
            self.policy = GraspingPolicy()

        try:
            self.camera = get_camera("webcam")
            self.user_desire = None
            self.stop_event = threading.Event()
            self.display_thread = threading.Thread(
                target=display_frames, args=(self.camera, self.stop_event), daemon=True
            )
            self.display_thread.start()
        except Exception as e:
            print(f"Error initializing camera: {e}")
            print("Running without camera functionality")
            self.camera = None
            self.stop_event = threading.Event()
            self.display_thread = None
            self.user_desire = None

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.cleanup()

    def cleanup(self):
        """Properly close resources."""
        if hasattr(self, "stop_event"):
            self.stop_event.set()

        if (
            hasattr(self, "display_thread")
            and self.display_thread is not None
            and self.display_thread.is_alive()
        ):
            self.display_thread.join(timeout=1.0)

        if (
            hasattr(self, "camera")
            and self.camera is not None
            and hasattr(self.camera, "close")
        ):
            self.camera.close()

        # Close any OpenCV windows
        cv2.destroyAllWindows()

    @property
    def time_since_last_ask(self):
        return time.time() - self.last_asking_time

    @property
    def time_to_evaluate_image_again(self):
        time_since_last_img_check = time.time() - self.last_img_checking_time
        result = time_since_last_img_check > self.img_check_delta_in_s
        if result:
            self.last_img_checking_time = time.time()
        return result

    def ask_user_for_desire(self):
        self.last_asking_time = time.time()
        self.ai.speak_with_personality("How can I help you?")

    def run(self):
        """Main function orchestrating the retrieval robot's behavior loop."""

        # listener = keyboard.Listener(on_press=on_press)
        # listener.start()

        try:
            while True:
                should_ask_user_for_desire = (
                    self.user_desire is None
                    and self.time_since_last_ask > self.asking_delta_in_s
                )

                if should_ask_user_for_desire:
                    self.ask_user_for_desire()

                if self.user_desire is None:
                    if DEBUG:
                        user_speech = "One Coca cola please."
                        self.user_desire = "Coca cola"
                        self.ai.add_to_history("user", user_speech)
                    else:
                        user_speech, self.user_desire = self.ai.extract_what_the_user_wants_from_voice()

                    if user_speech is None and self.user_desire is None:
                        self.ai.add_to_history("user", "User did not answer.")
                        continue

                    elif self.user_desire is None:
                        break

                    print(
                        text2art(self.user_desire)
                    )  # print ascii art of desired object for extra prominence in console.
                    
                    # Only add to history if not already added (for non-DEBUG mode)
                    if not DEBUG:
                        self.ai.add_to_history("user", user_speech)

                # check if we see the object of desire at the moment
                if self.user_desire is not None and self.time_to_evaluate_image_again:
                    print("loop")
                    with frame_lock:
                        if latest_frame is None:
                            raise Exception("No frame found")
                        frame_copy = (
                            latest_frame.copy() if latest_frame is not None else None
                        )

                    if frame_copy is not None:
                        downscaled_img = cv2.resize(frame_copy, (0, 0), fx=0.5, fy=0.5)
                        obj_visible = self.ai.check_if_user_object_is_visible_in_image(
                            self.user_desire, downscaled_img
                        )
                        print(obj_visible)

                        if not obj_visible:
                            self.ai.speak_with_personality(f"We did not yet find {self.user_desire}, we'll keep looking.")
                        else:
                            # ask if we should bring the object
                            question = self.ai.speak_with_personality(f"I found the {self.user_desire}! Shall I get it?")

                            user_answer, should_grasp = self.ai.record_and_extract_bool(question)
                            if user_answer is None and should_grasp is None:
                                should_grasp = False
                                user_answer = "User did not say anything"

                            self.ai.add_to_history("user", user_answer)

                            # execute policy
                            if should_grasp:
                                success = False

                                while not success:
                                    if DEBUG:
                                        success = mock_run_policy(max_runtime_s=15)
                                    else:
                                        success = self.policy.run(max_runtime_s=15)

                                    if success:
                                        self.ai.add_to_history(
                                            "user",
                                            "We successfully solved this problem. On to the next one!"
                                        )
                                    else:
                                        assistant_query = self.ai.speak_with_personality("Should we try this again?")
                                        user_answer, try_again = self.ai.record_and_extract_bool(
                                            assistant_query
                                        )

                                        if not try_again:
                                            self.ai.add_to_history(
                                                "user",
                                                "We were unfortunately not able to solve this problem. Let's check if we can do something else!"
                                            )
                                            break

                                # tell user that we grasped the obj
                                self.ai.speak_with_personality("Ok, we grasped the object, bringing it to you now.")

                                should_drop = False
                                # drop the stuff when back at the persons place
                                while not should_drop:
                                    # check if a hand is visible in the camera
                                    print("Not yet dropping, am not seeing the signal.")
                                    with frame_lock:
                                        frame_copy = (
                                            latest_frame.copy()
                                            if latest_frame is not None
                                            else None
                                        )

                                    if frame_copy is not None:
                                        downscaled = cv2.resize(
                                            frame_copy, (0, 0), fx=0.5, fy=0.5
                                        )

                                        should_drop = self.ai.check_if_user_object_is_visible_in_image(
                                            "open hand", downscaled
                                        )
                                    time.sleep(4)

                                self.ai.speak_with_personality(f"Here's your {self.user_desire}")
                                self.user_desire = None

                # other things
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.cleanup()


if __name__ == "__main__":
    dog = GoldenRetriever()
    dog.run()
