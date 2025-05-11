# import matplotlib.pyplot as plt
import os
import threading
import time

import cv2
import openai
from art import text2art
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt

load_dotenv()

from golden_robot_retriever.openai_interface import (
    convert_text_to_personality,
    text_to_speech,
    extract_what_the_user_wants_from_voice,
    record_and_extract_bool,
    check_if_user_object_is_visible_in_image,
)
from golden_robot_retriever.cameras import get_camera

space_pressed = False

latest_frame = None
frame_lock = threading.Lock()

DEBUG_SPEECH = False
DEBUG_POLICY = False

DISPLAY_IMG = False

if not DEBUG_POLICY:
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
            if DISPLAY_IMG:
                cv2.imshow("Camera Feed", display_frame)
                cv2.waitKey(1)
        except Exception as e:
            print(f"Error displaying frame: {e}")

        time.sleep(0.05)


def text_to_personality_speech(client, text, context):
    """Convert text to speech matching the golden retriever personality and play it."""
    new_text = convert_text_to_personality(client, text, context)

    speech_path = text_to_speech(client, new_text, context)
    os.system(f"mpg123 '{speech_path}'")

    return new_text


def ask_what_the_user_wants(client, context):
    """Ask the user how the robot can help them."""
    text = "How can I help you?"
    return text_to_personality_speech(client, text, context)


def ask_if_this_is_what_we_want(client, context, obj):
    """Ask the user if the detected object is what they requested."""
    text = f"I found the {obj}! Shall I get it?"
    return text_to_personality_speech(client, text, context)


def we_grabbed_stuff(client, context):
    """Inform the user that the object was grasped successfully."""
    text = "Ok, we grasped the object, bringing it to you now."
    return text_to_personality_speech(client, text, context)


def tell_that_we_cant_see_obj_yet(client, context, obj, obj_that_we_can_see):
    """Inform the user that the requested object hasn't been found yet."""
    text = (
        f"We did not yet find {obj} we only found {obj_that_we_can_see}"
    )
    return text_to_personality_speech(client, text, context)


def should_try_again(client, context):
    """Ask the user if another attempt should be made."""
    text = "Should we try this again?"
    return text_to_personality_speech(client, text, context)


def please_take_the_object(client, context, obj):
    """Tell the user that he should take the object."""
    text = f"Here's your {obj}"
    return text_to_personality_speech(client, text, context)


def mock_run_policy(max_runtime_s=15):
    """Execute the robot's grasping policy and return success status."""
    print("Attempting to execute policy")
    return True


class GoldenRetriever:
    def __init__(self):
        self.client = openai.OpenAI()
        self.last_img_checking_time = time.time()
        self.img_check_delta_in_s = 0.5
        self.last_asking_time = 0
        self.asking_delta_in_s = 5

        if not DEBUG_POLICY:
            self.policy = GraspingPolicy()

        try:
            self.camera = get_camera("webcam")
            self.user_desire = None
            self.conversation_history = []
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
            self.conversation_history = []

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
        text = ask_what_the_user_wants(self.client, self.conversation_history)
        self.conversation_history.append({"role": "assistant", "content": text})

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
                    if DEBUG_SPEECH:
                        user_speech = "One Coca cola please."
                        self.user_desire = "Coca cola"
                    else:
                        user_speech, self.user_desire = (
                            extract_what_the_user_wants_from_voice(self.client)
                        )

                    if user_speech is None and self.user_desire is None:
                        self.conversation_history.append(
                            {"role": "user", "content": "User did not answer."}
                        )
                        continue

                    elif self.user_desire is None:
                        break

                    print(
                        text2art(self.user_desire)
                    )  # print ascii art of desired object for extra prominence in console.
                    self.conversation_history.append(
                        {"role": "user", "content": user_speech}
                    )

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
                        obj_visible, what_we_see = (
                            check_if_user_object_is_visible_in_image(
                                self.client, self.user_desire, downscaled_img
                            )
                        )
                        print(obj_visible)

                        if not obj_visible:
                            tell_that_we_cant_see_obj_yet(
                                self.client,
                                self.conversation_history,
                                self.user_desire,
                                what_we_see,
                            )

                        else:
                            # ask if we should bring the object
                            question = ask_if_this_is_what_we_want(
                                self.client, self.conversation_history, self.user_desire
                            )
                            self.conversation_history.append(
                                {"role": "assistant", "content": question}
                            )

                            user_answer, should_grasp = record_and_extract_bool(
                                self.client, question
                            )
                            if user_answer is None and should_grasp is None:
                                should_grasp = False
                                user_answer = "User did not say anything"

                            self.conversation_history.append(
                                {"role": "user", "content": user_answer}
                            )

                            # execute policy
                            if should_grasp:
                                success = False

                                while not success:
                                    if DEBUG_POLICY:
                                        success = mock_run_policy(max_runtime_s=15)
                                    else:
                                        success = self.policy.run(max_runtime_s=20)

                                    if success:
                                        self.conversation_history.append(
                                            {
                                                "role": "user",
                                                "content": "We successfully solved this problem. On to the next one!",
                                            }
                                        )
                                    else:
                                        assistant_query = should_try_again(
                                            self.client, self.conversation_history
                                        )
                                        self.conversation_history.append(
                                            {
                                                "role": "assistant",
                                                "content": assistant_query,
                                            }
                                        )
                                        user_answer, try_again = (
                                            record_and_extract_bool(
                                                self.client, assistant_query
                                            )
                                        )

                                        if not try_again:
                                            self.conversation_history.append(
                                                {
                                                    "role": "user",
                                                    "content": "We were unfortunately not able to solve this problem. Let's check if we can do something else!",
                                                }
                                            )
                                            break

                                # tell user that we grasped the obj
                                text = we_grabbed_stuff(
                                    self.client, self.conversation_history
                                )
                                self.conversation_history.append(
                                    {
                                        "role": "assistant",
                                        "content": text,
                                    }
                                )

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

                                        should_drop, _ = (
                                            check_if_user_object_is_visible_in_image(
                                                self.client, "open hand", downscaled
                                            )
                                        )
                                    time.sleep(4)

                                please_take_the_object(
                                    self.client,
                                    self.conversation_history,
                                    self.user_desire,
                                )

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
