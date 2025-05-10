# import matplotlib.pyplot as plt
import os
import threading
import time

import cv2
import openai
import pyrealsense2 as rs
from art import text2art

from openai_interface import *

## setup key with
# echo "export OPENAI_API_KEY='yourkey'" >> ~/.bashrc
# source ~/.bashrc
## test that it worked with
# echo $OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]


space_pressed = False

latest_frame = None
frame_lock = threading.Lock()


def dummy_extract_what_the_user_wants_from_voice():
    return "Please bring me a can to drink", "can"


def display_frames(pipeline, stop_event):
    global latest_frame
    global space_pressed
    while not stop_event.is_set():
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        with frame_lock:
            latest_frame = frame.copy()

        cv2.imshow("Color", frame)
        # cv2.waitKey(1)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # space bar
            space_pressed = True

        time.sleep(0.05)


def ask_what_the_user_wants(client, context):
    text = "How can I help you?"

    new_text = convert_text_to_personality(client, text, context)

    speech_path = text_to_speech(
        client, new_text, context, GoldenRetrieverMood.CHEERFUL
    )
    os.system(f"mpg123 {speech_path}")

    return new_text


def ask_if_this_is_what_we_want(client, context):
    question_text = "Is this what you want?"
    new_text = convert_text_to_personality(client, question_text, context)

    speech_path = text_to_speech(
        client, new_text, context, GoldenRetrieverMood.SARCASTIC
    )
    os.system(f"mpg123 {speech_path}")

    return new_text


def should_try_again(client, context):
    question_text = "Should we try this again?"
    new_text = convert_text_to_personality(client, question_text, context)

    speech_path = text_to_speech(
        client, new_text, context, GoldenRetrieverMood.SARCASTIC
    )
    os.system(f"mpg123 {speech_path}")

    return new_text


def run_policy():
    print("Attempting to execute policy")
    return True


def main():
    global space_pressed

    client = openai.OpenAI()

    # listener = keyboard.Listener(on_press=on_press)
    # listener.start()

    last_img_checking_time = time.time()
    img_check_delta_in_s = 0.5

    last_asking_time = 0
    asking_delta_in_s = 5

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)
    sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
    sensor.set_option(rs.option.exposure, 10000.000)

    # Start display thread
    stop_event = threading.Event()
    display_thread = threading.Thread(
        target=display_frames, args=(pipeline, stop_event), daemon=True
    )
    display_thread.start()

    user_desire = None

    complete_context = []

    while True:
        time_since_last_ask = time.time() - last_asking_time
        if user_desire is None and time_since_last_ask > asking_delta_in_s:
            last_asking_time = time.time()
            text = ask_what_the_user_wants(client, complete_context)

            complete_context.append({"role": "assistant", "content": text})

        if True and user_desire is None:
            space_pressed = False
            # user_speech, user_desire = dummy_extract_what_the_user_wants_from_voice()
            user_speech, user_desire = extract_what_the_user_wants_from_voice(client)

            if user_speech is None and user_desire is None:
                complete_context.append(
                    {"role": "user", "content": "User did not answer."}
                )
                continue

            elif user_desire is None:
                break

            art = text2art(user_desire)
            print(art)

            complete_context.append({"role": "user", "content": user_speech})

        # check if we see the object of desire at the moment
        if user_desire is not None:
            time_since_last_img_check = time.time() - last_img_checking_time
            if time_since_last_img_check > img_check_delta_in_s:
                last_img_checking_time = time.time()

                with frame_lock:
                    frame_copy = (
                        latest_frame.copy() if latest_frame is not None else None
                    )

                if user_desire is not None and frame_copy is not None:
                    downscaled = cv2.resize(frame_copy, (0, 0), fx=0.5, fy=0.5)
                    # cv2.imshow("Scaled", downscaled)
                    # cv2.waitKey(1)

                    obj_visible = check_if_user_object_is_visible_in_image(
                        client, user_desire, downscaled
                    )

                    print(obj_visible)

                    # ask if we should bring this one
                    if obj_visible:
                        question = ask_if_this_is_what_we_want(client, complete_context)
                        complete_context.append(
                            {"role": "assistant", "content": question}
                        )

                        user_answer, should_grasp = record_and_extract_bool(
                            client, question
                        )
                        if user_answer is None and should_grasp is None:
                            should_grasp = False
                            user_answer = "User did not say anything"

                        complete_context.append(
                            {"role": "user", "content": user_answer}
                        )

                        # execute policy
                        if should_grasp:
                            success = False

                            while not success:
                                success = run_policy()

                                if success:
                                    user_desire = None
                                    complete_context.append(
                                        {
                                            "role": "user",
                                            "content": "We successfully solved this problem. On to the next one!",
                                        }
                                    )
                                else:
                                    assistant_query = should_try_again(
                                        client, complete_context
                                    )
                                    complete_context.append(
                                        {
                                            "role": "assistant",
                                            "content": assistant_query,
                                        }
                                    )
                                    user_answer, try_again = record_and_extract_bool(
                                        client, assistant_query
                                    )

                                    if not try_again:
                                        complete_context.append(
                                            {
                                                "role": "user",
                                                "content": "We were unfortunately not able to solve this problem. Let's check if we can do something else!",
                                            }
                                        )
                                        break

        # other things
        time.sleep(0.05)

    stop_event.set()
    display_thread.join()


if __name__ == "__main__":
    main()
