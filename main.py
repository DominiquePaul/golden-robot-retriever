from voice_interface import *
from art import text2art

import time

import pyrealsense2 as rs

import cv2

# import matplotlib.pyplot as plt
import os
import threading

from pynput import keyboard

space_pressed = False

latest_frame = None
frame_lock = threading.Lock()


def dummy_extract_what_the_user_wants_from_voice():
    return "Please bring me a can to drink", "can"


def display_frames(pipeline, stop_event):
    global latest_frame
    while not stop_event.is_set():
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        with frame_lock:
            latest_frame = frame.copy()

        cv2.imshow("Color", frame)
        cv2.waitKey(1)

        time.sleep(0.05)


def on_press(key):
    global space_pressed
    if key == keyboard.Key.space:
        space_pressed = True


def ask_what_the_user_wants(context):
    text = "How can I help you?"

    new_text = convert_text_to_personality(text, context)

    speech_path = text_to_speech(new_text, context, GoldenRetrieverMood.CHEERFUL)
    os.system(f"mpg123 {speech_path}")

    return new_text


def ask_if_this_is_what_we_want(context):
    question_text = "Is this what you want?"
    new_text = convert_text_to_personality(question_text, context)

    speech_path = text_to_speech(new_text, context, GoldenRetrieverMood.SARCASTIC)
    os.system(f"mpg123 {speech_path}")

    return new_text


def execute_policy():
    print("Attempting to execute policy")
    return True


def main():
    global space_pressed

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    last_img_checking_time = time.time()
    img_check_delta_in_s = 0.5

    last_asking_time = 0
    asking_delta_in_s = 3

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

    # TODO: maintain the whole context for more sassiness/context
    complete_context = ""

    while True:
        time_since_last_ask = time.time() - last_asking_time
        if user_desire is None and time_since_last_ask > asking_delta_in_s:
            last_asking_time = time.time()
            text = ask_what_the_user_wants(complete_context)

            complete_context += "Assistant: " + text

        if space_pressed and user_desire is None:
            space_pressed = False
            # user_speech, user_desire = dummy_extract_what_the_user_wants_from_voice()
            user_speech, user_desire = extract_what_the_user_wants_from_voice()

            if user_desire is None:
                break

            art = text2art(user_desire)
            lines = art.splitlines()
            # clear_previous_output(previous_lines)
            print(art)

            complete_context += "User: " + user_speech

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
                        user_desire, downscaled
                    )

                    print(obj_visible)

                    # ask if we should bring this one
                    if obj_visible:
                        question = ask_if_this_is_what_we_want(complete_context)
                        complete_context += "Assistant" + question

                        user_answer, should_grasp = record_and_extract_bool(question)
                        complete_context += "User: " + user_answer

                        # execute policy
                        if should_grasp:
                            success = execute_policy()

                            if success:
                                user_desire = None
                                complete_context += "We successfully solved this problem. On to the next one!"

        # other things
        time.sleep(0.05)

    stop_event.set()
    display_thread.join()


if __name__ == "__main__":
    main()
