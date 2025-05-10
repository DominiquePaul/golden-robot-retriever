from voice_interface import *
from art import text2art

import time

import pyrealsense2 as rs

import cv2
import matplotlib.pyplot as plt

from pynput import keyboard

space_pressed = False
user_desire = None


def dummy_extract_what_the_user_wants_from_voice():
    return "Please bring me a can of coke", "coke"


def on_press(key):
    global space_pressed
    if key == keyboard.Key.space:
        space_pressed = True


def main():
    global space_pressed, user_desire

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    last_img_checking_time = time.time()
    img_check_delta_in_s = .5

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)
    sensor = pipeline.get_active_profile().get_device().query_sensors()[0]

    # Set the exposure anytime during the operation
    sensor.set_option(rs.option.exposure, 100000.000)

    # Get camera intrinsics
    # color_stream = profile.get_stream(rs.stream.color)

    user_desire = None

    while True:
        if space_pressed:
            space_pressed = False
            user_speech, user_desire = dummy_extract_what_the_user_wants_from_voice()
            art = text2art(user_desire)
            lines = art.splitlines()
            # clear_previous_output(previous_lines)
            print(art)

        # check if we see the object of desire at the moment
        time_since_last_img_check = time.time() - last_img_checking_time
        if time_since_last_img_check > img_check_delta_in_s:
            last_img_checking_time = time.time()

            # take image, check if what we are looking for is in the image, and reachable
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            frame = np.asanyarray(color_frame.get_data())

            cv2.imshow("Color", frame)
            cv2.waitKey(1)

            if user_desire is not None:
                # downscaled = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
                downscaled = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                
                cv2.imshow("Scaled", downscaled)
                cv2.waitKey(1)
                
                obj_visible = check_if_user_object_is_visible_in_image(user_desire, downscaled)

                # ask if we should bring this one
                # ask_if_this_is_what_we_want()

                # should_grasp = record_and_extract_bool()

                # # execute policy
                # if should_grasp:
                #     execute_policy()

        # other things
        time.sleep(0.05)


if __name__ == "__main__":
    main()
