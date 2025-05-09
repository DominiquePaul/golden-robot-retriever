from voice_interface import *
from art import text2art

import time

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

    while True:
        if space_pressed:
            space_pressed = False
            user_speech, user_desire = dummy_extract_what_the_user_wants_from_voice()
            art = text2art(user_desire)
            lines = art.splitlines()
            # clear_previous_output(previous_lines)
            print(art)

        # other things
        time.sleep(0.01)


if __name__ == "__main__":
    main()
