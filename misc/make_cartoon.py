import base64
import datetime
import time
import numpy as np

from openai import OpenAI
import cv2
import pyrealsense2 as rs

from pynput import keyboard

space_pressed = False


def on_press(key):
    global space_pressed
    if key == keyboard.Key.space:
        space_pressed = True


def main():
    global space_pressed

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
    sensor.set_option(rs.option.exposure, 10000.000)

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        cv2.imshow("Color", frame)
        cv2.waitKey(1)

        if True or space_pressed:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = f"image_{timestamp}.jpg"
            retval, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

            with open(img_path, "wb") as f:
                f.write(buffer)
            client = OpenAI()

            prompt = "This is me in the image. First onvert this to a studio ghibli like image. I then want to draw this with a single line. Please convert this to a blck and white version of an image that can be drawn in one line."

            result = client.images.edit(
                model="gpt-image-1",
                image=[open(img_path, "rb")],
                prompt=prompt,
            )

            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)


            # Save the image to a file
            with open(f"{timestamp}_cartoon.png", "wb") as f:
                f.write(image_bytes)

            break

        time.sleep(0.05)


if __name__ == "__main__":
    main()
