# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from tensorflow.keras.models import load_model
import resizer as re
import threading
import argparse
import imutils
import time
import cv2

WIDTH = HEIGHT = 100

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame2 = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# loading model
model = load_model("model98keypoints.h5")

# initialize the video stream and allow the camera sensor to
# warmup
vs = VideoStream(src=0).start()
time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


obj = re.Resizer(WIDTH, HEIGHT, 1.1)


def get_keypoints():
    global vs, lock, frame_original, outputFrame2
    while True:
        frame_original = vs.read()
        frame = imutils.resize(frame_original, width=400)
        img, faces = obj.get_resized_withoutdata(frame)
        try:
            faces = faces[0]
            temp = img[0].copy()
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            temp = temp.reshape(1, WIDTH, HEIGHT, 1)
            data = model.predict(temp)

            for i in range(0, len(data[0]), 2):
                cv2.circle(img[0], center=(data[0][i], data[0][i + 1]), radius=1,
                           color=(255, 255, 255))

            frame[faces[1]:faces[1] + faces[3], faces[0]:faces[0] + faces[2], :] = cv2.resize(img[0],
                                                                                              (faces[2], faces[3]))
        except:
            pass
        with lock:
            outputFrame2 = frame.copy()

def generate1():
    # grab global references to the output frame and lock variables
    global outputFrame2, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame2 is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", frame_original)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


def generate2():
    # grab global references to the output frame and lock variables
    global outputFrame2, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame2 is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame2)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/original_feed")
def origianl_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate1(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/keypoints_feed")
def keypoints_feed():
    return Response(generate2(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())
    t = threading.Thread(target=get_keypoints)
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
