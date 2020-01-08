import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import PPO

from keras.models import load_model

import utils

sio = socketio.Server()
app = Flask(__name__)
# model = load_model("D:\Socket\model.h5")
model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10

numEpisode = 0
endEpisode = False
episodeReward = 0.0
prevStraight = 0
prevNonStraight = 0

width = 200
height = 66
channels = 3

speed_limit = MAX_SPEED

numActions = 0

actionsTaken = [[0.0]]
advantagey = [0.0]

num_envs = 1

@sio.on('act') #run this for a simple example, to check what you need for act
def act(sid, data):
    if data:
        global speed_limit, numActions, actionsTaken, episodeReward, endEpisode, prevStraight, prevNonStraight, numEpisode
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        reward = float(data["reward"])
        endEpisode = data["resetEnv"]
        checkpointStraight = int(data["checkpointStraight"])
        checkpointNonStraight = int(data["checkpointNonStraight"])

        if prevStraight < checkpointStraight:
            reward += 0.5
        elif prevNonStraight < checkpointNonStraight:
            reward += 1.0

        if reward == 0: reward = -0.01

        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)  # from PIL image to numpy array
            image = utils.preprocess(image)  # apply the preprocessing
            image = np.array([image])  # the model expects 4D array
            return image, reward,

        except Exception as e:
            print(e)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)



@sio.on('telemetry')
def telemetry(sid, data): #rename to act, and dont run everything in this file
    if data:
        global speed_limit, numActions, actionsTaken, episodeReward, endEpisode, prevStraight, prevNonStraight, numEpisode
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        reward = float(data["reward"])
        endEpisode = data["resetEnv"]
        checkpointStraight = int(data["checkpointStraight"])
        checkpointNonStraight = int(data["checkpointNonStraight"])

        if prevStraight < checkpointStraight:
            reward += 0.5
        elif prevNonStraight < checkpointNonStraight:
            reward += 1.0

        episodeReward += reward
        if endEpisode == "True":
            if numEpisode % 50 == 0:
                model.save()
                numEpisode += 1

            else:
                numEpisode += 1
            print(endEpisode)
            print(episodeReward)
            print(episodeReward)
            print(episodeReward)
            print(episodeReward)
            episodeReward = 0.0
            endEpisode = False
            prevStraight = 0
            prevNonStraight = 0

        if reward == 0: reward = -0.01
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        # save frame
        # if args.image_folder != '':
        #     timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        #     image_filename = os.path.join(args.image_folder, timestamp)
        #     image.save('{}.jpg'.format(image_filename))

        try:
            image = np.asarray(image)  # from PIL image to numpy array
            image = utils.preprocess(image)  # apply the preprocessing
            image = np.array([image])  # the model expects 4D array

            # predict the steering angle for the image
            steering_angle = float(model.train(image, actionsTaken, [reward], advantagey, learning_rate=1e-4)[0])
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            if numActions < 1000: numActions += 1
            if steering_angle > 0.8:
                steering_angle = 0.75


            if steering_angle < -0.8:
                steering_angle = -0.75


            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            # if speed > 0 and (steering_angle < 0.5 and steering_angle > -0.5):
            #     throttle = 0.4
            # else:
            #     throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2
            throttle = 1.0
            print('Angle:{},  Throttle:{}, Speed:{}, Reward:{}'.format(steering_angle, throttle, speed, reward))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def reset():
    for i in range(80):
        send_control(0.0, 0.0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Remote Driving')
#     parser.add_argument(
#         'model',
#         type=str,
#         help='Path to model h5 file. Model should be on the same path.'
#     )
#     parser.add_argument(
#         'image_folder',
#         type=str,
#         nargs='?',
#         default='',
#         help='Path to image folder. This is where the images from the run will be saved.'
#     )
#     args = parser.parse_args()
#
#     # model = load_model(args.model)
#     model = PPO.PPO([height, width, channels], num_actions=1, action_min=-1.0, action_max=1.0,
#                     epsilon=0.1, value_scale=0.005, entropy_scale=0.01,
#                     model_checkpoint=None, model_name="ppo")
#
#     if args.image_folder != '':
#         print("Creating image folder at {}".format(args.image_folder))
#         if not os.path.exists(args.image_folder):
#             os.makedirs(args.image_folder)
#         else:
#             shutil.rmtree(args.image_folder)
#             os.makedirs(args.image_folder)
#         print("RECORDING THIS RUN ...")
#     else:
#         print("NOT RECORDING THIS RUN ...")
#
#     # wrap Flask application with engineio's middleware
#     app = socketio.Middleware(sio, app)
#
#     # deploy as an eventlet WSGI server
#     eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
