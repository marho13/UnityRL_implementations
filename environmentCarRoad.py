#import utils
import argparse
import base64
from datetime import datetime
import os
import time
import shutil
import sys
import torch
import os
import keras
from collections import namedtuple

import tensorflow as tf
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
#import PPO
import DQN
from matplotlib.pyplot import imshow
# from matplotlib.pyplot import imsave
from scipy.misc import imsave
#from torchsummary import summary

def lastCheckpoint(listy):
    max = 0
    index = 0
    for a in range(len(listy)):
        if int(listy[a][4:-4]) > max:
            max = int(listy[a][4:-4])
            index = a
    print(max, index)
    return listy[index], max

def removeNonCheck(files):
    for f in range(len(files) - 1, -1, -1):
        if files[f][:4] == "DQN_":
            pass
        else:
            del files[f]
    return files

timestep = 0

sio = socketio.Server()
app = Flask(__name__)

Experience = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

files = (os.listdir())
# print(files)
files = removeNonCheck(files)
files.sort()
# print(files)
try:
    file, maxInd = lastCheckpoint(files)
    # print(file)
    loadingBool = True
except:
    loadingBool = False

numEpisode = 0
endEpisode = False
episodeReward = [0.0]
prevStraight = 0
prevNonStraight = 0
prevDist = 0.0

numActions = 0
num_envs = 1
state = []
next_state = []
next_state1 = []
updateImages = []
offRoadList = []
num1 = 0
num0 = 0

width = 200
height = 66
channels = 3

size = width * height * channels
print(size)
############## Hyperparameters ##############
state_dim = size
action_dim = 15

#Set by us
solved_reward = 230  # stop training if avg_reward > solved_reward
log_interval = 20  # print avg reward in the interval
max_episodes = 50000  # max training episodes
max_timesteps = 3000  # max timesteps in one episode
n_latent_var = 64  # number of variables in hidden layer
update_timestep = 200  # update policy every n timesteps

#Change these first
lr = 0.002
betas = (0.9, 0.999)
gamma = 0.99  # discount factor
K_epochs = 4  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
#############################################
# print(dir(DQN))
memory = DQN.Memory()
model = DQN.DQN(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

# memory = PPO.Memory()
# model = PPO.PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
print("About to load model...")
if loadingBool:
    try:
        model.policy.load_state_dict(torch.load(file))
        model.policy_old.load_state_dict(torch.load(file))
        print("Loaded file: {}", file)
    except:
        print("Couldn't load checkpoint")
else:
    print("Not loading a file, seems the checkpoint isn't there")
    # print(model.policy_net)
# print("Did it work?")
# with tf.device('/gpu:1'):
#     roadDetection = tf.keras.applications.MobileNet(input_shape=(66, 200, 3), weights=None, classes=1)
#     SGD = keras.optimizers.SGD(lr=0.01, decay=1e-6)
#     roadDetection.compile(optimizer='SGD', loss='mean_squared_error', metrics=['accuracy'])

# print(model.policy.state_dict())
# print(model.policy_old.state_dict())
# print(model.optimizer.state_dict())

def reshaper(imgs):
    imgs = np.asarray(imgs)/256
    imgs = np.reshape(imgs, newshape=(160, 320, 3))
    # imsave("{}/{}.png".format(directory, num), imgs)
    return imgs


class Operations:
    def actionTranslation(self, action):#0-6 = -, 7 = 0, 8-14 = +
        action = np.argmax(action)
        dicty = {0:-0.77, 1:-0.66, 2:-0.55, 3:-0.44, 4:-0.33, 5:-0.22, 6:-0.11, 7:0.0, 8:0.11, 9:0.22, 10:0.33, 11:0.44, 12:0.55, 13:0.66, 14:0.77}
        return dicty[action]

    def createImage(self, image):
        output = Image.open(BytesIO(base64.b64decode(image)))
        try:
            output = np.asarray(output)  # from PIL image to numpy array
            output = utils.preprocess(output)  # apply the preprocessing
            output = np.array(output)  # the model expects 4D array
            # output = np.array(output.flatten())
            output = np.asarray(output)/256
            return output

        except Exception as e:
            print(e)
            sys.exit(1)

    def checkReward(self, reward):
        if reward == 0.0:
            return -0.001
        else:
            return reward

o = Operations()
# @sio.on('updated')
# def updated(sid, data):
#     global next_state1
#     next_state1 = o.createImage(data['image'])

@sio.on('telemetry')
def telemetry(sid, data):
    global speed_limit, numActions, actionsTaken, episodeReward, endEpisode, prevStraight, \
        prevNonStraight, numEpisode, o, timestep, roadDetection, prevDist, \
        offRoadList, updateImages, num1, num0, state, next_state

    timestep += 1
    next_state = o.createImage(data['image'])
    if state == []:
        state = o.createImage(data['image'])
    # next_state = reshaper(next_state)
    # state = reshaper(state)
    checkpointStraight = int(data["checkpointStraight"])
    checkpointNonStraight = int(data["checkpointNonStraight"])
    onRoad = (data["onRoad"])

    reward = 0.0

    #perform action in here, then update the reward and image
    action = model.policy_net.act(state)
    action = action.cpu().detach().numpy()
    # if len(action) > 1:
    maxaction = float(np.argmax(action))
    # else:
    #     maxaction = float(action[0])
    translatedaction = o.actionTranslation(int(maxaction))

    send_control(translatedaction, 1.0)

    #Temp way to update reward, so that the asynchronous part of the task does not ruin progress
    if checkpointStraight > prevStraight:
        reward += 2.0

    if checkpointNonStraight > prevNonStraight:
        reward += 4.0

    reward -= 0.001
    done = str(data["resetEnv"])

    if int(checkpointNonStraight)+int(checkpointStraight)> 5:
        reward += 0.01*((int(checkpointNonStraight)+int(checkpointStraight))//5)

    prevStraight = checkpointStraight
    prevNonStraight = checkpointNonStraight

    # if str(onRoad) == "True":
    #     image = reshaper(state, num1, "1")
    #     num1+=1
    #
    # else:
    #     image = reshaper(image, num0, "0")
    #     num0+=1
    #     reward -= 0.01
    if str(onRoad) == "False":
        reward -= 0.01

    if done.lower() != "true":
        memory.push(Experience(torch.tensor(state), torch.tensor([action]), torch.tensor(next_state), torch.tensor([reward])))
    else:
        memory.push(Experience(torch.tensor(state), torch.tensor([action]), None, torch.tensor([reward])))

    episodeReward[-1] += reward

    state = next_state


    # update if its time
    # if memory.can_provide_sample(update_timestep): #create timestep at the start #could store number of timesteps for each episode, which helps show how far you got
    #     print(len(memory))
    #     model.update(memory, update_timestep)

        # memory.clear_memory()

    if done.lower() == "true":
        # update if its time
        if memory.can_provide_sample(update_timestep):  # create timestep at the start #could store number of timesteps for each episode, which helps show how far you got
            model.update(memory, update_timestep)

        numEpisode += 1
        episodeReward[-1] += reward
        print("episode: {}, gave a reward of {}, with the last reward being {} over {} actions".format(len(episodeReward), episodeReward[-1], data["reward"], timestep))
        # print("Random actions taken: {}".format(model.policy_net.randPolicy["Rand"]))
        # print("Policy actions taken: {}".format(model.policy_net.randPolicy["Policy"]))
        # model.policy_net.randPolicy["Rand"] = 0
        # model.policy_net.randPolicy["Policy"] = 0
        episodeReward.append(0.0)
        timestep = 0
        if (len(episodeReward)) % 500 == 0 and len(episodeReward) > 1:
            torch.save(model.policy_net.state_dict(), './DQN_{}.pth'.format(len(episodeReward)+maxInd))
            writer = open("DQNUnityrew.csv", mode="a")
            [writer.write(str(rew) + "\n") for rew in episodeReward[-500:]]
            print("saving!")
    prevStraight = checkpointStraight
    prevNonStraight = checkpointNonStraight


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def reset(sid, data):
    while True:
        if data['resetEnv'] == True:
            break


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4568)), app)

