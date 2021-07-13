import utils
import argparse
import base64
from datetime import datetime
import os
import time
import shutil
import sys
import torch
import os
from torch.autograd import Variable
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import PPOMultiVariate as PPO
from matplotlib.pyplot import imshow
# from matplotlib.pyplot import imsave
from scipy.misc import imsave

def lastCheckpoint(listy):
    maxInd = 0
    index = 0
    for a in range(len(listy)):
        for b in range(len(listy[a])):
            try:
                _ = (int(listy[a][b:-4]))
                if int(listy[a][b:-4]) > maxInd:
                    maxInd = int(listy[a][b:-4])
                    index = a
                break

            except:
                pass
    print(maxInd, index)
    if maxInd > 0:
        return listy[index], maxInd
    return None, 0

def removeNonCheck(files):
    for f in range(len(files) - 1, -1, -1):
        if files[f][:4] == "PPO_":
            pass
        else:
            del files[f]
    return files

timestep = 0

sio = socketio.Server()
app = Flask(__name__)

files = (os.listdir())
print(files)
files = removeNonCheck(files)
files.sort()
print(files)
file, fileMax = lastCheckpoint(files)

# print(file)

numEpisode = 0
endEpisode = False
episodeReward = [0.0]
prevStraight = 0
prevNonStraight = 0
prevDist = 0.0
prevReward = 0.0

episodeExperience = []
numActions = 0
num_envs = 1
image = []
prevImage = []
prevAction = None
imageList = []
updateImages = []
offRoadList = []
num1 = 0
num0 = 0
repeatNum = 16
batchSize = 64

width = 200
height = 66
channels = 3

size = width * height * channels * repeatNum
print(size)
############## Hyperparameters ##############
state_dim = size
action_dim = 1
discount = 0.99

#Set by us
solved_reward = 230  # stop training if avg_reward > solved_reward
log_interval = 20  # print avg reward in the interval
max_episodes = 50000  # max training episodes
max_timesteps = 3000  # max timesteps in one episode
n_latent_var = 64  # number of variables in hidden layer
update_timestep = 4096  # update policy every n timesteps

#Change these first
lr = 0.0003
betas = (0.9, 0.999)
gamma = 0.9  # discount factor
K_epochs = 4  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
#############################################

model = PPO.PPO(state_dim, action_dim, lr, 0.001, gamma, K_epochs, eps_clip)
print("About to load model...{}".format(file))

try:
    model.policy.load_state_dict(torch.load(file))
    model.policy_old.load_state_dict(torch.load(file))
    print("Loaded file: {}", file)
except:
    print("Couldn't load checkpoint")

# print("Did it work?")
# with tf.device('/gpu:1'):
#     roadDetection = tf.keras.applications.MobileNet(input_shape=(66, 200, 3), weights=None, classes=1)
#     SGD = keras.optimizers.SGD(lr=0.01, decay=1e-6)
#     roadDetection.compile(optimizer='SGD', loss='mean_squared_error', metrics=['accuracy'])

# print(model.policy.state_dict())
# print(model.policy_old.state_dict())
# print(model.optimizer.state_dict())

def reshaper(imgs, num, directory):
    imgs = np.asarray(imgs)/256
    imgs = np.reshape(imgs, newshape=(66, 200, 3))
    # imsave("{}/{}.png".format(directory, num), imgs)
    return imgs

class Operations:
    def actionTranslation(self, action):#0-6 = -, 7 = 0, 8-14 = +
        dicty = {0:-0.77, 1:-0.66, 2:-0.55, 3:-0.44, 4:-0.33, 5:-0.22, 6:-0.11, 7:0.0, 8:0.11, 9:0.22, 10:0.33, 11:0.44, 12:0.55, 13:0.66, 14:0.77}
        return dicty[action]

    def argMax(self, action):
        return np.argmax(action)

    #Adds Image to the front, removing from the back of the array
    def addImage(self, imageList, image):
        #Insert to the front, delete back image
        imageList = np.insert(imageList, 0, image)
        del imageList[size/repeatNum:]
        return imageList

    #Takes the image and repeats or adds it in the correct location
    def createExperience(self, imageList, image):
        if imageList == []:
            output = np.repeat(image, repeatNum)
        else:
            output = self.addImage(imageList, image)
        return output, imageList


    def createImage(self, image):
        output = Image.open(BytesIO(base64.b64decode(image)))
        try:
            output = np.asarray(output)  # from PIL image to numpy array
            output = utils.preprocess(output)  # apply the preprocessing
            output = np.array([output])  # the model expects 4D array
            output = output.flatten()
            return output/255.0

        except Exception as e:
            print(e)
            sys.exit(1)

    def checkReward(self, reward):
        if reward == 0.0:
            return -0.001
        else:
            return reward

o = Operations()

@sio.on('telemetry')
def telemetry(sid, data):
    global speed_limit, numActions, actionsTaken, episodeReward, endEpisode, prevStraight, \
        prevNonStraight, numEpisode, o, timestep, roadDetection, prevDist, \
        offRoadList, updateImages, num1, num0, imageList, prevImage, prevAction, prevReward, memory

    timestep += 1
    image = o.createImage(data['image'])

    checkpointStraight = int(data["checkpointStraight"])
    checkpointNonStraight = int(data["checkpointNonStraight"])
    onRoad = (data["onRoad"])
    done = str(data["resetEnv"])

    # We need to create the image buffers, or add images to the buffer
    inputImage, imageList = o.createExperience(imageList, image)



    if prevImage == []:
        prevImage = inputImage

    if done.lower() == "true":
        done = True
        if timestep >= update_timestep:
            timestep, episodeReward, numEpisode = train(numEpisode, prevReward, model, timestep, episodeReward, data["reward"], prevStraight, prevNonStraight)
            reset()
            time.sleep(1)
            send_control(0.0, 0.0)
        else:
            print(episodeReward[-1], timestep)
            reset()
            time.sleep(1)
            send_control(0.0, 0.0)
    else:
        done = False

    model.buffer.rewards.append(prevReward)
    model.buffer.is_terminals.append(done)

    reward = 0.0

    #perform action in here, then update the reward and image
    action = model.select_action(inputImage)
    # actionMax = np.argmax(action)
    # action = o.actionTranslation(actionMax)
    send_control(action[0], 1.0)

    #Temp way to update reward, so that the asynchronous part of the task does not ruin progress
    if checkpointStraight > prevStraight:
        reward += 2.0

    if checkpointNonStraight > prevNonStraight:
        reward += 7.0

    reward -= 0.01

    reward += onRoadFunc(onRoad, int(checkpointStraight)+int(checkpointNonStraight))

    # Saving reward:
    # memory.onRoads.append(onRoad)
    # memory.rewards.append(reward)
    # memory.images.append(image)

    episodeReward[-1] += reward
    prevReward = reward
    prevStraight = checkpointStraight
    prevNonStraight = checkpointNonStraight

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    reset()
    send_control(0, 0)
    send_control(0.0, 0.0)


def reset():
    ready()
    send_control(0.0, 0.0)
    send_control(0.0, 0.0)
    send_control(0.0, 0.0)

def ready():
    sio.emit("ready",
             data={})
    send_control(0.0, 0.0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

def train(numEpisode, reward, model, timestep, episodeReward, datarew, prevStraight, prevNon):
    numEpisode += 1
    episodeReward[-1] += reward
    print("episode: {}, gave a reward of {}, with the last reward being {} over {} actions with {}".format(
        len(episodeReward), episodeReward[-1], datarew, timestep, (prevStraight + prevNon)))

    model.update()

    episodeReward.append(0.0)
    timestep = 0

    if ((len(episodeReward))+fileMax) % 50 == 0 and len(episodeReward)>1:
        model.save('./PPO_road_{}.pth'.format(len(episodeReward) + fileMax))
        writer = open("rewards.txt", mode="a")
        [writer.write(str(rew) + "\n") for rew in episodeReward[-50:]]
        print("saving!")

    return timestep, episodeReward, numEpisode

def onRoadFunc(onRoad, numCheckpoints):
    if str(onRoad) == "True":
        return 0.0

    else:
        if numCheckpoints < 5:
            return -0.01
        else:
            return -0.001

if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4568)), app)

