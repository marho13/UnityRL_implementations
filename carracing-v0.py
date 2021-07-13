import gym
from gym.envs.box2d import CarRacing
import scipy.misc
import numpy as np
import random
import PPO
import torch
# import tensorflow as tf
import cv2
import keras
import time

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

def randomGenerator():
    negativeAction = random.randrange(-10, -5, 1)
    positivAction = random.randrange(5, 10, 1)

    actionNum = random.choice([negativeAction, positivAction])
    return actionNum/10

def actionTranslation():
    actionDict = {"w": np.array([[0, 1.0, 0]]), "a": np.array([[-1.0, 0, 0]]), "s": np.array([[0, 0, 1]]),
                  "d": np.array([[1.0, 0, 0]]), "wa": np.array([[-1.0, 1.0, 0]]), "wd": np.array([[1.0, 1.0, 0]])}
    try:
        act = input("Which action to take?").lower()
        return actionDict[act]

    except:
        action = actionTranslation()
        return action

def actionRecreator(action):
    dicty = {0: np.array([-1.0, 1.0, 0]), 1: np.array([-0.75, 1.0, 0]), 2: np.array([-0.5, 1.0, 0]), 3: np.array([-0.25, 1.0, 0]), 4: np.array([0.0, 1.0, 0]), 5: np.array([0.25, 1.0, 0]),
             6: np.array([0.5, 1.0, 0]), 7: np.array([0.75, 1.0, 0]), 8: np.array([1.0, 1.0, 0]), 9: np.array([-1.0, 0.5, 0]), 10: np.array([0.0, 0.5, 0]), 11: np.array([1.0, 0.5, 0]), 12: np.array([0.0, 0.0, 1.0])}
    return dicty[action]

def resizer(img):
    return cv2.resize(img[0], dsize=(224, 224))

# def predict(img):
#     with tf.device('/device:gpu:1'):
#         ans = roadDetection.predict(np.array([img]))
#     return ans

# random.seed(42)

env = gym.make("CarRacing-v0")
env.seed(42)
env = DummyVecEnv([env])
obs = env.reset()

# fileName = "50timesteps/carracing_episode_1200.pth"
dones = False

numEpisodes = 250000
inc = 1200
n_latent_var = 64

lr = 0.002
betas = (0.9, 0.999)
gamma = 0.9  # discount factor
K_epochs = 4  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
max_timesteps = 3000
update_timestep = 50
timestep = 0

shapey = 96*96*3

episodeRew = 0.0

obsflat = np.array([np.array(obs).flatten()])
memory = PPO.Memory()
model = PPO.PPO(shapey, 13, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

# print(env.action_space)

# try:
#     model.policy.load_state_dict(torch.load(fileName))
#     model.policy_old.load_state_dict(torch.load(fileName))
#     print("Loaded file: ", fileName)
# except:
#     print("Couldn't load checkpoint")

# InputLayer = keras.layers.Input(batch_shape=(None, 224, 224, 3))
# roadDetection = keras.applications.MobileNetV2(input_tensor=InputLayer, weights=None, classes=2)
# Nadam = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
# roadDetection.compile(optimizer=Nadam, loss='mean_squared_error', metrics=['accuracy'])
# roadDetection.load_weights("gym.h5")

print(env.observation_space)

actionNumber = randomGenerator()
action = [[0.0, 1.0, 0.0]]
for episode in range(inc, numEpisodes+inc):
    start = time.time()
    while not dones:
        action = model.policy_old.act(obsflat[0], memory)
        action = actionRecreator(np.argmax(action))
        action = np.array([action])

        obs, rewards, dones, info = env.step(action)
        obsflat = np.array([np.array(obs).flatten()])
        # obs = resizer(obs)
        # pred = predict(obs)

        # if np.argmax(pred[0]) == 0:
        #     rewards[0] -= 0.1
        episodeRew += rewards[0]
        # a += 1
        memory.rewards.append(rewards[0])

        timestep += 1
        # env.render()

        if timestep % update_timestep == 0:  # create timestep at the start #could store number of timesteps for each episode, which helps show how far you got
            model.update(memory)

            memory.clear_memory()

    print("Episode number: ", episode, " and used ", timestep, "timesteps, with a reward of: ", episodeRew, "     Time: ", time.time()-start)
    timestep = 0
    dones = False
    episodeRew = 0.0
    if ((episode+1) % 100) == 0:
        torch.save(model.policy.state_dict(), '50timesteps/carracing_episode_{}.pth'.format(episode+1))
        print("Saving")
