import gym
env = gym.make("CarRacing-v0")
print(env.action_space)

#Torch split = size of minibatch
#Numpy split = number of splits


#Loop through the data (state, action, reward, advantage)
#train_op = optimizer