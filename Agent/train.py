import wimblepong
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import cv2

eps = np.finfo(np.float32).eps.item()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale, env.unwrapped.fps = 1, 30
opponent = wimblepong.SimpleAi(env, 2) #Agent() #
env.set_names('player', opponent.get_name())

desired_frame_size = (1, 50, 50)
def prepro(image):
    s = image / 255.0  # normalize
    s = s[::4, ::4].mean(axis=-1)  # 100x100 (2), 50x50 (4)
    s = np.expand_dims(s, axis=-1)  # 100x100x1
    state = np.reshape(s, desired_frame_size) #(1, 100, 100)
    return state

def discount_rewards(r, gamma=0.99):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r) + eps
    return discounted_r

class Policy(nn.Module):
    def __init__(self, input_image_channels=2, actions=3):
        super(Policy, self).__init__()
        self.input_image_channels = input_image_channels
        self.actions = actions

        self.Conv1 = nn.Conv2d(in_channels=input_image_channels, out_channels=16, kernel_size=5, stride=1)
        self.Batch1 = nn.BatchNorm2d(16)
        self.Conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.Batch2 = nn.BatchNorm2d(32)
        self.Conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.Batch3 = nn.BatchNorm2d(32)

        self.Fc = nn.Linear(in_features=32 * 9 * 9, out_features=actions)
        self.log_probs = []

    def forward(self, frame):
        frame = F.relu(self.Batch1(self.Conv1(frame)))
        frame = F.relu(self.Batch2(self.Conv2(frame)))
        frame = F.relu(self.Batch3(self.Conv3(frame)))
        state = self.Fc(frame.view(frame.size(0), -1))  # fully connected
        output = F.softmax(state, dim=1)
        return output

    def select_action(self, state, evaluation=False):
        state = state.float().unsqueeze(0)
        probs = self(state)
        m = Categorical(probs)
        if evaluation:
            return torch.argmax(m.probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

def observationConstruct(obs1, obs2):
    if obs2 is not None:
        #desired_frame_size = (1, 100, 100)
        #mid = obs1[0][:,6:-6] - obs2[0][:,6:-6]
        #left = obs1[0][:,:6]
        #right = obs1[0][:,-6:]
        #full = np.vstack([left.T, mid.T, right.T]).T
        #s = np.expand_dims(full, axis=-1)  # 100x100x1
        #state = np.reshape(s, desired_frame_size) #(1, 100, 100)
        ch1 = obs1
        ch2 = obs1-obs2
        state = np.concatenate((ch1, ch2), axis=-1) # if prev_state is not None else np.concatenate((cur_state, cur_state), axis=-1) # 100x100x2
    else: 
        state = np.concatenate((obs1, np.zeros(obs1.shape)), axis=-1)
    state = state.reshape(2,50,50)
    return state

def runepisode(policy, steps=1500, render=False):
    (observation, observation2) = env.reset()  # 200 x 200 x 3 each obs
    curx = prepro(observation)
    prevx = None
    observations, rewards, game_lengths = [], [], 0
    observation, observation2 = env.reset()
    for _ in range(steps):
        if render:
            env.render()
        game_lengths+=1
        x = curx - prevx if prevx is not None else curx #np.zeros(desired_frame_size)
        x = observationConstruct(curx, prevx)
        #cv2.imshow('x ', np.reshape(x, (100,100,1))) #uncomment for rendering
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        x = torch.tensor(x).to(device)
        action = policy.select_action(x)
        action2 = opponent.get_action(observation2)#observation2, evaluation=True)
        (observation, observation2), (r1, r2), done, info = env.step((action, action2))
        reward = 0.# float(r1)
        if r1 == 10:
            reward = 1.
        elif r1 == -10:
            reward = -1.
        prevx = curx
        curx = prepro(observation)
        observations.append(x)
        rewards.append(reward)
        if done:
            break

    return rewards, observations, game_lengths

def train(render=False, checkpoint='model.mdl'):
    policy = Policy()
    try:
        policy = Policy()
        w = torch.load(checkpoint, map_location=device)
        policy.load_state_dict(w, strict=False)
        policy.train()
        print("Resumed checkpoint {}".format(checkpoint))
    except:
        print("Created policy network from scratch")
    policy.to(device)
    print("device: {}".format(device))
    optimizer = optim.RMSprop(policy.parameters(), lr=5e-6)#5e-6 instead of 1e-5 instead of -4

    episode, max_games,highest_running_winrate = 0, 100000, 0
    scores,game_lengths,game_lengths_Avg,run_avg_plot = [],[],[],[]
    run_avg=0
    while True and episode <= max_games:
        rewards, observations, game_length = runepisode(policy, render=render)
        print("Total reward for episode {}: {}; Running average: {}".format(episode, np.sum(rewards), run_avg))
        game_lengths.append(game_length)
        drewards = discount_rewards(rewards)
        r_sum = np.sum(rewards)
        scores.append(1) if r_sum > 0 else scores.append(0)

        # Update policy network
        policy_loss = [-log_prob * reward for log_prob, reward in zip(policy.log_probs, drewards)]
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        del policy.log_probs[:]
        episode += 1

        run_avg = np.mean(np.array(scores[-200:]))
        run_avg_plot.append(run_avg)
        game_length_avg = np.mean(np.array(game_lengths))
        game_lengths_Avg.append(game_length_avg)
        if (run_avg - 0.05) > np.mean(np.array(scores[-500:-200])) and False: #highest_running_winrate:
            highest_running_winrate = run_avg
            torch.save(policy.state_dict(), "policygradient_"+str(highest_running_winrate)+'.mdl')
            print('Saved policy----------------------------------------------------------------')
        elif episode%1000 is 0:
            highest_running_winrate = run_avg
            torch.save(policy.state_dict(), "/content/drive/My Drive/RL_project/Models/policygradient_"+str(episode)+"_"+str(highest_running_winrate)+'.mdl')
            print('Saved policy out of order----------------------------------------------------------------')

        if episode % 100 == 0:
            plt.plot(run_avg_plot, label='run_avg')
            plt.legend()
            plt.show()
            plt.plot(game_lengths_Avg, label = 'game_length_avg')
            plt.legend()
            plt.show()
    torch.save(policy.state_dict(), "/content/drive/My Drive/RL_project/Models/policygradient_final.mdl")
    print('Saved policy----------------------------------------------------------------')

def test():
    checkpoint = 'model.mdl'
    policy = Policy()
    w = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(w, strict=False)
    print("Resumed checkpoint {}".format(checkpoint))
    policy.to(device)
    scores = np.array([])
    for i in range(1000):
        print("Starting game {}".format(i))
        (observation, observation2) = env.reset()  # 200 x 200 x 3 each obs
        curx = prepro(observation)
        prevx = None
        observations, rewards, game_lengths = [], [], 0
        for _ in range(1500):
            #with torch.no_grad():
            #env.render()
            x = curx - prevx if prevx is not None else curx  # np.zeros(desired_frame_size)
            x = observationConstruct(curx, prevx)
            #cv2.imshow('x ', np.reshape(x, (100,100,1))) #uncomment for rendering
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            x = torch.tensor(x).to(device)
            action = policy.select_action(x, evaluation=True)
            action2 = opponent.get_action()#observation2)
            (observation, observation2), (r1, r2), done, info = env.step((action, action2))

            prevx = curx
            curx = prepro(observation)

            if done:
              if r1>0:
                scores = np.append(1, scores)
              else:
                scores = np.append(0, scores)
              break
    print("_________________TRAINING DONE___________________")
    for s in scores:
        print(s)
    print("Total winning rate: {}".format(np.mean(scores)))

#test()
train(render=False, checkpoint="model.mdl")
