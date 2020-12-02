import gym
import wimblepong
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
opponent = wimblepong.SimpleAi(env, 2)
env.set_names('player', opponent.get_name())

desired_frame_size = (1, 100, 100)
def prepro(image):
    s = image / 255.0  # normalize
    s = s[::2, ::2].mean(axis=-1)  # 100x100
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
    def __init__(self, input_image_channels=1, actions=3):
        super(Policy, self).__init__()
        self.input_image_channels = input_image_channels
        self.actions = actions

        self.Conv1 = nn.Conv2d(in_channels=input_image_channels, out_channels=16, kernel_size=5, stride=2)
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

def runepisode(policy, steps=1500, render=False):
    (observation, observation2) = env.reset()  # 200 x 200 x 3 each obs
    curx = prepro(observation)
    prevx = None
    observations, rewards, game_lengths = [], [], 0
    for _ in range(steps):
        if render:
            env.render()
        game_lengths+=1
        x = curx - prevx if prevx is not None else curx #np.zeros(desired_frame_size)
        cv2.imshow('x ', np.reshape(x, (100,100,1)))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        x = torch.tensor(x).to(device)
        action = policy.select_action(x)
        action2 = opponent.get_action()
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

def train(render=False, checkpoint='policygradient.mdl'):
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
    optimizer = optim.RMSprop(policy.parameters(), lr=1e-4)

    episode, max_games,highest_running_winrate = 0, 10000, 0
    scores,game_lengths,game_lengths_Avg,run_avg_plot = [],[],[],[]
    while True and episode <= max_games:
        rewards, observations, game_length = runepisode(policy, render=render)
        print("Total reward for episode {}: {}".format(episode, np.sum(rewards)))
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

        run_avg = np.mean(np.array(scores))
        run_avg_plot.append(run_avg)
        game_length_avg = np.mean(np.array(game_lengths))
        game_lengths_Avg.append(game_length_avg)
        if (run_avg - 0.05) > highest_running_winrate:
            highest_running_winrate = run_avg
            torch.save(policy.state_dict(), "policygradient_"+str(highest_running_winrate)+'.mdl')
            print('Saved policy----------------------------------------------------------------')

        if episode % 100 == 0:
            plt.plot(run_avg_plot, label='run_avg')
            plt.legend()
            plt.show()
            plt.plot(game_lengths_Avg, label = 'game_length_avg')
            plt.legend()
            plt.show()

def test():
    checkpoint = '/home/eugen/Desktop/MyCourses/x_Github_Folders/RL_Project/AC/model.mdl'
    checkpoint = '/home/eugen/Desktop/MyCourses/x_Github_Folders/RL_Project/AC/model_0.64.mdl'

    policy = Policy()
    w = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(w, strict=False)
    policy.eval()
    print("Resumed checkpoint {}".format(checkpoint))
    policy.to(device)

    while True:
        (observation, observation2) = env.reset()  # 200 x 200 x 3 each obs
        curx = prepro(observation)
        prevx = None
        observations, rewards, game_lengths = [], [], 0
        for _ in range(1500):
            env.render()
            x = curx - prevx if prevx is not None else curx  # np.zeros(desired_frame_size)
            cv2.imshow('x ', np.reshape(x, (100,100,1)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            x = torch.tensor(x).to(device)
            action = policy.select_action(x)
            action2 = opponent.get_action()
            (observation, observation2), (r1, r2), done, info = env.step((action, action2))

            prevx = curx
            curx = prepro(observation)

            if done:
                break

#train(render=True)
#test()
