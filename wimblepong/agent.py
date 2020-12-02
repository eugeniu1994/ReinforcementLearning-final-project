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

epsilon = np.finfo(np.float32).eps.item()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def discount_rewards(r, gamma=0.99):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r) + epsilon
    return discounted_r

class Policy(nn.Module):
    def __init__(self, input_image_channels=2, actions=3):
        super(Policy, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=input_image_channels, out_channels=16, kernel_size=5, stride=1)
        self.Batch1 = nn.BatchNorm2d(16)
        self.Conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.Batch2 = nn.BatchNorm2d(32)
        self.Conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.Batch3 = nn.BatchNorm2d(32)
        self.Fc = nn.Linear(in_features=32 * 9 * 9, out_features=actions)

    def forward(self, image):
        image = F.relu(self.Batch1(self.Conv1(image)))
        image = F.relu(self.Batch2(self.Conv2(image)))
        image = F.relu(self.Batch3(self.Conv3(image)))
        state = self.Fc(image.view(image.size(0), -1))
        state = F.softmax(state, dim=1)
        return state

class Agent(object):
    def __init__(self, env=None, load=True):
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = Policy().to(self.train_device)
        if load:
            self.load_model()
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=5e-3)
        self.gamma = 0.99
        self.env = env

        self.log_probs = []
        self.observations = []
        self.rewards = []

        self.curx = None
        self.prevx = None

    def prepro(self, image):
        s = image / 255.0  # normalize
        s = s[::4, ::4].mean(axis=-1)  # 50x50
        s = np.expand_dims(s, axis=-1)  # 50x50x1
        state = np.reshape(s, (1, 50, 50)) #(1, 50, 50)
        return state

    def observationConstruct(self, obs1, obs2):
        #2 channels: 1. observation; 2. observation difference (=velocity)
        if obs2 is not None:
            ch1 = obs1
            ch2 = obs1-obs2
            state = np.concatenate((ch1, ch2), axis=-1)
        else:
            #1st step - no previous image available
            state = np.concatenate((obs1, np.zeros(obs1.shape)), axis=-1)
        state = state.reshape(2,50,50)
        return state

    def get_name(self):
        return "Undisputed"

    def reset(self):
        self.log_probs = []
        self.observations = []
        self.rewards = []

    def load_model(self, file = 'undisputed.mdl'):
        w = torch.load(file, map_location=self.train_device)
        self.policy.load_state_dict(w, strict=False)
        print("Model was loaded")

    def get_action(self, s, evaluation=False):
        #Process the current frame and construct input
        self.curx = self.prepro(s)
        s = self.observationConstruct(self.curx, self.prevx)
        s = torch.tensor(s)

        #Forward the input to the policy
        s = s.float().unsqueeze(0)

        probs = self.policy(s)
        categ = Categorical(probs)
        if evaluation:
            self.prevx = self.curx
            return torch.argmax(categ.probs)
        action = categ.sample()
        self.log_probs.append(categ.log_prob(action))

        self.prevx = self.curx
        return action.item()

    def update_policy(self):
        R = discount_rewards(r = self.rewards, gamma=self.gamma)
        loss = [-log_prob * reward for log_prob, reward in zip(self.log_probs, R)]
        self.optimizer.zero_grad()
        loss = torch.cat(loss).sum()
        loss.backward()
        self.optimizer.step()
        #self.reset()

    def store_info(self, s, r):
        self.observations.append(self.observationConstruct(self.prepro(s), self.prevx))
        self.rewards.append(r)

def train(render = False, checkpoint='undisputed.mdl'):
    env = gym.make("WimblepongVisualMultiplayer-v0")
    env.unwrapped.scale, env.unwrapped.fps = 1, 30

    policy = Policy()
    try:
        policy = Policy()
        w = torch.load(checkpoint, map_location=device)
        policy.load_state_dict(w, strict=False)
        policy.train()
        print("Resumed checkpoint {}".format(checkpoint))
    except:
        print("Starting from scratch")
    player = Agent()
    opponent = Agent()
    env.set_names('player', opponent.get_name())

    episode, max_games, highest_running_winrate = 0, 10000, 0
    scores, game_lengths, game_lengths_Avg, run_avg_plot = [], [], [], []
    steps = 1500
    while True and episode <= max_games:
        player.reset()
        opponent.reset()
        (observation, observation2) = env.reset()  # 200 x 200 x 3 each obs
        for _ in range(steps):
            if render:
                env.render()
            action = player.get_action(observation)
            action2 = opponent.get_action(observation2)
            (observation, observation2), (r1, r2), done, info = env.step((action, action2))

            reward = float(np.sign(r1))
            player.store_info(s=observation,r=reward)

            if done:
                break

        R_sum = np.sum(player.rewards)
        print("Total reward for episode {}: {}".format(episode, R_sum))
        game_lengths.append(len(player.rewards))
        scores.append(1) if R_sum > 0 else scores.append(0)

        # Update policy network
        player.update_policy()
        episode += 1

        if episode > 100:
            run_avg = np.mean(np.array(scores)[-100:])
            game_length_avg = np.mean(np.array(game_lengths)[-100:])
        else:
            run_avg = np.mean(np.array(scores))
            game_length_avg = np.mean(np.array(game_lengths))

        run_avg_plot.append(run_avg)
        game_lengths_Avg.append(game_length_avg)
        if episode % 100 == 0:  # run_avg  > highest_running_winrate:
            highest_running_winrate = run_avg
            print('highest_running_winrate ', highest_running_winrate)
            print("model_" + str(highest_running_winrate) + '.mdl')
            torch.save(policy.state_dict(), "model_" + str(highest_running_winrate) + '.mdl')
            print('Saved policy----------------------------------------------------------------')

        if episode % 100 == 0:
            plt.figure(figsize=(12, 10))
            plt.plot(run_avg_plot, label='avg win rate')
            plt.legend()
            plt.show()
            plt.figure(figsize=(12, 10))
            plt.plot(game_lengths_Avg, label='avg timesteps')
            plt.legend()
            plt.show()




