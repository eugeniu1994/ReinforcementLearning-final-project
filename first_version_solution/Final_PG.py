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

desired_frame_size = (1, 100, 100)
epsilon = np.finfo(np.float32).eps.item()

def prepro(image):
    image = image / 255.0  # normalize
    image = image[::2, ::2].mean(axis=-1)  # 100x100
    image = np.expand_dims(image, axis=-1)  # 100x100x1
    state = np.reshape(image, desired_frame_size) #(1, 100, 100)
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
    discounted_r /= np.std(discounted_r) + epsilon
    return discounted_r

class Policy(nn.Module):
    def __init__(self, input_image_channels=1, actions=3):
        super(Policy, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=input_image_channels, out_channels=16, kernel_size=5, stride=2)
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
    def __init__(self):
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = Policy()
        print(policy)

        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.99

        self.log_probs = []
        self.observations = []
        self.rewards = []

    def get_name(self):
        return "Undisputed"

    def reset(self):
        self.log_probs = []
        self.observations = []
        self.rewards = []

    def load_model(self, file = 'model.mdl'):
        w = torch.load(checkpoint, map_location=self.train_device)
        self.policy.load_state_dict(w, strict=False)

        print('Loaded model')

    def get_action(self, s, evaluation=False):
        s = s.float().unsqueeze(0)
        probs = self.policy(s)
        categ = Categorical(probs)
        if evaluation:
            return torch.argmax(categ.probs)
        action = categ.sample()
        self.log_probs.append(categ.log_prob(action))

        return action.item()

    def update_policy(self):
        self.optimizer.zero_grad()
        R = discount_rewards(r = self.rewards, gamma=self.gamma)
        loss = [-log_prob * reward for log_prob, reward in zip(self.log_probs, R)]
        loss = torch.cat(loss).sum()
        loss.backward()
        self.optimizer.step()
        self.reset()

    def store_info(self, s, r):
        self.observations.append(s)
        self.rewards.append(r)

def train(render = False, checkpoint='model.mdl'):
    env = gym.make("WimblepongVisualMultiplayer-v0")
    env.unwrapped.scale, env.unwrapped.fps = 1, 30

    player = Agent()
    opponent = wimblepong.SimpleAi(env, 2)
    env.set_names('Undisputed', opponent.get_name())

    episode, max_games, highest_running_winrate = 0, 10000, 0
    scores, game_lengths, game_lengths_Avg, run_avg_plot = [], [], [], []
    steps = 1500
    while True and episode <= max_games:
        (observation, observation2) = env.reset()  # 200 x 200 x 3 each obs
        cur_state = prepro(observation)
        prev_state = None
        for _ in range(steps):
            if render:
                env.render()
            x = cur_state - prev_state if prev_state is not None else cur_state
            cv2.imshow('x ', np.reshape(x, (100, 100, 1)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            x = torch.tensor(x).to(player.train_device)
            action = player.get_action(x)
            action2 = opponent.get_action()
            (observation, observation2), (r1, r2), done, info = env.step((action, action2))

            reward = float(np.sign(r1))
            prev_state = cur_state
            cur_state = prepro(observation)
            player.store_info(s=x,r=reward)

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
            torch.save(player.policy.state_dict(), "model_" + str(highest_running_winrate) + '.mdl')
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

def test(render = False, checkpoint='model.mdl'):
    env = gym.make("WimblepongVisualMultiplayer-v0")
    env.unwrapped.scale, env.unwrapped.fps = 1, 30

    player = Agent()
    player.load_model(checkpoint)
    opponent = wimblepong.SimpleAi(env, 2)
    env.set_names(player.get_name(), opponent.get_name())

    steps, episode = 0,0
    data,run_avg_plot=[],[]
    nice_plot = []
    while episode<=12:
        (observation, observation2) = env.reset()  # 200 x 200 x 3 each obs
        cur_state = prepro(observation)
        prev_state = None
        while True:
            if render:
                env.render()
            x = cur_state - prev_state if prev_state is not None else cur_state
            cv2.imshow('x ', np.reshape(x, (100, 100, 1)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            x = torch.tensor(x).to(player.train_device)
            action = player.get_action(x)
            action2 = opponent.get_action()
            (observation, observation2), (r1, r2), done, info = env.step((action, action2))

            reward = np.sign(r1)
            prev_state = cur_state
            cur_state = prepro(observation)
            player.store_info(s=x,r=reward)

            if done:
                break

        R_sum = np.sum(player.rewards)
        print("Total reward for episode {}: {}".format(episode, R_sum))
        data.append(1 if R_sum > 0 else 0)
        nice_plot.append(1 if R_sum > 0 else 0)
        run_avg_plot.append(np.mean(data))
        player.reset()
        episode += 1

    plt.figure(figsize=(12, 10))
    plt.plot(data, label='Win rate')
    plt.plot(run_avg_plot, label='Win rate avg', linewidth = 5.0)
    plt.legend()
    plt.show()
    plt.figure(figsize=(12, 10))
    print('Avg win rate:{}'.format(np.average(data)))
    print('nice_plot ', np.shape(nice_plot))
#Training
#train(render=True)

#Testing
checkpoint = '/home/eugen/Desktop/MyCourses/x_Github_Folders/RL_Project/first_version_solution/model_0.65.mdl'
test(render=False, checkpoint = checkpoint)







