import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# hyperparameters
learning_rate = 1e-4
gamma = 0.99
buffer_size = 10000
batch_size = 32
frame_stack_size = 4

# preprocess game frames
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return np.array(resized, dtype=np.float32) / 255.0

# frame stacking
def stack_frames(stacked_frames, frame, is_new_episode):
    processed_frame = preprocess_frame(frame)
    if is_new_episode:
        stacked_frames = deque([processed_frame for _ in range(frame_stack_size)], maxlen=frame_stack_size)
    else:
        stacked_frames.append(processed_frame)
    stacked_state = np.stack(stacked_frames, axis=0)
    return stacked_state, stacked_frames

# neural network model
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(frame_stack_size, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.out = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        x = self.out(x)
        return x

# initialize environment and model
env = gym.make("ALE/Breakout-v5")
action_size = env.action_space.n
model = DQN(action_size)

# training loop (simplified)
initial_state = env.reset()
frame = initial_state[0]
stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for _ in range(frame_stack_size)], maxlen=frame_stack_size)
state, stacked_frames = stack_frames(stacked_frames, frame, True)

frames = []  # list to store frames for video

for i in range(1000):  # assume 1000 steps for example
    action = np.random.randint(0, action_size)
    next_state, reward, done, info, _ = env.step(action)
    frame = next_state  # extract the actual frame correctly
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, done)
    state = next_state

    # store the frame for video generation
    frames.append(frame)

    if done:
        break

# function to update the plot
def update_figure(frame_number, frames, img_plot):
    img_plot.set_data(frames[frame_number])

# set up the figure for animation
fig, ax = plt.subplots()
img_plot = ax.imshow(frames[0])

ani = FuncAnimation(fig, update_figure, frames=len(frames), fargs=(frames, img_plot), interval=50)
plt.show()
