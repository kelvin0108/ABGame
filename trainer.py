import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import a_b_game
import wandb  # 导入wandb

key = input("Please enter wandb authorize key (https://wandb.ai/authorize): ")
wandb.login(key=key)

# 初始化wandb
wandb.init(project="ABGame_DQN", entity="kelvin0108", name="name")

# 超参数
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 500
TARGET_UPDATE = 10
MEMORY_CAPACITY = 10000
LR = 0.001

# 初始化环境
env = a_b_game.ABGame()

# 初始化经验回放池
memory = deque(maxlen=MEMORY_CAPACITY)

# 其他参数
num_actions = 5040  # 0-9 不重複的猜测范围
state_size = 2  # state=(a, b)

# 初始化epsilon值
epsilon = EPSILON_START
steps_done = 0


class DQN(nn.Module):

    def __init__(self, state_size, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 初始化DQN模型和目标模型
policy_net = DQN(state_size, num_actions)
target_net = DQN(state_size, num_actions)

# 复制参数给目标网络
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# 优化器
optimizer = optim.Adam(policy_net.parameters(), lr=LR)


# 经验回放池中的单条经验
class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# 初始化经验回放池
memory = ReplayMemory(MEMORY_CAPACITY)


def select_action(state):
    global steps_done
    eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                    np.exp(-1. * steps_done / EPSILON_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax().item()  # 选择Q值最大的动作
    else:
        return random.randrange(num_actions)  # 随机选择动作



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return  # 不足够数据时不进行优化

    # 从经验回放池中采样
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
        *transitions)

    # 转换为PyTorch张量
    batch_state = torch.stack(
        [s.unsqueeze(0) if len(s.shape) == 1 else s for s in batch_state],
        dim=0)
    batch_next_state = torch.stack([
        s.clone().detach().float() if isinstance(s, torch.Tensor) else
        torch.tensor(s, dtype=torch.float32) for s in batch_next_state
    ],
                                   dim=0)

    batch_state = batch_state.clone().detach().float()
    batch_next_state = batch_next_state.clone().detach().float()
    batch_action = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1)
    batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1)
    batch_done = torch.tensor(batch_done, dtype=torch.float32).view(-1, 1)

    batch_state = batch_state.squeeze(1)  # 去掉第1个维度，使形状变为 [64, 2]

    # 计算当前Q值
    state_action_values = policy_net(batch_state).gather(1, batch_action)

    # 计算下一个状态的最大Q值
    next_state_values = target_net(batch_next_state).max(
        1)[0].detach().unsqueeze(1)
    expected_state_action_values = batch_reward + (GAMMA * next_state_values *
                                                   (1 - batch_done))

    # 计算 MSE Loss
    loss = nn.MSELoss()(state_action_values, expected_state_action_values)

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# 训练过程
num_episodes = 1000
all_rewards = []  # 保存每个回合的奖励
all_losses = []  # 保存每个回合的损失

for i_episode in range(num_episodes):
    print(i_episode)
    # 重置环境
    state = env.reset()
    state = torch.tensor(state[:2], dtype=torch.float32)  # 只取(a, b)

    total_reward = 0  # 每回合的总奖励

    for t in range(100):  # 每局最多进行100次猜测
        # 选择动作
        action = select_action(state)

        # 执行动作，将 action 作为单个参数传递
        guess_nums, reward, done, next_state = env.step(action)

        # 将状态转换为张量并添加批量维度
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # 存储到经验回放池
        memory.push(state, action, reward, next_state, done)

        # 更新状态
        state = next_state

        # 优化模型并计算损失
        loss = optimize_model()

        # 累加奖励
        total_reward += reward

        if done:
            break

    # 记录每回合的奖励和损失
    all_rewards.append(total_reward)
    if loss is not None:
        all_losses.append(loss)

    # 记录到 wandb
    wandb.log({
        "episode": i_episode,
        "total_reward": total_reward,
        "loss": loss
    })

    # 每隔一定回合，更新目标网络
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

# 结束训练
print("训练结束")

# 结束 wandb 记录
wandb.finish()

