import os, time, copy, random, logging
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from plotly import tools
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, iplot_mpl
from collections import namedtuple, deque

# 주식 데이터 로딩 
workingDir = r'./Stock_DRL/StockData'
data = pd.read_csv('./AAPl2.csv') # os.path.join(workingDir,r'BAUTO.csv'))
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
print(data.index.min(), data.index.max())
data.head()

# 학습, 테스트 데이터 분할
date_split = '2022-01-01' # '2016-01-01'
train = data[:date_split]
test = data[date_split:]
print(len(train), len(test))

# 학습 데이터 그래프 출력
data = [
    Candlestick(x=train.index, open=train['Open'], high=train['High'], low=train['Low'], close=train['Close'], name='train'),
    Candlestick(x=test.index, open=test['Open'], high=test['High'], low=test['Low'], close=test['Close'], name='test')
]
layout = {
        'shapes': [
            {'x0': date_split, 'x1': date_split, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper', 'line': {'color': 'rgb(0,0,0)', 'width': 1}}
        ],
    'annotations': [
        {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'left', 'text': ' test data'},
        {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'right', 'text': 'train data '}
    ]
}
figure = Figure(data=data, layout=layout)
iplot(figure)

# 주식 마켓 환경 정책
class StockMarketEnv:    
    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()
        
    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_t)]
        reward = 0
        return [self.position_value] + self.history, reward ,self.done# obs
    
    def step(self, act):
        reward = 0
        
        # 액션. 0=보유, 1=buy, 2=sell
        if act == 1: # 구매, 구매 시점 종가 기록
            self.positions.append(self.data.iloc[self.t, :]['Close']) 
        elif act == 2: # 판매 
            if len(self.positions) == 0:    # 구매 기록이 없을 때
                reward = -1
            else:                   
                profits = 0
                for p in self.positions:    # 구매 주식이 있을 때
                    profits += (self.data.iloc[self.t, :]['Close'] - p) # 이익 = 현재 종가 - 구매 주식 가격들
                reward += profits           # 리워드 최대치 계산(이익 총합)
                self.profits += profits
                self.positions = []
        
        self.t += 1 # 관찰 시점 다음 이동
        self.position_value = 0
        for p in self.positions:
            self.position_value += (self.data.iloc[self.t, :]['Close'] - p)
        self.history.pop(0)

        self.history.append(self.data.iloc[self.t, :]['Close'] - self.data.iloc[(self.t-1), :]['Close'])
        
        # 보상치 계산
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        
        return [self.position_value] + self.history, reward, self.done # 관찰, 보상, 에피소드 종료

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):  # 상태 차원수, 액션 차원수, 램덤값, FC1 유닛수, FC2 유닛수
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state)) # 관찰 상태 벡터 입력
        x = F.relu(self.fc2(x))
        return self.fc3(x)          # 액션 출력

# 에이전트 학습 준비
BUFFER_SIZE = int(1e5)  # 재생 메모리 버퍼 크기
MINI_BATCH_SIZE = 64    # 미니 배치 크기
GAMMA = 0.99            # 할인율 감마값
TAU = 1e-3              # 타겟 네트워크 파라메터 소프트 업데이트를 위한 타오 계수
LR = 5e-4               # 학습율
UPDATE_EVERY = 4        # 네트워크 갱신 빈도수 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, seed, DDQN=False):  # 상태 차원 크기, 액션 차원 크기, 램덤값, DDQN 학습 플래그
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network 지역, 타겟 네트워크 준비
        print("state size :",self.state_size)
        print("action size :",self.action_size)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # 재생 메모리 버퍼 준비
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, MINI_BATCH_SIZE, seed)

        # 타임 스템프 인덱스 설정
        self.t_step = 0
        self.DDQN = DDQN
        print("DDQN is now :",self.DDQN)

    def step(self, state, action, reward, next_state, done):
        # 재생 메모리 버퍼에 상태, 액션, 보상, 다음 상태, 종료 플래그 저장
        self.memory.add(state, action, reward, next_state, done)

        # 매 시점별 타임스템프 업데이트
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:    # 샘플수가 충분히 모이면, 재생버퍼에서 저장된 데이터 샘플링하여 학습
            if len(self.memory) > MINI_BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):   # 현재 상태, epsilon-greedy 액션 선택 엡실론값
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)  # 상태값에 대한 로컬 모델 예측 액션값 획득
        self.qnetwork_local.train()                     # 로컬 모델 네트워크 학습

        # Epsilon-greedy 액션 선택
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):    # 관찰 데이터(s, a, r, s', done), 할인율 감마값
        states, actions, rewards, next_states, dones = experiences

        Q_expected = self.qnetwork_local(states).gather(1, actions) # 현재 상태에 대한 지역 네트워크 모델 현재 보상 획득

        # 다음 상태에 대한 타겟 네트워크 모델 미래 보상 획득
        if self.DDQN:   # DDQN 알고리즘
            next_actions = torch.max(self.qnetwork_local(next_states), dim=-1)[1]
            next_actions_t = torch.LongTensor(next_actions).reshape(-1,1).to(
                device=device)
            target_qvals = self.qnetwork_target(next_states)
            Q_targets_next = torch.gather(target_qvals, 1, next_actions_t).detach()
        else:   # DQN 알고리즘
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)  

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        loss = F.mse_loss(Q_expected, Q_targets)    # loss = 현재 보상 - 미래 보상

        self.optimizer.zero_grad()  # loss 최소화
        loss.backward()             # 네트워크 역전파
        self.optimizer.step()       # 한단계 실행

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)    # 목표 학습 모델 파라메터 갱신

    def soft_update(self, local_model, target_model, tau):  # 지역 모델, 목표 모델, 인터폴레이션 비율 타오값
        # Soft update model. θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, MINI_BATCH_SIZE, seed):    # 액션 차원 크기, 재생 메모리 버퍼 크기, 미니배치 크기, 랜덤값
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.MINI_BATCH_SIZE = MINI_BATCH_SIZE
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done): # 경험 및 재생 메모리에 현재 상태 데이터 추가
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):   # 미니배치 학습 데이터 샘플링하기
        experiences = random.sample(self.memory, k=self.MINI_BATCH_SIZE)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

def train_DQN(n_episodes=1500, max_t=300, eps_start=1.0, eps_end=0.01, eps_decay=0.995,pth_file = 'checkpoint.pth'):
    eps = eps_start                       # initialize the score
    scores_window = deque(maxlen=100)   # 스코어 저장 큐
    scores = []
    logging.info('Starting of agent training ......')
    for episode in range(1,n_episodes+1):
        next_state, reward, done = env.reset()  # 환경 초기화
        state = next_state                      # 현재 상태 획득
        score = 0
        for time_step in range(max_t):
            action = agent.act(np.array(state), eps)             # 액션 선택
            next_state, reward, done = env.step(action)         # 액션에 대한 보상 획득
            agent.step(state, action, reward, next_state, done) # 에이전트 액션 실행

            score += reward                                 # 보상 스코어값 누적        
            state = next_state                              # 다음 상태를 현재 상태로 설정
            eps = max(eps_end, eps_decay * eps)             # epsilon-greedy 액션 선택 엡실론값
            if done:
                break
        scores.append(score)
        scores_window.append(score)       # 현재 스코어 저장
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
        if np.mean(scores_window)>=5000:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), pth_file)
            break
    return scores, reward

state_size = 91 # 90일 이전 가격 + 현재 가격 상태 데이터 획득을 위한 크기 변수
action_size = 3 # 보유 = 0 , 구매 = 1 , 판매 = 2
agent = Agent(state_size, action_size, 99, False)
env = StockMarketEnv(train)

start_time = time.time()
episode_count = 50  # 
scores_dqn_base, reward = train_DQN(n_episodes=episode_count, pth_file='checkpoint_dqn.pth')
print("Total run time to achieve average score : %s seconds " % (time.time() - start_time))

# plot results
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_dqn_base)), scores_dqn_base)  # 개별 보상 스코어
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title('DQN Reward Graph over Time for Stock Price')
plt.show()

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(pd.Series(scores_dqn_base).rolling(10).mean())), pd.Series(scores_dqn_base).rolling(10).mean())  # 평균 보상 스코어
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title('DQN Reward Graph over Time for Stock Price')
plt.show()

