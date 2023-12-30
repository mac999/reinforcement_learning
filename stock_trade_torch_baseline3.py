import yfinance as yf
import numpy as np
import pandas as pd

# 강화 학습용 구글 주가 데이터 준비
ticker = "GOOG"         # Google stock market symbol
data = yf.download(ticker, start="2022-01-01", end="2024-12-31")    # Download the historical price data
print(data.head())      # Display the first few rows of the data
data = data.dropna()    # Remove missing values from the data

# 데이터 정규화
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)
data = pd.DataFrame(data_normalized, columns=data.columns)  

# 학습용, 테스트용 데이터 분할
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# 강화 학습 환경 준비
from stable_baselines3 import PPO
import gym
class TradingEnvironment(gym.Env):
    def __init__(self, data):
        self.data = data
        self.action_space = gym.spaces.Discrete(3)  # 액션은 주식 구입, 판매, 보유 3가지
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(data.columns),))

    def reset(self):
        self.current_step = 0
        self.account_balance = 100000  # 초기 주식 계좌 보유액. Initial account balance
        self.shares_held = 0
        self.net_worth = self.account_balance
        self.max_net_worth = self.account_balance

        return self._next_observation()

    def _next_observation(self):
        return self.data.iloc[self.current_step].values

    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        if self.current_step > len(self.data) - 1:
            self.current_step = 0

        return self._next_observation(), self._get_reward(), self.net_worth, {}

    def _take_action(self, action):
        if action == 0:     # 주식 구매
            self.shares_held += self.account_balance / self.data.iloc[self.current_step].values[0]
            self.account_balance -= self.account_balance
        elif action == 1:   # 주식 판매
            self.account_balance += self.shares_held * self.data.iloc[self.current_step].values[0]
            self.shares_held -= self.shares_held

        self.net_worth = self.account_balance + self.shares_held * self.data.iloc[self.current_step].values[0]

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def _get_reward(self):
        return self.net_worth - self.account_balance

# 강화학습
env = TradingEnvironment(train_data)        # Create the trading environment
model = PPO("MlpPolicy", env, verbose=1)    # Initialize the PPO model
model.learn(total_timesteps=10000)          # Train the model   

# 학습된 모델로 주식 거래 시뮬레이션
def simulate_trading_strategy(model, data):
    env = TradingEnvironment(data)
    obs = env.reset()

    actions = []
    net_worths = []
    for i in range(len(data)):
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
        actions.append(action)
        net_worths.append(env.net_worth)
    return actions, net_worths

import matplotlib.pyplot as plt
actions, net_worths = simulate_trading_strategy(model, test_data)

# Plot the net worth over time
plt.plot(net_worths, color='green') # color = 'green'

# plot scatter of actions(0, 1, 2) on net_worths plot graph, 0 is buy, 1 is sell, 2 is hold. 0 is red, 1 is blue, 2 is black
for i in range(len(actions)):
    if actions[i] == 0:
        plt.scatter(i, net_worths[i], color='red')
    elif actions[i] == 1:
        plt.scatter(i, net_worths[i], color='blue')
    else:
        plt.scatter(i, net_worths[i], color='black')
        
plt.xlabel("Time")
plt.ylabel("Net Worth")
plt.title("Net Worth over Time")
plt.show()
input()