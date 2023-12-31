import random, json, gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_STEPS = 20000
INITIAL_ACCOUNT_BALANCE = 10000

class StockTradingEnv(gym.Env):
	def __init__(self, df):
		super(StockTradingEnv, self).__init__()

		self.df = df
		self.reward_range = (0, MAX_ACCOUNT_BALANCE)
		self.current_step = 0

		# 액션은 구입, 판매, 보유
		self.action_space = spaces.Box(
			low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

		# 주식 데이터 Open, High, Low, Close, Volumn와 
		# 계좌잔액, 최대순자산, 보유주식, 비용, 총주식보유수, 총주식판매수 관찰함
		self.observation_space = spaces.Box(
			low=0, high=1, shape=(6, 6), dtype=np.float16)

	def _next_observation(self):
		# 5일간 데이터 획득해 0-1 사이로 정규화
		frame_rows = 6
		frame = np.array([self.df.iloc[self.current_step: self.current_step + frame_rows, 3].values / MAX_SHARE_PRICE,
			self.df.iloc[self.current_step: self.current_step + frame_rows, 4].values / MAX_SHARE_PRICE,
			self.df.iloc[self.current_step: self.current_step + frame_rows, 5].values / MAX_SHARE_PRICE,
			self.df.iloc[self.current_step: self.current_step + frame_rows, 6].values / MAX_SHARE_PRICE,
			self.df.iloc[self.current_step: self.current_step + frame_rows, 7].values / MAX_NUM_SHARES])

		# 관찰할 주식 데이터에 계좌잔액, 최대순자산, 보유주식, 비용, 총주식보유수, 총주식판매수 추가
		obs1 = np.append(frame, [[
			self.balance / MAX_ACCOUNT_BALANCE,
			self.max_net_worth / MAX_ACCOUNT_BALANCE,
			self.shares_held / MAX_NUM_SHARES,
			self.cost_basis / MAX_SHARE_PRICE,
			self.total_shares_sold / MAX_NUM_SHARES,
			self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
		]], axis=0)
		obs =obs1
		return obs

	def _take_action(self, action):
		# 타임스템 내에서 임의 가격으로 현재가를 생성 # current_price = random.uniform(self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])
		current_price = random.uniform(self.df.iloc[self.current_step, 3], self.df.iloc[self.current_step, 6])
		action_type = action[0]		# 액션
		amount = action[1]			# 비율

		if action_type < 1:
			# Buy amount % of balance in shares
			total_possible = int(self.balance / current_price)
			shares_bought = int(total_possible * amount)
			prev_cost = self.cost_basis * self.shares_held
			additional_cost = shares_bought * current_price

			self.balance -= additional_cost
			self.cost_basis = (
				prev_cost + additional_cost) / (self.shares_held + shares_bought)
			self.shares_held += shares_bought

		elif action_type < 2:
			# Sell amount % of shares held
			shares_sold = int(self.shares_held * amount)
			self.balance += shares_sold * current_price
			self.shares_held -= shares_sold
			self.total_shares_sold += shares_sold
			self.total_sales_value += shares_sold * current_price

		self.net_worth = self.balance + self.shares_held * current_price

		if self.net_worth > self.max_net_worth:
			self.max_net_worth = self.net_worth

		if self.shares_held == 0:
			self.cost_basis = 0

	def step(self, action):
		# 정책 내에서 액션 실행
		self._take_action(action)
		self.current_step += 1
		if self.current_step > len(self.df.iloc[:, 3].values) - 6:
			self.current_step = 0

		delay_modifier = (self.current_step / MAX_STEPS)
		reward = self.balance * delay_modifier	# 보상은 잔액의 지연 보상
		done = self.net_worth <= 0
		obs = self._next_observation()

		return obs, reward, done, {'net_worth':self.net_worth}

	def reset(self):
		# 환경 초기상태로 리셋
		self.balance = INITIAL_ACCOUNT_BALANCE
		self.net_worth = INITIAL_ACCOUNT_BALANCE
		self.max_net_worth = INITIAL_ACCOUNT_BALANCE
		self.shares_held = 0
		self.cost_basis = 0
		self.total_shares_sold = 0
		self.total_sales_value = 0
		self.current_step = random.randint(0, len(self.df.iloc[:, 3].values) - 6)

		return self._next_observation()

	def render(self, mode='human', close=False):
		# Render the environment to the screen
		profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

		print(f'Step: {self.current_step}')
		print(f'Balance: {self.balance}')
		print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
		print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
		print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
		print(f'Profit: {profit}')


import gym
import json
import datetime as dt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import pandas as pd

df = pd.read_csv('./tutorial/AAPL.csv')
df = df.sort_values('Date')
df.dropna(inplace=True)
df = df.sort_values('Date')
df = df.reset_index()

train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:].copy()

# 학습하기
env = DummyVecEnv([lambda: StockTradingEnv(train_data)])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=80000)

# 학습된 모델로 주식 거래 시뮬레이션
env2 = DummyVecEnv([lambda: StockTradingEnv(test_data)])
stocks = []
actions = []
rewards = []
net_worths = []
obs = env2.reset()
for i in range(len(test_data)):
	action, _states = model.predict(obs)
	obs, reward, done, info = env2.step(action)
	# stocks.append(obs[0])
	actions.append(action)
	rewards.append(reward)
	net_worths.append(info[0]['net_worth']) # net_worths.append(env.net_worth)
	env2.render()

import matplotlib.pyplot as plt
plt.plot(net_worths)
for i in range(len(actions)):
	if i % 30 != 0:
		continue
	if actions[i][0][0] < 1:
		plt.scatter(i, net_worths[i], color='red')
	elif actions[i][0][0] < 2:
		plt.scatter(i, net_worths[i], color='blue')
	else:
		plt.scatter(i, net_worths[i], color='black')

plt.xlabel('Time Step')
plt.ylabel('Net worth')
plt.title('Reward as a function of Time Step')
plt.show()
input()
