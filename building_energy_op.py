import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class BatteryEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data):
        super(BatteryEnv, self).__init__()

        # 에너지 가격 데이터(Entso-e platform) 설정
        self.data = data   
        self.max_price = np.max(data)  # Maximum price for normalization
        self.min_price = np.min(data)  # Minimum price
        # 배터리 초기화
        self.start_step = 0     # Starting point in the price data
        self.current_step = 0   # Tracks the current step in the data
        self.soc = 0.5          # Battery starts at SoC 50
        self.reward_range = (-np.inf, np.inf)
        self.profit = 0         # Profit at current time step
        self.reward = 0
        # 그래픽 플롯을 위한 리스트
        self.profit_history = []  # Profit stored over an episode
        self.profit_episode = []  # For use in the render() method
        # 에피소드 종료 플래그
        self.episode_over = False

        self.action_space = spaces.Discrete(3) # 충전%, 방전%, 보유 액션
        self.observation_space = spaces.Box(low=0, high=1, shape=(26,), dtype=np.float32)  # Observations contain the day-ahead price values for the next 24 hours, as well as SoC and Profit

    def _next_observation(self):
        # Get the day-ahead price data points for the next 24 hours and scale to between 0-1. Also get the current SoC and profit
        obs = np.concatenate([self.data[self.start_step:self.start_step+24].flatten() / self.max_price, np.array([self.soc, self.profit / self.max_price])], dtype=np.float32)

        return obs

    def _take_action(self, action):
        # 현재 관찰 데이터 획득(하루 전 에너지 가격으로 설정)
        current_price = self.data[self.current_step]

        if action == 0 and self.soc < 1:
            # Charge the battery by 25%
            self.profit -= current_price
            self.soc += 0.25

        elif action == 1 and self.soc > 0:
            # Discharge the battery by 25%
            self.profit += current_price
            self.soc -= 0.25

        else:   # Hold
            pass

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        self.profit_history.append(self.profit)

        if self.current_step == self.start_step+24:  # 에피소드 종료
            self.reward = self.profit   # Gets the daily profit as a reward
            self.episode_over = True    # Raises the end of the episode flag
            self.profit_episode = self.profit_history.copy()  # Store episode's profit history for render()
            self.profit_history = []    # Reset profit history for next episode
        else:
            self.episode_over = False

        # Perform next observation
        obs = self._next_observation()

        return obs, self.reward, self.episode_over, {'profit': self.profit}

    def reset(self):
        # 초기 상태로 환경변수 리셋
        self.current_step = self.start_step
        self.soc = 0.5      # Reset state of charge to 50%
        self.profit = 0     # Reset profits for next episode
        self.reward = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Plot profit history over an episode
        if self.episode_over:
            plt.plot(self.profit_episode)
            plt.xlabel('Time Step')
            plt.ylabel('Profit')
            plt.title('Profit as a function of Time Step')
            plt.show()

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

# 강화학습 
data = np.loadtxt('France_price_2022.csv', delimiter=',', usecols=[1], skiprows=0, max_rows=24)
env = make_vec_env(lambda: BatteryEnv(data), n_envs=1)  # The algorithms require a vectorized environment to run
model = PPO("MlpPolicy", env=env, verbose=1)
model.learn(total_timesteps=20000, progress_bar=True)
model.save("battery_trading_model")

# 저장된 학습모델 로딩 및 평가
model = PPO.load("battery_trading_model")
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print('Mean reward:', mean_reward)
obs = env.reset()
done = False
profits = []
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    profits.append(info[0]['profit'])
env.close()

# Plot profits over an episode
plt.plot(profits)
plt.xlabel('Time Step')
plt.ylabel('Profit')
plt.title('Profit as a function of Time Step')
plt.show()
input()
