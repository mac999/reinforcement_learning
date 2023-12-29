import json, random, time
from collections import deque

import gym
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        # 재생 메모리 버퍼 크기 설정. 큐 자료구조.
        self.replay_buffer = deque(maxlen=40000)

        # 하이퍼파리메터 설정
        self.gamma = 0.99
        self.epsilon = 1.
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.update_rate = 10

        # 메인 및 목표 네트워크 생성
        self.main_network = self.create_nn()
        self.target_network = self.create_nn()

        # 메인 네트워크 가중치와 동일하게 목표 네트워크 가중치 초기화
        self.target_network.set_weights(self.main_network.get_weights())

    def create_nn(self):
        model = Sequential() # 3개 레이어를 가진 신경망 정의
        model.add(Dense(32, activation='relu', input_dim=self.state_size))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def update_target_network(self):
        # 메인 및 목표 네트워크 가중치 동일하게 설정
        self.target_network.set_weights(self.main_network.get_weights())

    def save_experience(self, state, action, reward, next_state, terminal):
        # 재생 메모리에 경험 데이터를 추가함
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def sample_experience_batch(self, batch_size):
        # 재생 메모리에서 경험 데이터들 샘플링. 배치크기만큼 획득
        exp_batch = random.sample(self.replay_buffer, batch_size)

        # s, a, r, s', 종료 정보에 대한 샘플링 데이터 획득
        state_batch = np.array([batch[0] for batch in exp_batch]).reshape(batch_size, self.state_size)
        action_batch = np.array([batch[1] for batch in exp_batch])
        reward_batch = [batch[2] for batch in exp_batch]
        next_state_batch = np.array([batch[3] for batch in exp_batch]).reshape(batch_size, self.state_size)
        terminal_batch = [batch[4] for batch in exp_batch]

        # 생성된 학습용 배치 데이터셋을 튜플 형태로 리턴
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def pick_epsilon_greedy_action(self, state):
        # 확률 ε 로 액션 선택
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)

        # 메인 네트워크에서 주어진 s로 액션 선택
        state = state.reshape((1, self.state_size)) # 배치 차원 형태로 텐서 차원 변환
        q_values = self.main_network.predict(state, verbose=0)
        return np.argmax(q_values[0])  # 최대 보상값을 얻는 액션 인덱스 획득

    def train(self, batch_size):
        # 경험 데이터에서 배치 데이터 샘플링 
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.sample_experience_batch(batch_size)

        # 목표(미래) 네트워크에서 최대 Q 보상값을 가진 액션 획득
        next_q = self.target_network.predict(next_state_batch, verbose=0)
        max_next_q = np.amax(next_q, axis=1)
        # 메인(현재) 네트워크에서 보상값 획득
        q_values = self.main_network.predict(state_batch, verbose=0)

        # 현재 액션의 보상값은 미래 보상값의 감마(감쇠율)을 곱해 할당함
        for i in range(batch_size):
            q_values[i][action_batch[i]] = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * max_next_q[i]

        # 과거 보상치와 현재 보상치의 loss가 없는 방향으로 메인(현재) 네트워크 학습시킴. 
        self.main_network.fit(state_batch, q_values, verbose=0)

if __name__ == '__main__':
    # CartPole 환경 설정
    env = gym.make("CartPole-v1")
    state = env.reset()

    # 환경의 상태, 액션 크기 설정
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 에피소드, 에피소드 당 타임단계, 배치 크기 설정
    num_episodes = 20 # 150
    num_timesteps = 200 # 500
    batch_size = 64
    dqn_agent = DQNAgent(state_size, action_size)
    time_step = 0 # 목표 네트워크 갱신에 사용되는 타임단계값 초기화
    rewards, epsilon_values = list(), list()  # 학습 후 출력위해 보상, 엠실론 값 리스트 초기화

    for ep in range(num_episodes):

        tot_reward = 0

        state = env.reset()

        print(f'\nTraining on EPISODE {ep+1} with epsilon {dqn_agent.epsilon}')
        start = time.time()

        for t in range(num_timesteps):
            time_step += 1

            # 타임스템마다 메인 네트워크 가중치로 목표 네트워크 갱신
            if time_step % dqn_agent.update_rate == 0:
                dqn_agent.update_target_network()

            action = dqn_agent.pick_epsilon_greedy_action(state)  # ε-greedy policy으로 미래 액션 선택
            next_state, reward, terminal, _ = env.step(action)  # 환경에서 액션 실행
            dqn_agent.save_experience(state, action, reward, next_state, terminal)  # 재생 메모리에 경험 저장

            # 현재 상태를 다음 상태로 설정
            state = next_state
            tot_reward += reward

            if terminal:  # 강화학습 종료 조건 만족 시 루프 종료
                print('Episode: ', ep+1, ',' ' terminated with Reward ', tot_reward)
                break

            # 재생 메모리가 가득차면, 딥러닝 모델 학습함
            if len(dqn_agent.replay_buffer) > batch_size:
                dqn_agent.train(batch_size)

        rewards.append(tot_reward)
        epsilon_values.append(dqn_agent.epsilon)

        # 매 에피소드 종료 마다 엠실론 값을 감소시키로도록 갱신
        if dqn_agent.epsilon > dqn_agent.epsilon_min:
            dqn_agent.epsilon *= dqn_agent.epsilon_decay

        # 수행 결과 출력
        elapsed = time.time() - start
        print(f'Time elapsed during EPISODE {ep+1}: {elapsed} seconds = {round(elapsed/60, 3)} minutes')

        # 만약 최근 10개 에피소드 보상이 4990 보다 크면 종료
        if sum(rewards[-10:]) > 4990:
            print('Training stopped because agent has performed a perfect episode in the last 10 episodes')
            break

    # plot rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards of the training')
    plt.show()

    # plot epsilon values
    plt.plot(epsilon_values)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon values of the training')
    plt.show()

    # Save trained model
    dqn_agent.main_network.save('trained_agent.h5')
    print("Trained agent saved in 'trained_agent.h5'")