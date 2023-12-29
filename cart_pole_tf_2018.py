import random, gym, torch
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make('CartPole-v1') # , render_mode='rgb_array') #, render_mode='human')

states = env.observation_space.shape[0]
actions = env.action_space.n
print('States {}, Actions {}'.format(states, actions))

model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(actions, activation='linear'))

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=1e-2
)

agent.compile(Adam(lr=1e-3), metrics=['mae'])
agent.fit(env, nb_steps=50, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=2, visualize=True)
print(np.mean(results.history['episode_reward']))
model.save('trained_model_weights.h5') # agent.save_weights('trained_model_weights.h5')
env.close()

from keras.models import load_model
model = load_model('trained_model_weights.h5')

# predict
new_state = np.array([[0.1, 0.2, 0.3, 0.4]])  # Replace with your actual state
new_state = new_state.reshape(1,1,4)
q_values = model.predict(new_state)
predicted_action = np.argmax(q_values)
print(f'q_value = {q_values}, predict = {predicted_action}')    # predicted_action==0 then left, not then right


'''
episodes = 10
for episodes in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = random.choice([0, 1])
        n_state, reward, done, flag, info = env.step(action)
        score += reward
        env.render()
    
    print('Episode:{} Score:{}'.format(episodes, score))

env.close()
'''