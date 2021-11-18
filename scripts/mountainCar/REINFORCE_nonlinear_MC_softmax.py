import gym
from collections import Counter
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from scripts.mountainCar.REINFORCE_nonlinear_MC_softmax_policy import *
import matplotlib

matplotlib.use('TkAgg')
learning_rate = 0.001
gamma = 0.99
hidden = 128
episodes = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)
print(torch.__version__)

env = gym.make('MountainCar-v0')
env._max_episode_steps = 250

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

agent = Policy(observation_space, action_space, hidden, learning_rate, gamma).to(device)
scores = []
losses = []
actions = []

high_score = 0
for i_epi in range(episodes):
    observation = env.reset()
    done = False
    score = 0.0

    while not done:
        prob = agent(torch.from_numpy(observation).float().to(device))
        m = Categorical(prob)
        action = m.sample()
        observation, reward, done, info = env.step(action.item())
        new_reward = agent.get_reward(observation[0])
        agent.put_data((new_reward, prob[action]))
        score += new_reward
        actions.append(action.item())

        if i_epi % 100 == 0:
            pass
            # env.render()

    if score > high_score:
        print("*** New highscore! ***")
        high_score = score

    if (i_epi+1) % 20 == 0:
        print(f"{i_epi+1} episodes done!")

    if done:
        scores.append(score)
    #     if info['TimeLimit.truncated'] == True:
    #         response = 'Step limit maxed.'
    #     print(f"# of episode: {n_epi}, score: {score}")
    #     if observation[0] >= 0.5:
    #         print('Success')
    #         break

    losses.append(agent.train(device))

print('Completed!')
fig, axs = plt.subplots(3)
fig.suptitle('Results.')

counter = Counter(actions)
axs[0].bar(counter.keys(), counter.values())
axs[0].set_title('Actions preferred')

axs[1].plot(scores)
axs[1].set_title('Reward-Episodes')

axs[2].plot(losses)
axs[2].set_title('Loss-Episodes')

plt.show()
env.close()