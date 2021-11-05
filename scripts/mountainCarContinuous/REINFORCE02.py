import gym
import numpy as np
import matplotlib.pyplot as plt
import copy

NUM_EPISODES = 1000
LEARNING_RATE = 0.01
GAMMA = 0.9

env = gym.make('CartPole-v0')
nA = env.action_space.n
np.random.seed(13)

w = np.random.rand(4, 2)

episode_rewards = []


def policy(state, w):
    z = state.dot(w)
    exp = np.exp(z)

    return exp/np.sum(exp)


def softmax_grad(softmax):
    s = softmax.reshape(-1, 1) # 2x1

    return np.diagflat(s) - np.dot(s, s.T) # 2x2, gradient of log(soft_max policy)


for e in range(NUM_EPISODES):
    state = env.reset()[None, :]

    grads = []
    rewards = []

    score = 0

    while True:
        probs = policy(state, w)
        action = np.random.choice(nA, p=probs[0])
        next_state, reward, done, _ = env.step(action)
        next_state = next_state[None, :]

        dsoftmax = softmax_grad(probs)[action, :]
        dlog = dsoftmax / probs[0, action]
        grad = state.T.dot(dlog[None, :])

        grads.append(dsoftmax)
        rewards.append(reward)

        score += reward

        state = next_state

        if done:
            break

    for i in range(len(grads)):
        w += LEARNING_RATE * grads[i] * sum([r * (GAMMA ** t) for t, r in enumerate(rewards[i:])])

    episode_rewards.append(score)
    print("EP: " + str(e) + " Score: " + str(score) + "         ", end="\r", flush=False)

plt.plot(np.arange(NUM_EPISODES), episode_rewards)
plt.show()
env.close()