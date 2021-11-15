# not finished!!
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

learning_rate = 0.001
gamma = 0.9
n_rollout = 10

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)

        self.fc_v = nn.Linear(256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def policy(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)

        return prob

    def value_function(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc_v(x)

        return value

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []

        for transition in self.data:
            state, action, reward, next_state, done = transition
            state_list.append(state)
            action_list.append([action])
            reward_list.append([reward])
            next_state_list.append(next_state)
            done_mask = 0.0 if done else 1.0
            done_list.append([done_mask])

        state_batch = torch.tensor(state_list, dtype=torch.float)
        action_batch = torch.tensor(action_list)
        reward_batch = torch.tensor(reward_list, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_list, dtype=torch.float)
        done_batch = torch.tensor(done_list, dtype=torch.float)

        self.data = []

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def train_net(self):
        state, action, reward, next_state, done = self.make_batch()
        td_target = reward + gamma * self.value_function(next_state) * done
        delta = td_target - self.value_function(state)

        pi = self.policy(state, softmax_dim=1)
        pi_a = pi.gather(1, action)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.value_function(state), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


def main():
    env = gym.make('CartPole-v1')
    model = ActorCritic()
    print_interval = 20
    score = 0.0

    num_episode = 10000
    for i_episode in range(num_episode):
        done = False
        state = env.reset()

        while not done:
            for t in range(n_rollout):
                prob = model.policy(torch.from_numpy(state).float())
                m = Categorical(prob)
                action = m.sample().item()
                next_state, reward, done, info = env.step(action)
                model.put_data((state, action, reward, next_state, done))

                state = next_state
                score += reward

                if done:
                    break

            model.train_net()

        if (i_episode+1) % print_interval == 0:
            print(f"# of episode :{i_episode}, avg score : {score/print_interval:.1f}")
            score = 0.0

    env.close()

if __name__=='__main__':
    main()