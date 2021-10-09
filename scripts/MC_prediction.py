import numpy as np

class MCPrediction(World):
    def __init__(self, grid_dim):
        super.__init__(grid_dim)
        self.reward = -1
        self.pi = [0.8, 0.1, 0.1]
        self.episode = 1
        self.initialize_data()

        # First-Visit MC Prediction
        # T : episode 종료 시점 : int
        # S : t를 index로 갖고, state index를 element로 갖는 state의 집합 : vector(inf by 1)
        # s : state : list(len : 3)
        # N : state counter, episode 중 나타났던 state 별로 횟수를 센다 : vector(100 미만의 자연수 by 1)
        # Gs : returns, 각 episode에서의 각 state에 대한 G 집합 : vector(100 미만의 자연수 by 1)
        # R : reward 집합, target index 외 모두 -1 : vector(100 by 1)
        # V : 각 state에 대한 value 집합 : vector(100 by 1)
        # pi : policy : [0.8, 0.1, 0.1]
        # 0으로 초기화 : N, Gs, V

    def initialize_data(self):
        # self.initialize_S()
        self.initialize_V()
        self.initialize_N()
        self.initialize_R()
        self.initialize_Gs()

    # def initialize_S(self):
    #     pass

    def initialize_V(self):
        pass

    def initialize_N(self):
        pass

    def initialize_R(self):
        pass

    def initialize_Gs(self):
        pass

    def iteration(self):
        while True:
            S, T = self.create_episode()   # episode 완료 -> T, S 확정

            for t in range(T, -1):
                G = gamma * G + R[t+1]
                if S[t] not in S:
                    Gs[S[t]] += G
            episode += 1
        for s in S:
            V[s] = Gs[s] / N[s]

        return V

    def create_episode(self):
        S, T = self.iter_step()
        return S, T
