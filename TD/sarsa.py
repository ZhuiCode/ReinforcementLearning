import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 

class CliffWalkingEnv:
    def __init__(self,ncol = 12, nrow = 4 ):
        self.ncol = ncol
        self.nrow = nrow
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        # p:概率
        # next_state:下一个状态
        # reward:奖励
        # done:是否终止
        self.x = 0
        self.y = self.nrow - 1
    
    def step(self,action):
        change = [[0,-1],[0,1],[-1,0],[1,0]]
        next_x = min(self.ncol-1,max(0,self.x+change[action][0]))
        next_y = min(self.nrow-1,max(0,self.y+change[action][1]))
        next_state = self.y*self.ncol+self.x
        reward = -1
        done = False
        if next_y ==self.nrow-1 and next_x > 0:
            done = True
            if next_x != self.ncol-1:
                reward = -100
        return next_state,reward,done
    
    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y*self.ncol+self.x
    

class SARSA:
    def __init__(self,ncol, nrow,alpha,epsilon,gamma,n_action=4):
        self.Q_table = np.zeros([ncol*nrow,n_action])
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_action = n_action

    def take_action(self,state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_action)
        else:
            return np.argmax(self.Q_table[state])

    def best_action(self,state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state][i] == Q_max:
                a[i] = 1
        return a
    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


ncol = 12
nrow = 4
env = CliffWalkingEnv(ncol, nrow)
np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 0.9
agent = SARSA(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500  # 智能体在环境中运行的序列的数量

return_list = []  # 记录每一条序列的回报
for i in range(10):  # 显示10个进度条
    # tqdm的进度条功能
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
            episode_return = 0
            state = env.reset()
            action = agent.take_action(state)
            done = False
            while not done:
                next_state, reward, done = env.step(action)
                next_action = agent.take_action(next_state)
                episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                agent.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Sarsa on {}'.format('Cliff Walking'))
plt.savefig('sarsa_cliff_walking.png')


def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


action_meaning = ['^', 'v', '<', '>']
print('Sarsa算法最终收敛得到的策略为：')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])