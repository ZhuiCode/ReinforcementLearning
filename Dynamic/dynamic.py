import copy

class CliffWalkingEnv:
    def __init__(self,ncol = 12, nrow = 4 ):
        self.ncol = ncol
        self.nrow = nrow
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        # p:概率
        # next_state:下一个状态
        # reward:奖励
        # done:是否终止
        self.P = self.creatP()
        print("环境初始化完成",self.P)

# 创建转移矩阵,按照4*12的网格世界创建，每个网格内部的动作都是等概率的，且每个网格包括去往不同方向的多个(p, next_state, reward, done)内容。而这个内容规定了所有可能的下一个状态和奖励，一个格子有四个方向的动作，因此可以用一个二维列表来表示
    def creatP(self):
        P = [[[]for j in range(4)]for i in range(self.ncol*self.nrow)]
        change = [[0,-1],[0,1],[-1,0],[1,0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    if i==self.nrow-1 and j > 0:
                        P[i*self.ncol+j][a] = [(1, i*self.ncol+j, 0, True)]
                        continue
                    next_x = min(self.ncol-1,max(0,j+change[a][0]))
                    next_y = min(self.nrow-1,max(0,i+change[a][1]))
                    next_state = next_y*self.ncol+next_x
                    reward = -1
                    done = False
                    if next_y ==self.nrow-1 and next_x > 0:
                        done = True
                        if next_x != self.ncol-1:
                            reward = -100
                    P[i*self.ncol+j ][a] = [(1, next_state, reward, done)]
        return P


class PolicyIteration:
    def __init__(self,env,theta,gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow
        self.pi = [[0.25,0.25,0.25,0.25] for i in range(self.env.ncol * self.env.nrow)]
        self.theta = theta
        self.gamma = gamma

    def policy_evalution(self):
        cnt = 1
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, reward, done = res
                        qsa += p * (reward + self.gamma * self.v[next_state]) * (1 - done)
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print("policy_evalution cnt:", cnt)
    
    def policy_improvement(self):
        for s in range(self.env.ncol * self.env.nrow):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state]) * (1 - done)
                qsa_list.append(qsa)
            max_q = max(qsa_list)
            cntq  = qsa_list.count(max_q)
            self.pi[s] = [1 / cntq if q == max_q else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi
    
    def policy_iteration(self):
        while 1:    
            self.policy_evalution()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if old_pi == new_pi:
                break

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])



class ValueIteration:
    def __init__(self,env,theta,gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow
        self.theta = theta
        self.gamma = gamma
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]

    def value_iteration(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, reward, done = res
                        qsa += p * (reward + self.gamma * self.v[next_state]) * (1 - done)
                    qsa_list.append(qsa)
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print("value_iteration cnt:", cnt)
        self.get_policy()
    
    def get_policy(self):
        for s in range(self.env.ncol * self.env.nrow):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state]) * (1 - done)
                qsa_list.append(qsa)
            max_q = max(qsa_list)
            cntq  = qsa_list.count(max_q)
            self.pi[s] = [1 / cntq if q == max_q else 0 for q in qsa_list]

env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])
