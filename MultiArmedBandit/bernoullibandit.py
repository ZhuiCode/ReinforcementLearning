import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self, K):
        # 随机生成K个动作的获得奖励的概率值
        self.probs = np.random.uniform(size = K)
        self.best_idx = np.argmax(self.probs) #最大获奖概率的动作
        self.best_prob = self.probs[self.best_idx] #最大获奖概率
        self.K = K 

    def step(self, k):
        '''采取动作看，返回奖励，奖励只有0和1，即时随机数大于该动作的概率值，则为1，否则为0'''
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
        
class Solver:
    def __init__(self,bandit):
        self.bandit = bandit
        self.counts = np.zeros(bandit.K) # 每根拉杆的尝试次数
        self.regret = 0  # 当前步的累积懊悔
        self.actions =  [] # 维护一个列表,记录每一步的动作
        self.regrets = [] # 维护一个列表,记录每一步的累积懊悔
    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
    def run_one_step(self):
        raise NotImplementedError
    
    def run(self, n):
        for _ in range(n):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon = 0.01,init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)
        print('epsilon-贪婪算法初始化时各动作的估计值：', self.estimates,self.estimates.ndim)

    def run_one_step(self):
        if np.random.rand() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k) #得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] +
         1) * (r - self.estimates[k])
        return k

class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit,init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0
        print('epsilon-贪婪算法初始化时各动作的估计值：', self.estimates,self.estimates.ndim)

    def run_one_step(self):
        self.total_count += 1
        if np.random.rand() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k) #得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] +
         1) * (r - self.estimates[k])
        return k


def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.savefig('bernoulli_bandit.png')
    #plt.show()    

class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.coef = coef
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates+ self.coef * np.sqrt(np.log(self.total_count)/(2*(self.counts+1)))
        k = np.argmax(ucb)  # 选择未尝试过的动作
        print("estimates ",np.argmax(self.estimates))
        print("ucb ",k)
       
        r = self.bandit.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


np.random.seed(1)
K = 10
bandit_10_arm  = BernoulliBandit(K)
'''
np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.0001)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])
'''
'''
np.random.seed(1)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])


epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [
    EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
]
epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)

plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)
'''

np.random.seed(1)
coef = 1  # 控制不确定性比重的系数
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(5000)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])