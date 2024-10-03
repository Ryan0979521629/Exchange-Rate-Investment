# 預測外匯匯率(藉由reinforcement learning)
## 學號:110201532 姓名:范植緯 系級:資工3A
## 使用版本
Python 3.10.11
使用的是jupyter notebooks
The version of the notebook server is: 6.5.4

## 解釋程式碼執行
### 1.函示庫
以下為做這次程式所需要用到的函式庫
```
from time import time
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import tqdm
import os
import pandas as pd
```
### 設定交易環境，遊戲規則的程式碼
```
class Actions(Enum): 
    Buy_NTD = 0
    Buy_AUD = 1
    Buy_CAD = 2
    Buy_EUR = 3
    Buy_GBP = 4
    Buy_HKD = 5
    Buy_JPY = 6
    Buy_SGD = 7
    Buy_USD = 8

class Positions(Enum):
    # 代表持有幣別
    NTD = 0
    AUD = 1
    CAD = 2
    EUR = 3
    GBP = 4
    HKD = 5
    JPY = 6
    SGD = 7
    USD = 8

    def opposite(self,action):
      return Positions(action)
```
上面這一段主要是在用position表示當前位置，action表示移動的幣別
```
class TradingEnv(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self, df, window_size, render_mode=None):
        assert df.ndim == 2
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.global_previous_total_profit = 1  # 定義全局變數
        self.render_mode = render_mode
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float32,
        )

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None

        self._last_position = None
        self._action = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))
        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.NTD
        self._position_history = (self.window_size * [None]) + [self._position]
        self._action = 0
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        self._action = action
        self._truncated = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._truncated = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        trade = False

        if action != self._position.value:
            trade = True

        if trade:
            self._last_position = self._position
            self._position = self._position.opposite(action)
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, self._truncated, info

    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position
        )

    def _get_observation(self):
        return self.signal_features[self._current_tick-self.window_size:self._current_tick]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _render_frame(self):
        self.render()

    def choice_price_col(self,position, buy_or_sell="買入"):
        foreign_price = None
        if position == Positions.AUD:
          foreign_price = self.prices[f'AUD即期{buy_or_sell}'].to_numpy()
        elif position == Positions.CAD:
          foreign_price = self.prices[f'CAD即期{buy_or_sell}'].to_numpy()
        elif position == Positions.EUR:
          foreign_price = self.prices[f'EUR即期{buy_or_sell}'].to_numpy()
        elif position == Positions.GBP:
          foreign_price = self.prices[f'GBP即期{buy_or_sell}'].to_numpy()
        elif position == Positions.HKD:
          foreign_price = self.prices[f'HKD即期{buy_or_sell}'].to_numpy()
        elif position == Positions.JPY:
          foreign_price = self.prices[f'JPY即期{buy_or_sell}'].to_numpy()
        elif position == Positions.SGD:
          foreign_price = self.prices[f'SGD即期{buy_or_sell}'].to_numpy()
        elif position == Positions.USD:
          foreign_price = self.prices[f'USD即期{buy_or_sell}'].to_numpy()
        return foreign_price


    def render(self, mode='human'):

        def _plot_position():
            # 有買賣
            if self._action != self._position.value:

              # 現在不是持有台幣(即有買入外幣)
              if self._position != Positions.NTD:
                # 買入用紅色
                buy_price_col = self.choice_price_col(self._position)
                plt.scatter(self._current_tick, buy_price_col[self._current_tick], color='red')

              # 上一步不是持有台幣(即有賣出外幣)
              if self._last_position != Positions.NTD:
                # 賣出用綠色
                sell_price_col = self.choice_price_col(self._last_position)
                plt.scatter(self._current_tick, sell_price_col[self._current_tick], color='green')
        start_time = time()

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices['AUD即期買入'].to_numpy(), label="AUD")
            plt.plot(self.prices['CAD即期買入'].to_numpy(), label="CAD")
            plt.plot(self.prices['EUR即期買入'].to_numpy(), label="EUR")
            plt.plot(self.prices['GBP即期買入'].to_numpy(), label="GBP")
            plt.plot(self.prices['HKD即期買入'].to_numpy(), label="HKD")
            plt.plot(self.prices['JPY即期買入'].to_numpy(), label="JPY")
            plt.plot(self.prices['SGD即期買入'].to_numpy(), label="SGD")
            plt.plot(self.prices['USD即期買入'].to_numpy(), label="USD")
            # plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.0, 1.0))

            # 起始點標藍色
            plt.scatter(self._current_tick, self.prices['AUD即期買入'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices['CAD即期買入'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices['EUR即期買入'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices['GBP即期買入'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices['HKD即期買入'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices['JPY即期買入'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices['SGD即期買入'].to_numpy()[self._current_tick], color='blue')
            plt.scatter(self._current_tick, self.prices['USD即期買入'].to_numpy()[self._current_tick], color='blue')

        _plot_position()

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        end_time = time()
        process_time = end_time - start_time

        pause_time = (1 / self.metadata['render_fps']) - process_time
        assert pause_time > 0., "High FPS! Try to reduce the 'render_fps' value."

        plt.pause(pause_time)


    def render_all(self, title=None):

        plt.cla()
        plt.plot(self.prices['AUD即期買入'].to_numpy(), label="AUD")
        plt.plot(self.prices['CAD即期買入'].to_numpy(), label="CAD")
        plt.plot(self.prices['EUR即期買入'].to_numpy(), label="EUR")
        plt.plot(self.prices['GBP即期買入'].to_numpy(), label="GBP")
        plt.plot(self.prices['HKD即期買入'].to_numpy(), label="HKD")
        plt.plot(self.prices['JPY即期買入'].to_numpy(), label="JPY")
        plt.plot(self.prices['SGD即期買入'].to_numpy(), label="SGD")
        plt.plot(self.prices['USD即期買入'].to_numpy(), label="USD")
        plt.legend(bbox_to_anchor=(1.0, 1.0))

        last_positions = Positions.NTD

        for i, position in enumerate(self._position_history):
          if position != None:
            # 有買賣
            if position != last_positions:
              # 現在不是持有台幣(即有買入外幣)
              if position != Positions.NTD:
                price_col = self.choice_price_col(position)
                plt.scatter(i, price_col[i], color='red')

              # 上一步不是持有台幣(即有賣出外幣)
              if last_positions != Positions.NTD:
                price_col = self.choice_price_col(last_positions)
                plt.scatter(i, price_col[i], color='green')

              last_positions = self._position_history[i]

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError

```
這裡的TradingEnv是藉由OpenAi Gym庫所創建的自定義環境，主要的function，如reset()是將環境重置、返回初始狀態
step()表示執行動作，更新狀態、計算獎勵等
```
class ForexEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, render_mode=None):
        assert len(frame_bound) == 2
        self.frame_bound = frame_bound
        super().__init__(df, window_size, render_mode)

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):

        if action != self._position.value:
          # 原本非台幣
          if self._position != Positions.NTD:
            # 此處賣出為銀行方，等於投資者的買入
            buy_price_col = self.choice_price_col(self._position, "賣出")
            buy_price = buy_price_col[self._last_trade_tick]

            # 此處買入為銀行方，等於投資者的賣出
            sell_price_col = self.choice_price_col(self._position, "買入")
            sell_price = sell_price_col[self._current_tick]
            self._total_profit = (self._total_profit / buy_price) * sell_price

        # 結束
        if self._truncated:
          if action != Actions.Buy_NTD.value:
            buy_price_col = self.choice_price_col(Positions(action), "賣出")
            buy_price = buy_price_col[self._last_trade_tick]


            sell_price_col = self.choice_price_col(Positions(action), "買入")
            sell_price = sell_price_col[self._current_tick]

            self._total_profit = (self._total_profit / buy_price) * sell_price
    def get_total_profit(self):
        return self._total_profit
```
這一段的ForexEnv是從env創造出來的子類，用來處理外匯交易特有的邏輯，像是這裡的_update_profit就是基由前面已創造的position來找到當前位置的價錢
```
torch.manual_seed(1234)
np.random.seed(1234)
```
這一段設定random seed
### 開始訓練資料
```
train_df = pd.read_csv('./train.csv')
# 處理空值
train_df.replace("-", 0, inplace=True)
train_df
```
這一段將資料讀入
```
def my_calculate_reward(self, action):
    """
    可以修改此function，更改reward設計
    """
    if self._position == Positions.NTD:
      step_reward = 0

    else:
      price_col = self.choice_price_col(self._position)
      current_price = price_col[self._current_tick]
      last_day_price = price_col[self._current_tick-1]
      step_reward = (current_price - last_day_price) / last_day_price
    return step_reward

def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.iloc[start:end, :].filter(like='即期')

    # 這邊可修改想要使用的 feature
    signal_features = env.df.iloc[:,1:].to_numpy()[start:end]
    return prices, signal_features

class MyForexEnv(ForexEnv):
    # 除 _process_data 和 _calculate_reward 外，其餘功能 (class function) 禁止覆寫
    _process_data = my_process_data
    _calculate_reward = my_calculate_reward

# window_size: 能夠看到幾天的資料當作輸入, frame_bound: 想要使用的資料日期區間
# 可修改 frame_bound 來學習不同的環境資料，frame_bound起始值必須>=window_size
# 不可修改此處 window_size 參數 ，最後計算分數時 window_size 也會設為10
env = MyForexEnv(df=train_df, window_size=10, frame_bound=(100, 1000))
```
這一段設定了reward function該怎麼定義的，這個function決定了整個強化式學習是否好的關鍵，_process_data負責將即期的資料抓入
### 檢視環境參數
```
print("env information:")
print("> env.shape:", env.shape)
print("> df.shape:", env.df.shape)
print("> prices.shape:", env.prices.shape)
print("> signal_features.shape:", env.signal_features.shape)

env.reset()
env.render()
```
### 在未訓練得情況下，跑一次
```
observation = env.reset()

while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print(info)
        break

env.render_all()
plt.show()
```
### 搭建network，此時模型輸入是320dim，輸出為上面Action的9種動作之一
```
class QNetwork(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 9)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)
```
會有這部分的神經網路主要是因為這個主題有可能會有太多的state，用傳統的Q-table不太能記錄那麼多，因此有用到神經網路來找近似值。
### 設定agent
```
class QLearningAgent():
    def __init__(self, q_network):
        self.q_network = q_network
        self.optimizer = torch.optim.SGD(q_network.parameters(), lr=0.01)
        self.q_values_history = []  # 新增一個屬性來保存 Q 值歷史
    def learn(self, state, action, reward, next_state, done):
        # 計算 Q-value 的目標值
        gamma=0.95
        q_value_target = reward + (1 - done) * gamma * torch.max(self.q_network(next_state))

        # 計算 Q-value 的預測值
        q_value_pred = self.q_network(state)[action]
        self.q_values_history.append(q_value_pred.item())
        # 計算均方誤差損失
        loss = F.mse_loss(q_value_pred, q_value_target)

        # 更新網路
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def choose_action(self, state, epsilon):
        # 以 epsilon-greedy 方式選擇動作
        if np.random.rand() < epsilon:
            return np.random.choice(env.action_space.n)
        else:
            # 根據 Q-value 選擇最佳動作
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()
    def save_ckpt(self, ckpt_path):
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_param_groups': self.optimizer.param_groups,
        }, ckpt_path)

    def load_ckpt(self, ckpt_path):
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            self.q_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'optimizer_param_groups' in checkpoint:
                self.optimizer.param_groups = checkpoint['optimizer_param_groups']
        else:
            print("Checkpoint file not found, use default settings")
```
這邊我設定了這個Q-learning的agent，並且network就是上面我們的神經網路，在learn的部分中，q_value_target是我們當前的目標Q值，q_value_pred是要去用當前的state和動作去神經網路查預測值，並且複製到list中可供後面去看model是否學習好。
最後去以均方誤差當作損失，藉由梯度下降調整神經網路的參數，在choose action的部分我是以epsilon-greedy來進行，如果低於epsilon那就做exploration，如果高於就做exploitation，其餘的save_ckpt和load_ckpt是將當前模型紀錄或取用的function。
### 建立agent和network
```
q_network = QNetwork(env.shape[0] * env.shape[1])
agent = QLearningAgent(q_network)
```
### train data
```
total_episodes = 400
epsilon_start = 1.0
epsilon_final = 0.1
epsilon_decay_steps = 200
history_total_rewards=[]
CHECKPOINT_PATH = './model.ckpt' # agent model 儲存位置
# 訓練循環
epsilon = epsilon_start
epsilon_decay = (epsilon_start - epsilon_final) / epsilon_decay_steps
agent.q_network.train()  # 訓練前，先確保 network 處在 training 模式

for episode in range(total_episodes):
     # 重置環境並獲取初始狀態
    
    state = torch.FloatTensor(env.reset()[0]).reshape(-1)
    total_reward = 0

    while True:
        action = agent.choose_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        #return observation, step_reward, self._truncated, info
        # 將元組中的所有元素轉換為 NumPy 陣列，同樣檢查並轉換維度
        next_state_elements = [np.array(item) for item in next_state]
        for i in range(len(next_state_elements)):
            if next_state_elements[i].ndim == 0:
                next_state_elements[i] = np.array([next_state_elements[i]])

        next_state = torch.FloatTensor(np.concatenate(next_state_elements))
        agent.learn(state, action, reward, next_state, done)

        total_reward += reward
        state = next_state

        if done:
            break

    # 更新 epsilon
    epsilon = max(epsilon_final, epsilon - epsilon_decay)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")
agent.save_ckpt(CHECKPOINT_PATH)
```
這邊開頭一系列的epsilon是拿來輸入action的，考慮到在一開始train的過程中，我們應該需要較多的去exploration，而到後面train會較需要exploitation，故在for迴圈的最後面有一個更新epsilon，取當前的最大，而在while True當中就是把當前的state跟接下來的state抓下來(需要做一些維度的轉換)，然後再丟到剛剛寫的learn()中，如果做完了，他會在根據total_episodes 來繼續執行，直到找到跑完這個次數，最後在用agent當中有的save_ckpt把資料儲存到指定位置。
### 檢視更新歷程圖
```
modified_list = [x + 1 for x in agent.q_values_history]
plt.plot(modified_list,label="Q-learning")

plt.legend()
plt.title("Total Rewards")
plt.show()
```
由於上面我們在跑迴圈時有順便紀錄了跑完這一回合後的Q，因此就可以拿來看是否有在學習。
### 測試
```
env = MyForexEnv(df=train_df, window_size=10, frame_bound=(10, 800))

network = PolicyGradientNetwork(env.shape[0] * env.shape[1])
test_agent = PolicyGradientAgent(network)

checkpoint_path = './model.ckpt'

test_agent.load_ckpt(checkpoint_path)
test_agent.network.eval()  # 測試前先將 network 切換為 evaluation 模式

observation,_ = env.reset()
while True:
    action, _ = test_agent.sample(observation)
    observation, reward, done, info = env.step(action)
    if done:
      break

env.render_all()
plt.show()
```
有了train完的模型，我們可以替換環境資料區間來進行測試
### 載入TEST資料
```
test_df = pd.read_csv('./test.csv')
test_df.replace("-", 0, inplace=True)
test_df
```
### 看SCORE會跑多少
```
frame_bounds = [(10,100), (10,300), (10,800)]
mean_profit = 0

for frame_bound in frame_bounds:
  env = MyForexEnv(df=test_df, window_size=10, frame_bound=frame_bound)

  network = QNetwork(env.shape[0] * env.shape[1])
  test_agent = QLearningAgent(network)

  checkpoint_path = './model.ckpt'

  test_agent.load_ckpt(checkpoint_path)
  test_agent.q_network.eval()

  # 我們會跑10次取平均X
  for i in range(10):
    observation,_ = env.reset()
    while True:
        state = torch.FloatTensor(observation).reshape(-1)  # 将观察状态转换为 PyTorch 张量
        action = test_agent.choose_action(state, epsilon=0.0)  # 使用 epsilon=0.0 表示完全根据 Q-value 选择动作

        observation, reward, done, info = env.step(action)
        if done:
          break

    # env.render_all()
    # plt.show()

    mean_profit += env.get_total_profit()

mean_profit /= (10 * len(frame_bounds))
print("Score:", mean_profit)
```
這邊最後就是根據題目要求的資料大小，根據當前模型去跑，跑10次在取平均。

