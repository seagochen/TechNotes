@[toc]

# 1. 什么是强化学习（Reinforcement Learning）

## 1.1. 我们从一个小游戏开始

如果你是第一次听说「强化学习」，那么你可来对地方了。为了理解什么是强化学习，我们先从一个简单的游戏说起，这就是吃豆人。

![在这里插入图片描述](https://img-blog.csdnimg.cn/e5f912cdb8904b10a5e611007e679420.png#pic_center)

吃豆人的规则很简单，作为玩家，你要做的就是操作的像“饼”一样的「吃豆人」，让它躲开游戏里追着你的那些NPC的同时，把路上能遇到的所有的豆子都吃掉，只要在规定时间内吃掉所有的豆子你就赢了；

如果被NPC抓到丢了三条命，或者超过规定时间而没有吃掉全都的豆子你就输了。

在这个过程中，我们的大脑是怎么理解「吃豆人」这个游戏的呢？给予我们快乐的是躲开全部NPC，并且在规定时间内吃掉所有的豆子。而厌恶的是在这个过程中被NPC追上，或者超时。所以我们的大脑会在游戏过程中思考，在当前的环境下做什么决策能最大程度上争取到利益。

那么，你可能会问，**这跟强化学习有什么联系呢？**

现在我们再来聊一个经典的例子。

大概在2009年，东京大学做了一项针对灵长类动物的智力测验，科学家们找来一只黑猩猩，在黑猩猩面前摆了一台触屏显示器，并且随机地显示数字。

黑猩猩只要按照数字的顺序完成游戏，就能得到糖果、点心。

如果选错数字就要重来游戏，如果失败三次以上当天就得不到点心。就这样，实验从简单到困难，屏幕上的数字越来越多，到黑猩猩完全掌握数字规律后，研究人员又开始减少数字驻留时间，以达到测量黑猩猩的瞬间记忆能力的目的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/344db1e85fbd4d9d87a6d45b07181daa.webp#pic_center)
这里一个有个很有意思的地方在于实验的奖惩机制，是怎么让黑猩猩理解数字符号的。要知道自印度人发明数字后，数字这个概念仅限于人类可以理解。黑猩猩是怎么理解「1」之后是「2」这个概念。

想要解释这个问题，就要引入行为心理学里——「正向强化」这个概念。也就是说，当某种行为过程与某种奖励正相关后，大脑会在这种刺激之下建立起相应的规则。黑猩猩或许不理解符号「1」的含义，但是一定知道只要按着「1」，「2」，「3」这样的顺序玩游戏，就能得到小点心。


**什么是「强化学习」？**

通过以上的例子，我们揭示了「强化学习（Reinforcement Learning）」的本质，即通过某些方式方法，让模型理解规则的过程；由于这样的规则在一遍遍「反向强化」和「正向强化」的刺激下，使得模型找到一个针对特定问题最合适的解决方案。

现在，为了让你更好的理解「强化学习」过程，我们来做个简单的机器猩猩，让它也来试着理解数字之间的关系。

## 1.2. 先从理解游戏规则开始
首先，对于计算机来说，它没有像人一样的感知和记忆能力，所以我们需要设计某种特定形式的数据表，去记录模型每一次决策过程的情况；这个决策表，最好能分阶段记录决策表现，这样我们的模型便能在当下状态选出最好的行为决策。

通常，要实现这样的目的，我们要设计一个名为「Q table」的表。在这个例子中，我们需要让模型掌握【1，2，3，4，5】这几个数字的顺序关系，在不考虑数字被选取后消失这个规则的前提下，在每一次的决策过程中，模型都会面临五种【数字1，数字2， 数字3，数字4，数字5】行为的选择，以及最多五轮状态。

因此，这个表就会是下面这个样子的：


STEP |  # 1 | # 2 | # 3 | # 4 | # 5
---------|---------|---------|---------|---------|----------- 
STEP 1 | 0 | 0 | 0 | 0 | 0
STEP 2 | 0 | 0 | 0 | 0 | 0
STEP 3 | 0 | 0 | 0 | 0 | 0
STEP 4 | 0 | 0 | 0 | 0 | 0
STEP 5 | 0 | 0 | 0 | 0 | 0

这个表现在所有的值都被设为0，这表明规则还未确立。

我们的目的是让表最终表现为下面这个样子：

STEP |  # 1 | # 2 | # 3 | # 4 | # 5
---------|---------|---------|---------|---------|----------- 
STEP 1 | 17.75 | 0 | 0 | 0 | 0 
STEP 2 | 0 | 16.1 | 0 | 0 | 0
STEP 3 |  0 | 0 | 14.2 | 0 | 0
STEP 4 |  0 | 0 | 0 | 12.9 | 0
STEP 5 | 0 | 0 | 0 | 0  | 12.3 

它表明在状态1时，选择数字最有可能是正确的，状态2时，选择数字2最有可能是正确的，依此到最后一个状态5。

那么我们有什么办法可以达成上面这个目标呢？


# 2. 最简单的强化学习算法——Q Learning

为了达成上面这个目标，我们需要引入一个名为「Q Learning」的算法，实现「Q Learning」的核心算法叫「Bellman Equation」，这是一种基于马尔可夫决策过程的搜索算法。关于该方程的一些证明过程，有兴趣的朋友可以看看这篇论文 [《论文研读 —— 3. Convergence of Q-learning: a simple proof》](https://blog.csdn.net/poisonchry/article/details/122234414)。

在这个章节里，我们不再解释Bellman方程的证明，而是着重算法的实现。

![在这里插入图片描述](https://img-blog.csdnimg.cn/6b176e75fc3d443085621274462ec47e.png#pic_center)

在一些论文里，「Bellman 方程」也称为Q函数，函数里的 $Q(s_t, a_t)$ 表示当前动作状态的期望，当我们有了「Q表」后，$Q(s_t, a_t)$就可以简单的等价于对「Q表」的查表和更新过程；

$\alpha$ 又称「学习率」，在大多数深度学习相关的文献里它又用 $\lambda$ 表示，$\gamma$ 在这里一般称为「折扣率」，目的在于减少远期决策对当前决策的影响权重。

这里，我个人觉得稍微复杂点的可能有两点，一个是「奖励函数 $r_t$」，另一个则是最佳未来估计策略 $\max Q(s_{t+1}, a)$。

这里，我们分开进行讨论。

## 2.1. 奖励函数
用一句话以概之，就是做对了奖励，做错了惩罚，完成游戏给小点心。所以这个过程，如果想省事一点，那么就做成一张表的形式，然后把游戏规则的奖励用查表的形式来表示，于是：

Reward |  # 1 | # 2 | # 3 | # 4 | # 5
---------|---------|---------|---------|---------|----------- 
State 1| 10 | -1 | -1 | -1 | -1
State 2| -1 | 10 | -1 | -1 | -1 
State 3| -1 | -1 | 10 | -1 | -1 
State 4| -1 | -1 | -1 | 10 | -1 
State 5| -1 | -1 | -1 | -1 | 10

这个奖励表我们在程序跑起来后，不会做任何修改；其目的就是让模型在当下状态作出最优解。

## 2.2. 最佳未来估计策略

尽管我们的程序在执行过程中，更多的会关心当下的决策期望，但是我也希望说它能适当的坚固长期决策。换句话说，我们不仅要让它能够「朝四」，也要能适当的考虑「暮三」。最佳未来估计就是描述这样的一个过程。

也就是说，如果我们的模型在当前的环境 $s_1$ 的情况下，决定执行策略 $a_1$ 后，我们也希望它能适当考虑在接下来的环境 $s_2$ 里采用的策略组 $[a_{(S2, 1)}, a_{(S2, 2)}, a_{(S2, 3)}, \cdots a_{S2, n}]$ 能得到的最大奖励期望。

你也可以把这个过程理解为游戏里的「插眼」，我们在打游戏时如果有「战争迷雾」的情况下，通常为了预警或者监视，通常会有计划的在一些位置「插眼」，这样当敌人袭来，或者有什么动作时，我们就能提前做预警。

「最佳未来估计」就是这样的一个策略，同时你或许会注意到，最佳未来估计前面有个「折扣率」的玩意，这是一个范围是 $[0, 1]$ 的值，不同的大小，会带来不同的效果。

当折扣率 $\gamma$ 值越大，模型会倾向于远期策略，反之则模型会倾向于近期策略。

## 2.3. 游戏过程

Q-Learning 过程如果用伪码表示，就是下面这个执行过程：

```
Initialize Q(s, a) arbitarily
Repeat (for each episode):
	Initialize s
	Repeat (for each step of episode):
		Choose a from s using policy derived from Q (e.g. greedy)
		Take action a, observe r, s'
		Q(s, a) = Q(s, a) + alpha * [r + gamma * max(Q(s+1, a)) ]
		s = s'
	until s is terminal
```

基本上在弄明白上面的概念后你已经可以手写一个「Q learning」的实现算法。

# 3. 代码实现
方便起见，我还是用Python，在弄懂计算原理后你用Java或者其他什么语言都很容易复现的。


## 3.1. Q Table
首先，我们要实现一个Q-Table，这是我们程序采取决策所依赖的最关键的基础组件。

```python
import numpy as np

q_table = np.zeros((5, 5), dtype=np.int32) # 创建二维的q-table，
										   # 行作为action，列作为state
```

## 3.2. Rule Table
我们为了方便起见，可以把奖惩做成一张表，这样就可以通过查询表值得到程序执行某个指令得到的奖励情况

```python
q_rule = np.full_like(q_table, -1, dtype=np.float32) # 它的大小跟 q table 一样
```

然后修改一些值，使得模型在做对选择后得到正确的奖励

```python
for i in range(5)
	q_table[i, i] = 10
```

接下来我们要封装一下 q_rule，当程序作出错误的选择后，跳出当前的循环，这样程序只能按照 1，2，3，4，5的顺序执行指令

```python
def derive_q_rule(state_idx, action_idx):
	rule_val = q_table[state_idx, action_idx]
	if rule_val == -1:
		return False, rule_val
	else:
		return True, rule_val
```

## 3.3. 计算期望
我们需要让计算机能够计算出当前决策的收益期望，也就是计算更新后的「Q Table」，所以需要这样的一个函数

```python
def derive_updated_q_val(state_idx, action_idx, alpha, gamma):
    # derive the q-value from q table
    q_val = q_table[state_idx, action_idx]

    # derive the rule value from rule table
    ret, rule_val = derive_rule_val(state_idx, action_idx)

    # compute the updated q-value
    if state_idx == 4:
        updated_q_val = (1 - alpha) * q_val + alpha * (rule_val + gamma * np.max(q_table[state_idx]))
    else:
        updated_q_val = (1 - alpha) * q_val + alpha * (rule_val + gamma * np.max(q_table[state_idx + 1]))

    # return the updated q-value
    return ret, updated_q_val
```

这里稍微注意一点，就是当执行到第5个状态，由于它已经是最终状态了，所以我们仅查找该状态内收益最大的执行动作。


## 3.3. 环境交互

强化学习与普通的深度学习不一样的是，强化学习所处理的问题是动态的，也就是说它在每时每刻遇到的问题是不一样的，模型或者说代理（机器人）要根据我们给出的「Q Table」作出当下最合适的决策，所以有：

```python

def choose_state_action(state_idx, epsilon, alpha, gamma):
    # choose action
    # if random number less than epsilon, choose random action
    # else choose the action with the highest q-value
    if  np.random.random() < epsilon:
        action_idx = np.random.randint(0, 5)
    else:
        action_idx = np.argmax(q_table[state_idx])

    # derive updated q value
    ret, updated_q_val = derive_updated_q_val(state_idx, action_idx, alpha, gamma)

    # update q table
    if ret:
        q_table[state_idx, action_idx] = updated_q_val

    # return the ret
    return ret
```

我们给模型加入了一定的随机性，这样它会随机地尝试其他可能的策略，以便找出最优的解

## 3.4. 完整的模型
现在，我们把上面的这些模块组装在一起，看看完整的代码是什么样子的
```python
import numpy as np

# create q table
q_table = np.zeros((5, 5), dtype=np.float32)

# create rule table 
rule_table = np.full_like(q_table, -1.0)

# set some col and row to 10, and (4, 4) to 100
for i in range(5):
    rule_table[i, i] = 10


# derive rule table with index
def derive_rule_val(state_idx, action_idx):
    rule_val = rule_table[state_idx, action_idx]
    if rule_val == -1:
        return False, rule_val
    else:
        return True, rule_val


# environment function
def derive_updated_q_val(state_idx, action_idx, alpha, gamma):
    # derive the q-value from q table
    q_val = q_table[state_idx, action_idx]

    # derive the rule value from rule table
    ret, rule_val = derive_rule_val(state_idx, action_idx)

    # compute the updated q-value
    if state_idx == 4:
        updated_q_val = (1 - alpha) * q_val + alpha * (rule_val + gamma * np.max(q_table[state_idx]))
    else:
        updated_q_val = (1 - alpha) * q_val + alpha * (rule_val + gamma * np.max(q_table[state_idx + 1]))

    # return the updated q-value
    return ret, updated_q_val


def choose_state_action(state_idx, epsilon, alpha, gamma):
    # choose action
    # if random number less than epsilon, choose random action
    # else choose the action with the highest q-value
    if np.random.random() < epsilon:
        action_idx = np.random.randint(0, 5)
    else:
        action_idx = np.argmax(q_table[state_idx])

    # derive updated q value
    ret, updated_q_val = derive_updated_q_val(state_idx, action_idx, alpha, gamma)

    # update q table
    if ret:
        q_table[state_idx, action_idx] = updated_q_val

    # return the ret
    return ret


if __name__ == "__main__":
    # set some paramters
    episodes = 20
    alpha = 0.1
    gamma = 0.5
    epsilon = 0.1

    # counting the number of steps
    step_count = 0

    # for each episode
    for episode in range(episodes):
        # set the current state
        state_idx = 0

        # set the step count to 0
        step_count = 0

        # while not reach the goal state
        while state_idx < 5:
            # choose action
            ret = choose_state_action(state_idx, epsilon, alpha, gamma)

            # if choose action successfully
            if ret:
                # set the next state
                state_idx = state_idx + 1

            # if choose action unsuccessfully
            else:
                # back to start point
                state_idx = 0

            # increase the step count
            step_count = step_count + 1

        # print the episode, step count
        print('episode: {}, step count: {}\nq-table:\n{}'.format(episode, step_count, q_table))

```


## 3.5. 运行结果
这个程序其实差不多6-7回合就会收敛，不过我们还是看看执行20次会有什么情况
```
episode: 0, step count: 591
q-table:
[[17.470617  0.        0.        0.        0.      ]
 [ 0.       14.982189  0.        0.        0.      ]
 [ 0.        0.       10.045432  0.        0.      ]
 [ 0.        0.        0.        1.9       0.      ]
 [ 0.        0.        0.        0.        1.      ]]
episode: 1, step count: 5
q-table:
[[17.472666  0.        0.        0.        0.      ]
 [ 0.       14.986242  0.        0.        0.      ]
 [ 0.        0.       10.135889  0.        0.      ]
 [ 0.        0.        0.        2.76      0.      ]
 [ 0.        0.        0.        0.        1.95    ]]
episode: 2, step count: 5
q-table:
[[17.47471   0.        0.        0.        0.      ]
 [ 0.       14.994412  0.        0.        0.      ]
 [ 0.        0.       10.2603    0.        0.      ]
 [ 0.        0.        0.        3.5815    0.      ]
 [ 0.        0.        0.        0.        2.8525  ]]
episode: 3, step count: 5
q-table:
[[17.47696    0.         0.         0.         0.       ]
 [ 0.        15.007986   0.         0.         0.       ]
 [ 0.         0.        10.413344   0.         0.       ]
 [ 0.         0.         0.         4.365975   0.       ]
 [ 0.         0.         0.         0.         3.7098749]]
episode: 4, step count: 10
q-table:
[[17.48309    0.         0.         0.         0.       ]
 [ 0.        15.0545845  0.         0.         0.       ]
 [ 0.         0.        10.749577   0.         0.       ]
 [ 0.         0.         0.         5.114871   0.       ]
 [ 0.         0.         0.         0.         4.524381 ]]
episode: 5, step count: 8
q-table:
[[17.49309   0.        0.        0.        0.      ]
 [ 0.       15.115423  0.        0.        0.      ]
 [ 0.        0.       10.930363  0.        0.      ]
 [ 0.        0.        0.        5.829603  0.      ]
 [ 0.        0.        0.        0.        5.298162]]
episode: 6, step count: 5
q-table:
[[17.499552   0.         0.         0.         0.       ]
 [ 0.        15.150399   0.         0.         0.       ]
 [ 0.         0.        11.128807   0.         0.       ]
 [ 0.         0.         0.         6.511551   0.       ]
 [ 0.         0.         0.         0.         6.0332537]]
episode: 7, step count: 5
q-table:
[[17.507116   0.         0.         0.         0.       ]
 [ 0.        15.191799   0.         0.         0.       ]
 [ 0.         0.        11.341504   0.         0.       ]
 [ 0.         0.         0.         7.1620584  0.       ]
 [ 0.         0.         0.         0.         6.731591 ]]
episode: 8, step count: 5
q-table:
[[17.515995  0.        0.        0.        0.      ]
 [ 0.       15.239695  0.        0.        0.      ]
 [ 0.        0.       11.565456  0.        0.      ]
 [ 0.        0.        0.        7.782432  0.      ]
 [ 0.        0.        0.        0.        7.395012]]
episode: 9, step count: 5
q-table:
[[17.52638    0.         0.         0.         0.       ]
 [ 0.        15.293998   0.         0.         0.       ]
 [ 0.         0.        11.798033   0.         0.       ]
 [ 0.         0.         0.         8.3739395  0.       ]
 [ 0.         0.         0.         0.         8.025261 ]]
episode: 10, step count: 5
q-table:
[[17.538443  0.        0.        0.        0.      ]
 [ 0.       15.3545    0.        0.        0.      ]
 [ 0.        0.       12.036926  0.        0.      ]
 [ 0.        0.        0.        8.937809  0.      ]
 [ 0.        0.        0.        0.        8.623998]]
episode: 11, step count: 5
q-table:
[[17.552322  0.        0.        0.        0.      ]
 [ 0.       15.420897  0.        0.        0.      ]
 [ 0.        0.       12.280124  0.        0.      ]
 [ 0.        0.        0.        9.475228  0.      ]
 [ 0.        0.        0.        0.        9.192798]]
episode: 12, step count: 5
q-table:
[[17.568134  0.        0.        0.        0.      ]
 [ 0.       15.492813  0.        0.        0.      ]
 [ 0.        0.       12.525873  0.        0.      ]
 [ 0.        0.        0.        9.987346  0.      ]
 [ 0.        0.        0.        0.        9.733158]]
episode: 13, step count: 5
q-table:
[[17.585962  0.        0.        0.        0.      ]
 [ 0.       15.569825  0.        0.        0.      ]
 [ 0.        0.       12.772654  0.        0.      ]
 [ 0.        0.        0.       10.475269  0.      ]
 [ 0.        0.        0.        0.       10.2465  ]]
episode: 14, step count: 5
q-table:
[[17.605858  0.        0.        0.        0.      ]
 [ 0.       15.651475  0.        0.        0.      ]
 [ 0.        0.       13.019152  0.        0.      ]
 [ 0.        0.        0.       10.940067  0.      ]
 [ 0.        0.        0.        0.       10.734175]]
episode: 15, step count: 5
q-table:
[[17.627846  0.        0.        0.        0.      ]
 [ 0.       15.737285  0.        0.        0.      ]
 [ 0.        0.       13.26424   0.        0.      ]
 [ 0.        0.        0.       11.38277   0.      ]
 [ 0.        0.        0.        0.       11.197466]]
episode: 16, step count: 5
q-table:
[[17.651926  0.        0.        0.        0.      ]
 [ 0.       15.826768  0.        0.        0.      ]
 [ 0.        0.       13.506955  0.        0.      ]
 [ 0.        0.        0.       11.804366  0.      ]
 [ 0.        0.        0.        0.       11.637592]]
episode: 17, step count: 5
q-table:
[[17.678072  0.        0.        0.        0.      ]
 [ 0.       15.919439  0.        0.        0.      ]
 [ 0.        0.       13.746478  0.        0.      ]
 [ 0.        0.        0.       12.205809  0.      ]
 [ 0.        0.        0.        0.       12.055713]]
episode: 18, step count: 8
q-table:
[[17.736353   0.         0.         0.         0.       ]
 [ 0.        16.100662   0.         0.         0.       ]
 [ 0.         0.        13.9821205  0.         0.       ]
 [ 0.         0.         0.        12.588014   0.       ]
 [ 0.         0.         0.         0.        12.452927 ]]
episode: 19, step count: 5
q-table:
[[17.76775    0.         0.         0.         0.       ]
 [ 0.        16.189701   0.         0.         0.       ]
 [ 0.         0.        14.213309   0.         0.       ]
 [ 0.         0.         0.        12.9518585  0.       ]
 [ 0.         0.         0.         0.        12.83028  ]]

Process finished with exit code 0

```

怎么样，弄明白后是不是特别简单？