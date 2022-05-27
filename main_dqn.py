from dqn_agent import DQN
from gym_2048 import Game2048Env
import torch
import numpy as np 
import time
import logger
from utils import log2_shaping, Perfomance_Saver, Model_Saver


train_episodes = 20000
test_episodes = 50
ifrender = False
eval_interval = 25
epsilon_decay_interval = 100
log_interval = 5



def train():
    episodes = train_episodes
    logger.configure(dir="./log/", format_strs="stdout,tensorboard,log")
    agent = DQN(num_state=16, num_action=4)
    env = Game2048Env()

    pf_saver = Perfomance_Saver()
    model_saver = Model_Saver(num=10)

    eval_max_score = 0
    for i in range(episodes):
        state, reward, done, info = env.reset()
        state = log2_shaping(state)

        start = time.time()
        loss = None
        while True:
            if agent.buffer.memory_counter <= agent.memory_capacity:
                action = agent.select_action(state, random=True)
            else:
                action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)
            next_state = log2_shaping(next_state)
            reward = log2_shaping(reward, divide=1)

            agent.store_transition(state, action, reward, next_state)
            state = next_state

            if ifrender:
                env.render()

            if agent.buffer.memory_counter % agent.train_interval == 0 and agent.buffer.memory_counter > agent.memory_capacity:  # 相当于填满后才update
                loss = agent.update()

            if done:
                if i % log_interval == 0:
                    if loss:
                        logger.logkv('loss', loss)
                    logger.logkv('training progress', (i+1) / episodes)
                    logger.logkv('episode reward', info['score'])
                    logger.logkv('episode steps', info['steps'])
                    logger.logkv('highest', info['highest'])
                    logger.logkv('epsilon', agent.epsilon)
                    logger.dumpkvs()

                    loss = None

                if i % epsilon_decay_interval == 0:   # episilon decay
                    agent.epsilon_decay(i, episodes)
                break
        
        end = time.time()
        print('episode time:{} s\n'.format(end - start))

        # eval 
        if i % eval_interval == 0 and i:
            eval_info = test(episodes=test_episodes, agent=agent)
            average_score, max_score, score_lis = eval_info['mean'], eval_info['max'], eval_info['list']

            pf_saver.save(score_lis, info=f'episode:{i}')

            if int(average_score) > eval_max_score:
                eval_max_score = int(average_score)
                name = 'dqn_{}.pkl'.format(int(eval_max_score))
                agent.save(name=name)
                model_saver.save("./save/" + name)

            logger.logkv('eval average score', average_score)
            logger.logkv('eval max socre', max_score)
            logger.dumpkvs()



def test(episodes=20, agent=None, load_path=None, ifrender=False, log=False):
    if log:
        logger.configure(dir="./log/", format_strs="stdout")
    if agent is None:
        agent = DQN(num_state=16, num_action=4)
        if load_path:
            agent.load(load_path)
        else:
            agent.load()
    
    env = Game2048Env()
    score_list = []
    highest_list = []
 
    for i in range(episodes):
        state, _, done, info = env.reset()
        state = log2_shaping(state)

        start = time.time()
        while True:
            action = agent.select_action(state, deterministic=True)
            next_state, _, done, info = env.step(action)
            next_state = log2_shaping(next_state)
            state = next_state

            if ifrender:
                env.render()

            if done:
                if log:
                    logger.logkv('episode number', i + 1)
                    logger.logkv('episode reward', info['score'])
                    logger.logkv('episode steps', info['steps'])
                    logger.logkv('highest', info['highest'])
                    logger.dumpkvs()
                break
        
        end = time.time()
        if log:
            print('episode time:{} s\n'.format(end - start))

        score_list.append(info['score'])
        highest_list.append(info['highest'])
    
    print('mean score:{}, mean highest:{}'.format(np.mean(score_list), np.mean(highest_list)))
    print('max score:{}, max hightest:{}'.format(np.max(score_list), np.max(highest_list)))
    result_info = {'mean':np.mean(score_list), 'max':np.max(score_list), 'list':score_list}
    return result_info


if __name__ == "__main__":
    # test(episodes=test_episodes, ifrender=ifrender)
    train()
