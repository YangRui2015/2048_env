# -*- coding: utf-8 -*-
# base_agent.py
# author: yangrui
# description: 
# created: 2019-09-29T15:01:38.383Z+08:00
# last-modified: 2019-09-29T15:01:38.383Z+08:00
# email: yangrui19@mails.tsinghua.edu.cn

from gym_2048 import Game2048Env
import random


class BaseAgent():
    def act(self, state):
        raise NotImplementedError


class RandomAgent(BaseAgent):
    def act(self, state):
        return random.randint(0, 3)


if __name__ == "__main__":
    import time
    import numpy as np 

    def run(ifrender=False):
        agent = RandomAgent()
        env = Game2048Env()
        state, reward, done, info = env.reset()
        if ifrender:
            env.render()
 
        start = time.time()
        while True:
            action = agent.act(state)
            # print('action: {}'.format(action))
            state, reward, done, info = env.step(action)
            if ifrender:
                env.render()
            if done:
                print('\nfinished, info:{}'.format(info))
                break
        
        end = time.time()
        print('episode time:{} s\n'.format(end - start))
        return end - start, info['highest'], info['score'], info['steps']

    time_lis, highest_lis, score_lis, steps_lis = [], [], [], []
    for i in range(1000):
        t, highest, score, steps = run()
        time_lis.append(t)
        highest_lis.append(highest)
        score_lis.append(score)
        steps_lis.append(steps)
    
    print('eval result:\naverage episode time:{} s, average highest score:{}, average total score:{}, average steps:{}'.format(np.mean(time_lis), np.mean(highest_lis), np.mean(score_lis), np.mean(steps_lis)))
    


