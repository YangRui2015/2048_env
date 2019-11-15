# -*- coding: utf-8 -*-
# dqn_agent.py
# author: yangrui
# description: 
# created: 2019-10-12T11:07:45.524Z+08:00
# last-modified: 2019-10-12T11:07:45.524Z+08:00
# email: yangrui19@mails.tsinghua.edu.cn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import utils
from NN_module import CNN_Net, FC_Net
from Buffer_module import Buffer


class DQN():
    batch_size = 128
    lr = 1e-4
    epsilon = 0.15   
    memory_capacity =  int(1e4)
    gamma = 0.99
    q_network_iteration = 200
    save_path = "./save/"
    soft_update_theta = 0.1
    clip_norm_max = 1
    train_interval = 5
    conv_size = (32, 64)   # num filters
    fc_size = (512, 128)

    def __init__(self, num_state, num_action, enable_double=False, enable_priority=True):
        super(DQN, self).__init__()
        self.num_state = num_state
        self.num_action = num_action
        self.state_len = int(np.sqrt(self.num_state))
        self.enable_double = enable_double
        self.enable_priority = enable_priority

        self.eval_net, self.target_net = CNN_Net(self.state_len, num_action,self.conv_size, self.fc_size), CNN_Net(self.state_len, num_action, self.conv_size, self.fc_size)
        # self.eval_net, self.target_net = FC_Net(self.num_state, self.num_action), FC_Net(self.num_state, self.num_action)

        self.learn_step_counter = 0
        self.buffer = Buffer(self.num_state, 'priority', self.memory_capacity)
        # self.memory = np.zeros((self.memory_capacity, num_state * 2 + 2))     
        self.initial_epsilon = self.epsilon
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)


    def select_action(self, state, random=False, deterministic=False):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) 
        if not random and np.random.random() > self.epsilon or deterministic:  # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value.reshape(-1,4), 1)[1].data.numpy()
        else: # random policy
            action = np.random.randint(0,self.num_action)
        return action


    def store_transition(self, state, action, reward, next_state):
        state = state.reshape(-1)
        next_state = next_state.reshape(-1)

        transition = np.hstack((state, [action, reward], next_state))
        self.buffer.store(transition)
        # index = self.memory_counter % self.memory_capacity
        # self.memory[index, :] = transition
        # self.memory_counter += 1


    def update(self):
        #soft update the parameters
        if self.learn_step_counter % self.q_network_iteration ==0 and self.learn_step_counter:
            for p_e, p_t in zip(self.eval_net.parameters(), self.target_net.parameters()):
                p_t.data = self.soft_update_theta * p_e.data + (1 - self.soft_update_theta) * p_t.data
                
        self.learn_step_counter+=1

        #sample batch from memory
        if self.enable_priority:
            batch_memory, (tree_idx, ISWeights) = self.buffer.sample(self.batch_size)
        else:
            batch_memory, _ = self.buffer.sample(self.batch_size)

        batch_state = torch.FloatTensor(batch_memory[:, :self.num_state])
        batch_action = torch.LongTensor(batch_memory[:, self.num_state: self.num_state+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.num_state+1: self.num_state+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-self.num_state:])

        #q_eval
        q_eval_total = self.eval_net(batch_state)
        q_eval = q_eval_total.gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()

        if self.enable_double:
            q_eval_argmax = q_eval_total.max(1)[1].view(self.batch_size, 1)
            q_max = q_next.gather(1, q_eval_argmax).view(self.batch_size, 1)
        else:
            q_max = q_next.max(1)[0].view(self.batch_size, 1)

        q_target = batch_reward + self.gamma * q_max

        if self.enable_priority:
            abs_errors = (q_target - q_eval.data).abs()
            self.buffer.update(tree_idx, abs_errors)
            # loss = (torch.FloatTensor(ISWeights) * (q_target - q_eval).pow(2)).mean()   
            loss = (q_target - q_eval).pow(2).mean() # 可能去掉ISweight更好？？

            
            # print(ISWeights)
            # print(loss)

            # import pdb; pdb.set_trace()
        else:
            loss = F.mse_loss(q_eval, q_target)
        

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.clip_norm_max)
        self.optimizer.step()

        return loss

    
    def save(self, path=None, name='dqn_net.pkl'):
        path = self.save_path if not path else path
        utils.check_path_exist(path)
        torch.save(self.eval_net.state_dict(), path + name)

    def load(self, path=None, name='dqn_net.pkl'):
        path = self.save_path if not path else path
        self.eval_net.load_state_dict(torch.load(path + name))


    def epsilon_decay(self, episode, total_episode):
        self.epsilon = self.initial_epsilon * (1 - episode / total_episode)
