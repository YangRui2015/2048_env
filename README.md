# 2048 Environment for Reinforcement Learning and DQN Algorithm implementation

Thanks to the author of gym-2048 https://github.com/rgal/gym-2048. The code is easy to understand and runs efficiently. I just made some little changes to make it a better RL environment.
And I implemented dqn with many tricks using pytorch:
- [x] Randomly fill buffer first;
- [x] Soft target replacing;
- [x] Epsilon decay;
- [x] Clip gradient norm;
- [x] Double DQN;
- [x] Priority Experience Replay;



## Performance of environment
I used random policy to evaluate the performance for 1000 times. We can take random policy as a baseline.  
The evaluation main function is in base_agent.py.

### (1) with rendering:  
average episode time:0.10279795455932617 s;  
average step time: 0.7373 ms；  
average highest score:106.368;  
average total score:1078.252;  
average steps:139.417;  

### (2) without rendering:  
average episode time:0.03773710775375366 s;  
average step time: 0.2671 ms；  
average highest score:108.24;  
average total score:1102.088;  
average steps:141.288;  

some example:  
![image](https://github.com/YangRui2015/2048_env/blob/master/pictures/example.png)

## Performance of Priority DQN
Training for 45k episodes and max eval mean score is 7700(eval for 50 episodes).



## Update
1. add max steps and max illegal steps of one episode;
