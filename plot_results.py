import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np


data = np.genfromtxt('lunarlander_rewards.csv',skip_header=1, delimiter=',')

conv = []

max_val = -500

average = []
avg_last = -200

for i in range(len(data[:,1])):
    
    if (i > 100):
        avg_last = float(sum(data[i-100:i,1])) / 100

    if data[i,1] > max_val:
        max_val = data[i,1]
    conv.append(max_val)

    average.append(avg_last)
style.use('fivethirtyeight')

line_reward = plt.plot(data[:,0], data[:,1],label="Episode Reward", linestyle=":", color="silver")
line_max =  plt.plot(data[:,0], conv,label="Max Reward So Far",linewidth=3)
line_average =  plt.plot(data[:,0], average,label="Average Last 100", linewidth=3)
plt.legend(handles=[line_reward[0], line_max[0], line_average[0]])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.tight_layout()
plt.savefig('pg.png')
plt.show()
