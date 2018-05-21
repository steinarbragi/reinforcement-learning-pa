import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np

style.use('fivethirtyeight')


data = np.genfromtxt('lunarlander_rewards.csv',skip_header=1, delimiter=',')


plt.plot(data[0],data[1]);
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
