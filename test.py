from  lunarlander_wrapper  import  LunarLanderWrapper
from  my_agent  import  MyAgent
wrapper = LunarLanderWrapper ()
agent = MyAgent(wrapper=wrapper , seed =42)
rewards = []
rewards = [0.0] * 10
for  episode  in  range(10):
    
    rewards.append(agent.train())
    if  wrapper.solved(rewards[episode]):
        break
wrapper.close()