import numpy

import sys

import pandas as pd


class td_qlearning:


  alpha = 0.1

  gamma = 0.5

  next_state=0

  main_df = pd.DataFrame(columns=['state','action','qvalue'])



  def __init__(self, trajectory_filepath):

    df = pd.read_csv (trajectory_filepath,names= ["state", "action"],dtype=str)

    for ind in df.index:

      if(ind < len(df)-1):

        self.next_state=df['state'][ind+1]

      retQvalue=self.setqvalue(df['state'][ind],df['action'][ind])



  def setqvalue(self, state, action):

    r = self.reward(state)

    newEntry={'state': state,'action': action, 'qvalue': 0.00}

    if(self.main_df['state'].str.contains(state).any()==False and self.main_df['action'].str.contains(action).any()==False):

      self.main_df = self.main_df.append(newEntry, ignore_index=True)

    select_states = self.main_df.loc[self.main_df['state'] == state]

    select_action = select_states.loc[self.main_df['action'] == action]

    mainIndex= select_action.index.tolist()[0]

    if(self.next_state[0]=="1"):

      maxVal=self.argmax(["C","D"])

    elif(self.next_state[0]=="2"):

      maxVal=self.argmax(["C","R"])

    elif(self.next_state[0]=="3"):

      maxVal=self.argmax(["U","D","R","L","C"])

    elif(self.next_state[0]=="4"):

      maxVal=self.argmax(["C","L"])

    elif(self.next_state[0]=="5"):

      maxVal=self.argmax(["C","U"])

    currQval = self.main_df.at[mainIndex,'qvalue'] + (self.alpha * (r + (self.gamma * maxVal) - self.main_df.at[mainIndex,'qvalue']))

    self.main_df.at[mainIndex,'qvalue']=currQval

    return currQval



  def qvalue(self,state,action):

    select_states = self.main_df.loc[self.main_df['state'] == state]

    select_action = select_states.loc[self.main_df['action'] == action]

    mainIndex= select_action.index.tolist()[0]
    return (self.main_df.at[mainIndex,'qvalue'])



  def policy(self, state):

    select_states = self.main_df.loc[self.main_df['state'] == state]

    maxIndex=select_states['qvalue'].idxmax()

    return select_states.at[maxIndex,'action']



  def reward(self, state):

    reward = 0

    for i in range(1,6):

      if state[i] == "1":

        reward +=1

    reward = -1 * reward

    return reward



  def argmax(self,actions):

    select_states = self.main_df.loc[self.main_df['state'] == self.next_state]

    for a in actions:

      if(select_states['action'].str.contains(a).any()==False):

        new_row = {'state':self.next_state,'action':a,'qvalue':0.00}

        self.main_df = self.main_df.append(new_row, ignore_index=True)

        select_states = select_states.append(new_row, ignore_index=True)

    max_value = select_states['qvalue'].max()

    return max_value


#change trajectory here for different tests:
