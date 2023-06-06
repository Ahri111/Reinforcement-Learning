import random
import numpy as np
import torch

goal = 63
energy = 299


action = {'action1': (5, np.pi/6),
          'action2': (5, np.pi/3),
          'action3': (6, np.pi/6),
          'action4': (6, np.pi/3),
          'action5': (7, np.pi/6),
          'action6': (7, np.pi/3),
          'action7': (8, np.pi/6),
          'action8': (8, np.pi/3),
          'action9': (9, np.pi/6),
          'action10': (9, np.pi/3), 
          }
fuel = energy


class YourCustomClass(object):
    def __init__(self):
        self.actions_dim = 10
        self.actions = [(5, np.pi/4),
                        (5, np.pi/3),
                        (10, np.pi/4),
                        (10, np.pi/3),
                        (15, np.pi/4),
                        (15, np.pi/3),
                        (20, np.pi/4),
                        (20, np.pi/3),
                        (25, np.pi/4),
                        (25, np.pi/3)]
        self.states_dim = 1
        
    def reset(self, state = None):
        if state != None:
            self.state = state
        else:
            self.state = torch.tensor([[0]])      
        return self.state

    def step(self, current_location, current_action):
        velocity, theta = self.actions[current_action]
        distance = velocity**2 * np.sin(2*theta)/9.81
        next_state = current_location + distance
        
        if 220 <= next_state <= 252:
            reward = 100
            
        elif 19 < next_state < 50:
            reward = -100
            
        else:
            reward = -1
        
        return reward, next_state
    
def state_resistance(velocity):
  if velocity == 5 or velocity == 6:
    resistance = 0
  
  elif velocity == 7 or velocity == 8:
    resistance = 1
  
  elif velocity == 9 or velocity == 10:
    resistance = 2
  
  elif velocity == -5 or velocity == -6:
    resistasnce = 0
  
  elif velocity == -7 or velocity == -8:
    resistance = -1

  elif velocity == -9 or velocity == -10:
    resistance = -2
  
  return resistance

def distance(v, angle_dir, resistance):
      return (v*np.cos(angle_dir) - resistance) * 2 * v * np.sin(angle_dir) / 9.81

def get_reward(fuel, next_state):
    
  if fuel < 0 or next_state > goal :
    reward = -100
  
  elif goal - 1 < next_state <= goal:
    reward = 100
  
  else:
    reward = -1
  
  return reward

def is_terminate(fuel, next_state):
    
  if fuel < 0:
    print('lose')
    return True

  elif next_state > goal:
    print('lose')
    return True
  
  elif goal - 4 < next_state <= goal:
    print('win')
    return True
  
  else:
    return False

 

def env(current_location, current_action, fuel):
  
  v, theta = action[current_action]
  wind_v = state_resistance(v)
  c_distance = distance(v, theta, wind_v)
  next_location = current_location + c_distance
  next_state = next_location
  c_reward = get_reward(fuel, next_state)
  done = is_terminate(fuel, next_state)
  environment = (next_state, c_reward, done, {})
  return environment

def sr(state):
    state = round(state)
    return state

def move(current_location, velocity, theta, resistance):
    NL = current_location + distance(velocity,theta,resistance)
    return NL

def env_inference(q,action_selection):
  location_t = 0
  fuel = 299
  new_state_t =0
  while not is_terminate(fuel, new_state_t):
    old_location_t = location_t
    old_state_t = sr(old_location_t)
    action_idx_t = np.argmax(q[old_state_t])
    action_type_t = action_selection[action_idx_t]
    fuel -= 10
    print(action_type_t)
    v_t, theta_t = action[action_type_t]
    location_t = move(location_t, v_t, theta_t)
    new_state_t = sr(location_t)
    print(location_t)
    print(new_state_t)
  print('It is the end of Q learning')