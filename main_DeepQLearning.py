from agents import DeepQLearning
from envs import YourCustomClass
import torch
import matplotlib.pyplot as plt
import random
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DISCOUNT = 0.99
EPS_INIT = 0.9
EPS_LOWER_BOUND = 0.05
ANNEAL = 0.0001
EPISODE_NUM = 100
BUFFER_SIZE = 256
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
terminal_state = 5000
C = 10


def is_terminate(number, next_state):
    
    if number > 6:
        print("Lose: Birdie")
        return 1
    
    if next_state < 0:
        print("impossible")
        return 1
    
    elif next_state > terminal_state:
        print("Lose: outside ")
        return 1
    
    elif terminal_state - 200 < next_state < terminal_state + 200:
        print("you win")
        return 1
    
    elif 2200 < terminal_state < 2300:
        print("lose: hazard") 
        return 1
    
    else:
        return 0

def main():
    env = YourCustomClass()
    agent = DeepQLearning(env.states_dim, env.actions_dim, discount=DISCOUNT, update_rate=LEARNING_RATE,
                          eps_init=EPS_INIT, eps_lower_bound=EPS_LOWER_BOUND, anneal=ANNEAL,
                          buffer_size=BUFFER_SIZE, device=DEVICE)
    average_step_record = []
    average_gain_record = []

    for episode in range(EPISODE_NUM):
        current_state = env.reset()
        number = 0
        while True:
            current_action = agent.epsilon_greedy(current_state)
            reward, next_state = env.step(current_action)
            number += 1
            done = is_terminate(number, next_state)
            agent.buffer.push(current_state, current_action, reward, next_state, done)
            agent.update_q(BATCH_SIZE)
            current_state = next_state
            
            if next_state = env.terminal_state:
                break
            
            
        agent.anneal_eps()
        
        if (episode + 1) %  C == 0:
            agent.target_network.load_state_dict(agent.network.state_dict())

        if (episode + 1) % 100 == 0:
            total_step = 0
            total_gain = 0
            count = 0
            for s in range(env.states_dim - 1):
                current_state = env.reset(init_state=s)
                count += 1
                step = 0
                gain = 0
                while True:
                    current_action = agent.greedy(current_state)
                    reward, next_state = env.step(current_action)
                    gain += reward
                    step += 1

                    if next_state == env.terminal_state:
                        break

                    if step > 100:
                        print(f"State {s} over 100")
                        break

                total_step += step
                total_gain += gain

            average_step = total_step / count
            average_gain = total_gain / count
            average_step_record.append(average_step)
            average_gain_record.append(average_gain)
            print(
                f"Episode {episode + 1}: {average_step:.2f} steps, {average_gain:.2f} gain")

    # Example
    for episode in range(5):
        current_state = env.reset()
        print(f"Episode example {episode + 1}")
        step = 0
        while True:
            step += 1
            state_tensor = torch.tensor([current_state]).float().to(agent.device)
            q_value = agent.target_net(state_tensor).detach().numpy()
            current_action = np.argmax(q_value)
            print(current_action, q_value)
            reward, next_state = env.step(current_action)
            current_state = next_state

            if step > 100:
                print(f"Over 100")
                break

            if next_state == env.terminal_state:
                break

    episode_index = list(range(100, 100 * len(average_step_record) + 1, 100))

    plt.figure(figsize=(10,5))
    plt.plot(episode_index, average_step_record, label='Average steps per 100 episodes')
    plt.plot(episode_index, average_gain_record, label='Average gain per 100 episodes')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
