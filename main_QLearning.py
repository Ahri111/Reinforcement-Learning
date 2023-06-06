from envs import YourCustomClass
from agents import QLearning
import matplotlib.pyplot as plt

# Initial State = ???
DISCOUNT = #
UPDATE_RATE = 0.1
EPS_INIT = #
EPS_LOWER_BOUND = #
ANNEAL = 0.0001
EPISODE_NUM = #
BUFFER_SIZE = #


def main():
    env = YourCustomClass()
    agent = QLearning(env.states_dim, env.actions_dim, discount=DISCOUNT, update_rate=UPDATE_RATE,
                      eps_init=EPS_INIT, eps_lower_bound=EPS_LOWER_BOUND, anneal=ANNEAL, buffer_size=BUFFER_SIZE)
    average_step_record = []
    average_gain_record = []

    for episode in range(EPISODE_NUM):
        current_state = env.reset()
        while True:
            current_action = agent.epsilon_greedy(current_state)
            reward, next_state = env.step(current_action)

            agent.replay_buffer.push(
                current_state, current_action, reward, next_state)
            
            agent.update_q()
            
            current_state = next_state
            
            if next_state == env.terminal_state:
                break

        agent.anneal_eps()

        if (episode + 1) % 100 == 0:
            total_step = 0
            total_gain = 0
            count = 0
            for s in range(env.states_dim - 1):
                current_state = env.reset(init_state=s)
                if current_state == -1:
                    continue
                count += 1
                step = 0
                gain = 0
                while True:
                    current_action = agent.greedy(current_state)
                    reward, next_state = env.step(current_action)


                    step += 1
                    gain += reward

                    if next_state == env.terminal_state:
                        break

                    if step > 100:
                        print(f"Over 100")
                        break

                total_step += step
                total_gain += gain

            average_step = total_step / count
            average_gain = total_gain / count
            average_step_record.append(average_step)
            average_gain_record.append(average_gain)
            print(
                f"Episode {episode + 1}: {average_step:.2f} steps, {average_gain:.2f} gain")

    # Example trajectory generator
    for episode in range(5):
        current_state = env.reset()
        print(f"Episode example {episode + 1}")
        while True:
            current_action = agent.greedy(current_state)
            print(current_action, agent.q[current_state, current_action])
            reward, next_state = env.step(current_action, render=True)
            current_state = next_state

            if next_state == env.terminal_state:
                break

    episode_index = list(range(100, 100 * len(average_step_record) + 1, 100))

    """

    INSERT YOUR CODE

    """


if __name__ == '__main__':
    main()
