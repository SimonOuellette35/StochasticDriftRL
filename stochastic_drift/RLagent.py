import data_generator as dg
import numpy as np
from dqn_agent_xgb import DQNAgent
import matplotlib.pyplot as plt

N = 2000
TRAINING_N = int(0.5 * N)
VALIDATION_N = N - TRAINING_N
NUM_SIMULATIONS = 50
COMPARE_WITH_HISTORICAL = True

Y = dg.normalRW_with_noise(N)

plt.plot(Y)
plt.title("Generated time series")
plt.show()

ROLLING_WINDOW = 0
REWARD_HORIZON = 5

state_size = 2
action_size = 3

def evaluate(agent, num_sim=100):

    SRs = []
    validation_pnls = []
    current_position = 0
    for _ in range(num_sim):
        # generate 100 data points with the real DGP
        newY = dg.normalRW_with_noise(500)

        currentPnL = 0.0
        for t in range(1, len(newY)):

            # action is either -1 (Short), 0 (Flat), or 1 (Long)
            state = [newY[t], float(current_position)]
            action = strategy(state, agent, False)

            # calculate P&L accrued from last time step
            log_return = np.log(newY[t]) - np.log(newY[t - 1])
            currentPnL += (log_return * current_position)
            validation_pnls.append(log_return * current_position)

            if action != current_position:
                currentPnL = 0.0

            # now apply action
            current_position = action

        # add current cumulative SR
        sr = sharpeRatio(validation_pnls)
        SRs.append(sr)

    return SRs

def strategy(state, agent, use_explo=True):
    state = np.reshape(state, [-1, state_size])
    return agent.act(state, use_explo) - 1

def exponential_decay(rewards):
    weight = 1.0
    total_weight = 0.0
    total_sum = 0.0
    for r in rewards:
        total_sum += weight * r
        total_weight += weight
        weight = weight * 0.9

    return (total_sum / total_weight) * 1000.0

def sharpeRatio(pnls):
    avg = np.mean(pnls)
    stdev = np.std(pnls)

    print "Avg pnl = ", avg
    print "Stdev of pnls = ", stdev

    return (avg / stdev) * np.sqrt(252)

if COMPARE_WITH_HISTORICAL:

    agent = DQNAgent(state_size, action_size)

    pnls = []
    training_evolution = []
    print "Training on data points %s to %s." % (ROLLING_WINDOW, N-1)
    for sim in range(NUM_SIMULATIONS):

        print "==> Running simulation", sim
        current_position = 0
        current_sim_pnls = []
        currentPnL = 0.0
        for t in range(ROLLING_WINDOW, TRAINING_N - REWARD_HORIZON):

            # action is either -1 (Short), 0 (Flat), or 1 (Long)
            state = [Y[t], float(current_position)]
            action = strategy(state, agent)

            # memorize state-action-reward tuple for later training
            future_rewards = (np.log(Y[t+1:t+1+REWARD_HORIZON]) - np.log(Y[t:t+REWARD_HORIZON])) * action
            reward = exponential_decay(future_rewards)
            agent.remember(state, action+1, reward)

            # calculate P&L accrued from last time step
            log_return = np.log(Y[t]) - np.log(Y[t-1])
            currentPnL += (log_return * current_position)
            pnls.append(log_return * current_position)
            current_sim_pnls.append(log_return * current_position)

            if action != current_position:
                currentPnL = 0.0

            # now apply action
            current_position = action

        print "==> Simulation performance: sum of P&Ls is", np.sum(current_sim_pnls)
        training_evolution.append(np.sum(current_sim_pnls))
        # Let's learn a bit from our exploration
        NUM_BATCHES = 10
        BATCH_SIZE = TRAINING_N
        for _ in range(NUM_BATCHES):
            agent.replay()

        print "Epsilon after training: ", agent.epsilon

    plt.plot(training_evolution)
    plt.show()

    print "Training set evaluation..."

    # First, the in-sample performance evaluation of our trained agent.
    training_pnls = []
    current_position = 0
    actions_taken = []
    currentPnL = 0.0
    for t in range(ROLLING_WINDOW, TRAINING_N):
        # action is either -1 (Short), 0 (Flat), or 1 (Long)
        state = [Y[t], float(current_position)]
        action = strategy(state, agent, False)

        actions_taken.append(action)

        # calculate P&L accrued from last time step
        log_return = np.log(Y[t]) - np.log(Y[t - 1])
        currentPnL += (log_return * current_position)
        training_pnls.append(log_return * current_position)

        if action != current_position:
            currentPnL = 0.0

        # now apply action
        current_position = action

    sr = sharpeRatio(training_pnls)
    print "Training set Sharpe ratio: ", sr
    print "Average daily P&L: ", np.mean(training_pnls)

    print "Validation set evaluation..."

    # Then, let's see how well this generalizes to future data...
    SRs = evaluate(agent)

    plt.plot(SRs)
    plt.title("History-based model: cumulative Sharpe ratio")
    plt.show()

    print "Cumulative validation sharpe ratio after 1000 simulations: ", SRs[-1]
