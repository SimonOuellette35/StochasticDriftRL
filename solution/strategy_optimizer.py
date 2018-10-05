import data_generator as dg
import numpy as np
from dqn_agent_xgb import DQNAgent
import matplotlib.pyplot as plt

N = 2000
TRAINING_N = int(0.5 * N)
VALIDATION_N = N - TRAINING_N
USE_MANUAL_STRATEGY = False
NUM_SIMULATIONS = 150
COMPARE_WITH_HISTORICAL = True

Y = dg.normalRW_with_noise(N)

ROLLING_WINDOW = 0
REWARD_HORIZON = 5

state_size = 2
action_size = 3

def evaluate(agent, num_sim=500):

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

# Here we produce a generative model using PyMC3.
import pymc3 as pm
import theano.tensor as tt

cutoff_idx = 1000
y_obs = np.ma.MaskedArray(Y, np.arange(N) > cutoff_idx)

print "y_obs.size = ", y_obs.size

interval = 200
with pm.Model() as OUmodel:

    sigma_mu = pm.HalfNormal('s_mu', sd=0.01)   # this must be sufficiently wide, considering that it's the sd
                                                # for a whole (y_obs.size // interval) period
    sigma = pm.HalfNormal('s', sd=0.01)

    mu = pm.GaussianRandomWalk('mu', mu=0.0, sd=sigma_mu, shape=interval)
    weights = tt.repeat(mu, y_obs.size // interval)

    offset_weights = pm.Deterministic('mu_offset', 1.0 + weights)
    y = pm.Normal('y', mu=offset_weights, sd=sigma, observed=y_obs)

    trace = pm.sample(1000, tune=1000)

    pm.traceplot(trace, varnames=[mu, sigma, sigma_mu])
    plt.show()

# Generate imagined trajectories and train agent on them
print "Generating data..."
simulations = []
for _ in range(NUM_SIMULATIONS):

    sigma_mus = trace['s_mu']
    sigmas = trace['s']

    # generative from the obtained samples
    random_walk = 1.0
    generated_y = []

    # we use the expectation, instead of random samples from the sigmas
    current_s_mu = np.mean(sigma_mus) / np.sqrt(y_obs.size // interval)
    current_s = np.mean(sigmas)

    for t in range(cutoff_idx):
        random_walk = random_walk + np.random.normal(0.0, current_s_mu, 1)[0]
        current_y = np.random.normal(random_walk, current_s, 1)[0]

        generated_y.append(current_y)

    print "length of generated_y = ", len(generated_y)
    simulations.append(generated_y)

agent = DQNAgent(state_size, action_size)

print "Now training agent on imagined trajectories..."

pnls = []
for simulation in range(NUM_SIMULATIONS):

    imagined_Y = simulations[simulation]

    if simulation < 5:
        plt.plot(imagined_Y)
        plt.show()

    current_position = 0
    current_sim_pnls = []
    currentPnL = 0.0
    for t in range(ROLLING_WINDOW, TRAINING_N - (REWARD_HORIZON+interval)):
        # action is either -1 (Short), 0 (Flat), or 1 (Long)
        state = [imagined_Y[t], float(current_position)]
        action = strategy(state, agent)

        # memorize state-action-reward tuple for later training
        future_rewards = (np.log(imagined_Y[t + 1:t + 1 + REWARD_HORIZON]) - np.log(imagined_Y[t:t + REWARD_HORIZON])) * action
        reward = exponential_decay(future_rewards)
        agent.remember(state, action + 1, reward)

        # calculate P&L accrued from last time step
        log_return = np.log(imagined_Y[t]) - np.log(imagined_Y[t - 1])
        currentPnL += (log_return * current_position)

        if simulation > 50: # we wait for the exploration phase to be over
            pnls.append(log_return * current_position)

        current_sim_pnls.append(log_return * current_position)

        if action != current_position:
            currentPnL = 0.0

        # now apply action
        current_position = action

    print "==> Simulation performance: sum of P&Ls is", np.sum(current_sim_pnls)

    # Let's learn a bit from our exploration
    NUM_BATCHES = 10
    for _ in range(NUM_BATCHES):
        agent.replay()

print "Epsilon after training: ", agent.epsilon

sr = sharpeRatio(pnls)
print "In-sample imagination Sharpe ratio: ", sr
print "In-sample average P&L: ", np.mean(pnls)

print "Now evaluating validation set performance of the agent trained on imagined trajectories..."

# Step 3) Evaluate out of sample performance
SRs = evaluate(agent)
plt.plot(SRs)
plt.title("Imagination-based model: cumulative Sharpe ratio")
plt.show()

print "Cumulative validation sharpe ratio after 1000 simulations: ", SRs[-1]
