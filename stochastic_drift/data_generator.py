import numpy as np

def normalRW_with_noise(N, use_constant_mean=False):

    if use_constant_mean:
        stochastic_mean = 1.0
    else:
        stochastic_mean = np.random.normal(1.0, 0.0003, 1)[0]

    Y = [stochastic_mean]
    for t in range(1, N):
        new_Y = np.random.normal(stochastic_mean, 0.001, 1)[0]

        if not use_constant_mean:
            tmp = np.random.normal(0.0, 0.0003, 1)[0]
            stochastic_mean += tmp

        Y.append(new_Y)

    return np.array(Y)
