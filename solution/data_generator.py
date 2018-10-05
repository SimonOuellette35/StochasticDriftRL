import numpy as np

# def generate_data(N):
#     # generate time-sensitive data
#     coefficients = []
#     X = np.random.normal(0.0, 0.1, [N])
#     noise = np.random.normal(0.0, 0.1, [N])
#
#     for i in range(N):
#         coefficients.append(i * 0.005)
#
#     coefficients = np.array(coefficients)
#
#     beta = X * coefficients
#
#     Y = 0.5 + beta + noise
#
#     return X, Y
#
# def stochastic_mu_OU(N, use_constant_mean=False):
#     revert_speed = 0.1
#
#     if use_constant_mean:
#         stochastic_mean = 1.0
#     else:
#         stochastic_mean = np.random.normal(1.0, 0.01, 1)[0]
#
#     Y = [stochastic_mean]
#     for t in range(1, N):
#         innovation = np.random.normal(0.0, 0.001, 1)[0]
#         new_Y = Y[t-1] * (1.0 + innovation)
#         new_Y = revert_speed * (stochastic_mean - new_Y) + new_Y
#
#         if not use_constant_mean:
#             tmp = np.random.normal(0.0, 0.0005, 1)[0]
#             stochastic_mean = stochastic_mean + tmp
#
#         Y.append(new_Y)
#
#     return np.array(Y)
#
# def lognormalRW_with_noise(N, use_constant_mean=False):
#
#     if use_constant_mean:
#         stochastic_mean = 1.0
#     else:
#         stochastic_mean = np.random.normal(1.0, 0.0003, 1)[0]
#
#     Y = [stochastic_mean]
#     for t in range(1, N):
#         innovation = np.random.normal(stochastic_mean, 0.001, 1)[0]
#         new_Y = innovation
#
#         if not use_constant_mean:
#             tmp = np.random.normal(1.0, 0.0003, 1)[0]
#             stochastic_mean = stochastic_mean * tmp
#
#         Y.append(new_Y)
#
#     return np.array(Y)

def normalRW_with_noise(N, use_constant_mean=False):

    if use_constant_mean:
        stochastic_mean = 1.0
    else:
        stochastic_mean = np.random.normal(1.0, 0.0003, 1)[0]

    Y = [stochastic_mean]
    for t in range(1, N):
        innovation = np.random.normal(stochastic_mean, 0.001, 1)[0]
        new_Y = innovation

        if not use_constant_mean:
            tmp = np.random.normal(0.0, 0.0003, 1)[0]
            stochastic_mean += tmp

        Y.append(new_Y)

    return np.array(Y)
