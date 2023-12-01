import numpy as np

prob = []
for i in range(6000):
    prob.append(np.random.random(200))
res = np.concatenate(prob)