#!/usr/bin/env python3
import gzip
import sys
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy.optimize import curve_fit

def parse_chunk(chunk):
    root_eval = []
    winrates, scores = [], []
    for e, line in enumerate(chunk):
        if e == 0:
            # Version
            continue
        if e == 1:
            # Training chunk name
            continue
        line = list(map(float, line.split(' ')))
        if e % 3 == 2:
            root_eval.append(line[-1])
        elif e % 3 == 0:
            # Subtract root eval
            winrates.append([i - root_eval[-1] if i != -1 else 0 for i in line])
        else:
            scores.append(line)
    out = []
    for i in range(len(winrates)):
        r = [root_eval[i]]*len(winrates[i])
        out.append(zip(winrates[i], scores[i], r))
    return out

chunks = glob.glob(sys.argv[1] + "*.gz")

if len(chunks) == 0:
    print("No chunks found")
    exit()

positions = []

for chunk in chunks:
    with gzip.open(chunk, 'r') as f:
        positions.extend(parse_chunk(f))

fpus = []
scores = []
root_evals = []
for i in positions:
    for j in i:
        fpus.append(j[0])
        scores.append(j[1])
        root_evals.append(j[2])

fpus = np.array(fpus)
scores = np.array(scores)
root_evals = np.array(root_evals)

reductions = np.linspace(-0.5, 0.5, 20)
l2_loss = []
for reduction in reductions:
    l2_loss.append(1.0/(len(fpus))*np.sum( (fpus - reduction)**2 ))

def func(x, a):
    return a*x[1]

#def func(x, a):
#    return a

popt, pcov = curve_fit(func, [scores, root_evals], fpus)
print(popt)
fpu_est = lambda x: func(x, *popt)

l2 = 0
for i in range(len(fpus)):
    l2 += (fpus[i] - fpu_est((scores[i], root_evals[i])))**2
l2 /= len(fpus)
print(l2, min(l2_loss))

print('Best constant reduction: {}'.format(reductions[np.argmin(l2_loss)]))

print(fpu_est((2e-2,0.5)))

plt.figure()
plt.hist(fpus, bins=50)
plt.xlabel('FPU')

plt.figure()
plt.title('L2 loss of constant FPU reduction')
plt.plot(reductions, l2_loss)
plt.xlabel('FPU reduction constant')
plt.ylabel('L2 loss')

plt.figure()
scores_sorted, fpus_sorted = zip(*sorted(zip(scores, fpus)))


plt.scatter(scores_sorted, fpus_sorted)
x1 = np.linspace(min(scores), max(scores), 500)
x2_1 = np.array([0.1]*len(x1))
x2_2 = np.array([0.5]*len(x1))
x2_3 = np.array([0.9]*len(x1))
y1 = [fpu_est(i) for i in zip(x1, x2_1)]
y2 = [fpu_est(i) for i in zip(x1, x2_2)]
y3 = [fpu_est(i) for i in zip(x1, x2_3)]
plt.plot(x1, y1, 'g')
plt.plot(x1, y2, 'g')
plt.plot(x1, y3, 'g')
plt.xlabel('Policy')
plt.ylabel('FPU')

plt.figure()
root_evals_sorted, fpus_sorted = zip(*sorted(zip(root_evals, fpus)))
x1 = np.linspace(min(root_evals), max(root_evals), 200)
x2_1 = np.array([0.01]*len(x1))
x2_2 = np.array([0.1]*len(x1))
x2_3 = np.array([0.9]*len(x1))
y1 = [fpu_est(i) for i in zip(x2_1, x1)]
y2 = [fpu_est(i) for i in zip(x2_2, x1)]
y3 = [fpu_est(i) for i in zip(x2_3, x1)]
plt.scatter(root_evals_sorted, fpus_sorted)
plt.plot(x1, y1, 'g')
plt.plot(x1, y2, 'g')
plt.plot(x1, y3, 'g')
plt.xlabel('Root eval')
plt.ylabel('FPU')

plt.show()
