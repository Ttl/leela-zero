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
            winrates.append([i - root_eval[-1] for i in line])
        else:
            scores.append([np.log(max(line)/(i + 1e-5)) for i in line])
    out = []
    for i in range(len(winrates)):
        out.append(zip(winrates[i], scores[i]))
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
for i in positions:
    for j in i:
        fpus.append(j[0])
        scores.append(j[1])

fpus = np.array(fpus)
scores = np.array(scores)

reductions = np.linspace(-0.5, 0.5, 20)
l2_loss = []
for reduction in reductions:
    l2_loss.append(np.sum( (fpus - reduction)**2 ))

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

if 0:
    coeff = np.polyfit(scores, fpus, 1)
    fpu_est = np.poly1d(coeff)
else:
    popt, pcov = curve_fit(func, scores, fpus)
    print popt
    fpu_est = lambda x: func(x, *popt)

l2 = np.sum( (fpus - fpu_est(scores))**2 )
print l2, min(l2_loss)

print('Best constant reduction: {}'.format(reductions[np.argmin(l2_loss)]))

plt.figure()
plt.hist(fpus, bins=50)

plt.figure()
plt.title('L2 loss of constant FPU reduction')
plt.plot(reductions, l2_loss)
plt.xlabel('FPU reduction constant')
plt.ylabel('L2 loss')

plt.figure()
plt.scatter(scores, fpus)
x = np.linspace(min(scores), max(scores), 100)
y = fpu_est(x)
plt.plot(x, y, 'g')
plt.show()
