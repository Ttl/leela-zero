import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

moves = []
current_move = []
choices = []

# coord = Coordinate of the played move
# n = Prior policy head output
# nn = Training data output
# vi = Total visits for all moves
Move = namedtuple('Move', ['coord', 'n', 'nn', 'vi'])

with open(sys.argv[1], 'r') as f:
    for line in f:
        if 'Thinking at most ' in line:
            # Start a new move
            if current_move != []:
                moves.append(current_move)
            current_move = []
        if '->' in line:
            # Add simulation to the current move
            coord = line[:4].strip()
            n = float(line[31:37].strip())
            nn = float(line[45:50].strip())
            vi = float(line[58:65].strip())
            choices.append( Move(coord=coord, n=n, nn=nn, vi=vi) )
        if 'non leaf nodes' in line:
            # Visits increased
            if choices != []:
                current_move.append(choices)
            choices = []

# Add the last move
moves.append(current_move)

print len(moves), 'moves in game'

mses = []
nn_changes = []
visits = []
move_changed = []

for move in moves:
    # Best move
    best = move[-1][0]

    prior = best.n
    nn_change = []
    mse = []
    v = []
    changed = []
    for m in move:
        for s in m:
            if s.coord == best.coord:
                changed.append( best.coord == m[0].coord )
                mse.append( (s.nn - best.nn)**2 )
                nn_change.append( s.nn - best.nn )
                v.append(s.vi)
                break

    mses.append(mse)
    nn_changes.append(nn_change)
    move_changed.append(changed)
    visits.append(v)

avg_mse = np.array(mses[0])
avg_nn = np.array(nn_changes[0])

s = 0
for e in range(1, len(mses)):
    if len(mses[e]) == len(avg_mse):
        s += 1
        avg_mse += np.array(mses[e])
        avg_nn += np.array(nn_changes[e])

print s, 'moves'
avg_mse /= s
avg_nn /= s

plt.figure()
plt.title('MSE of the best move training data')
plt.xlabel('Visits')
plt.ylabel('MSE')
plt.plot(visits[0], avg_mse)

plt.figure()
plt.title('Best move training data change')
plt.xlabel('Visits')
plt.ylabel('Change')
plt.plot(visits[0], avg_nn)

plt.figure()
plt.title('MSE/visits of the best move training data')
plt.plot(visits[0], avg_mse/visits[0])
plt.xlabel('Visits')
plt.ylabel('MSE/visits')

plt.show()
