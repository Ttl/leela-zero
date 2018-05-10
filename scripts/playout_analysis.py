import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple, defaultdict

moves = []
current_move = []
choices = []

# coord = Coordinate of the played move
# n = Prior policy head output
# nn = Training data output
# vi = Total visits for all moves
Move = namedtuple('Move', ['coord', 'n', 'nn', 'vi', 'pol'])
state = 'default'

with open(sys.argv[1], 'r') as f:
    for line in f:
        if state == 'policy':
            # Read policy scores
            if ':' not in line:
                state = 'default'
                continue
            line = line.split(':')
            move = line[0].strip()
            val = float(line[1])
            policy[move] = val
        if 'Policy scores' in line:
            state = 'policy'
            policy = {}
            continue
        if 'Thinking at most ' in line:
            # Start a new move
            if current_move != []:
                moves.append(current_move)
            current_move = []
        if '->' in line:
            # Add simulation to the current move
            coord = line[:4].strip()
            n = float(line[31:37].strip())/100.0
            nn = float(line[45:50].strip())/100.0
            vi = float(line[58:65].strip())
            pol = policy[coord]
            choices.append( Move(coord=coord, n=n, nn=nn, vi=vi, pol=pol) )
        if 'non leaf nodes' in line:
            # Visits increased
            if choices != []:
                # Last visit count is printed twice. Drop the duplicate.
                if len(current_move) == 0 or choices[0].vi != current_move[-1][0].vi:
                    current_move.append(choices)
            choices = []

# Add the last move
moves.append(current_move)

print len(moves), 'moves in game'

mses = []
nn_changes = []
visits = []
move_changed = []
policy_loss = []
policy_loss_changed = []

policy_chosen_move = []
policy_max = []
visits_to_change = []

ref_visits = set([])

# Sometimes LZ prints stats at non-specified visits. Leave these out to maintain uniform sampling

for m in moves[0]:
    for s in m:
        ref_visits.add(s.vi)

for move in moves:
    # Best move
    best = move[-1][0]
    
    nn_change = []
    mse = []
    v = []
    changed = []
    loss = defaultdict(lambda: 0)

    ref = {}

    for s in move[-1]:
        ref[s.coord] = s
        loss[0] += -s.nn*np.log(s.pol/s.nn)

    for m in move:
        found = False
        for s in m:
            if s.vi not in ref_visits:
                continue
            if ref[s.coord].nn > 0 and s.nn > 0:
                loss[s.vi] += -ref[s.coord].nn*np.log(s.nn/ref[s.coord].nn)
            elif ref[s.coord].nn > 0 and s.nn == 0:
                # Use small value for nn if it got 0 visits
                loss[s.vi] += -ref[s.coord].nn*np.log((1/8000.)/ref[s.coord].nn)
            else:
                loss[s.vi] += 0
            if s.coord == best.coord:
                changed.append( best.coord != m[0].coord )
                mse.append( (s.nn - s.pol)**2 )
                nn_change.append( s.nn - s.pol )
                v.append(s.vi)
                found = True
        if not found:
            changed.append( True )
            mse.append( (s.pol)**2 )
            nn_change.append( -s.pol )
            v.append(s.vi)

    max_pol = 0

    for m in move[-1]:
        if m.pol > max_pol:
            max_pol = m.pol
            
    policy_chosen_move.append(best.pol)
    policy_max.append(max_pol)

    for e, c in enumerate(changed):
        if c == False:
            visits_to_change.append(v[e])
            break

    policy_loss.append([loss[i] for i in sorted(loss.keys())])
    if changed[0]:
        policy_loss_changed.append([loss[i] for i in sorted(loss.keys())])
    mses.append(mse)
    nn_changes.append(nn_change)
    move_changed.append(changed)
    visits.append(v)

if any(len(policy_loss[i]) != len(policy_loss[0]) for i in range(len(policy_loss))):
    raise ValueError("Moves have different visit counts")

policy_loss = np.average(np.array(policy_loss), axis=0)
policy_loss_changed = np.average(np.array(policy_loss_changed), axis=0)
move_changed = np.average(np.array(move_changed), axis=0)

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
plt.title('Best move training data change')
plt.xlabel('Visits')
plt.ylabel('Change (percentage points)')
plt.plot(visits[0], avg_nn)

if 0:
    plt.figure()
    plt.title('MSE of the best move training data')
    plt.xlabel('Visits')
    plt.ylabel('MSE')
    plt.plot(visits[0], avg_mse)

    plt.figure()
    plt.title('MSE/visits of the best move training data')
    plt.plot(visits[0], avg_mse/visits[0])
    plt.xlabel('Visits')
    plt.ylabel('MSE/visits')

visits0 = [0] + visits[0]

plt.figure()
plt.title('Policy loss')
plt.plot(visits0, policy_loss)
plt.xlabel('Visits')
plt.ylabel('Policy loss')

plt.figure()
plt.title('Policy loss diff')
plt.plot(visits0[1::], np.diff(policy_loss))
plt.xlabel('Visits')
plt.ylabel('Difference')

if 1:
    plt.figure()
    plt.title('Policy loss vs visits (changed moves)')
    plt.plot(visits0, policy_loss_changed)
    plt.xlabel('Visits')
    plt.ylabel('Policy loss')

if 1:
    plt.figure()
    plt.title('Policy loss change per one visit (changed moves only)')
    plt.plot(visits[0], (policy_loss_changed[0]-np.array(policy_loss_changed[1:]))/visits[0])
    plt.xlabel('Visits')
    plt.ylabel('Policy loss change')

plt.figure()
plt.title('Move changed')
plt.plot(visits[0], move_changed)
plt.xlabel('Visits')

chosen_sorted, max_sorted = zip(*sorted(zip(policy_max, policy_chosen_move)))

plt.figure()
plt.title('Policy of the chosen move and maximum policy')
plt.plot(chosen_sorted)
plt.plot(max_sorted)
plt.xlabel('Move')
plt.show()
