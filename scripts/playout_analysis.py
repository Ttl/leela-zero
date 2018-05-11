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

# MSE of best move training data change
mses = []
# Change in best moves training data
nn_changes = []
# Number of visits for each move
visits = []
# Move does not equal the best move for each visit
move_changed = []
# KL divergence of all moves as function of visits
policy_loss = []
# KL divergence of the changed moves as function of visits
policy_loss_changed = []

# Policy head output for the best move
policy_chosen_move = []
# Maximum value of the policy head output
policy_max = []

# Minimum number of visits to change to the best move
visits_to_change = []

# Entropy of the policy head output
policy_entropy = []

ref_visits = set([])

# For uniform sampling use visits that the first move got
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

    max_visits = max(ref_visits)

    entropy = 0
    for s in move[-1]:
        ref[s.coord] = s
        if s.nn > 0:
            loss[0] += -s.nn*np.log(s.pol/s.nn)
        entropy += -s.pol*np.log(s.pol)

    policy_entropy.append(entropy)

    for m in move:
        found = False
        for s in m:
            if s.vi not in ref_visits:
                continue
            if ref[s.coord].nn > 0 and s.nn > 0:
                loss[s.vi] += -ref[s.coord].nn*np.log(s.nn/ref[s.coord].nn)
            elif ref[s.coord].nn > 0 and s.nn == 0:
                # Use small value for nn if it got 0 visits
                loss[s.vi] += -ref[s.coord].nn*np.log((1.0/max_visits)/ref[s.coord].nn)
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

policy_loss_avg = np.average(np.array(policy_loss), axis=0)
policy_loss_avg_changed = np.average(np.array(policy_loss_changed), axis=0)
move_changed_avg = np.average(np.array(move_changed), axis=0)

avg_mse = np.array(mses[0])
avg_nn = np.array(nn_changes[0])

s = 0
for e in range(1, len(mses)):
    if len(mses[e]) == len(avg_mse):
        s += 1
        avg_mse += np.array(mses[e])
        avg_nn += np.array(nn_changes[e])

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
plt.plot(visits0, policy_loss_avg)
plt.xlabel('Visits')
plt.ylabel('Policy loss')

if 0:
    plt.figure()
    plt.title('Policy loss diff')
    plt.plot(visits0[1::], np.diff(policy_loss_avg))
    plt.xlabel('Visits')
    plt.ylabel('Difference')

if 1:
    plt.figure()
    plt.title('Policy loss vs visits (changed moves)')
    plt.plot(visits0, policy_loss_avg_changed)
    plt.xlabel('Visits')
    plt.ylabel('Policy loss')

if 1:
    plt.figure()
    plt.title('Policy loss change per one visit')
    plt.plot(visits[0], (policy_loss_avg[0]-np.array(policy_loss_avg[1:]))/visits[0])
    plt.xlabel('Visits')
    plt.ylabel('Policy loss change')

if 1:
    plt.figure()
    plt.title('Policy loss change per one visit (changed moves only)')
    plt.plot(visits[0], (policy_loss_avg_changed[0]-np.array(policy_loss_avg_changed[1:]))/visits[0])
    plt.xlabel('Visits')
    plt.ylabel('Policy loss change')

opt = (policy_loss_avg_changed[0]-np.array(policy_loss_avg_changed[1:]))/visits[0]

print max(opt), visits[0][np.argmax(opt)]
print opt[np.searchsorted(visits[0], 3200)]

plt.figure()
plt.title('Move changed')
plt.plot(visits[0], move_changed_avg)
plt.xlabel('Visits')

max_sorted, chosen_sorted = zip(*sorted(zip(policy_max, policy_chosen_move)))
entropy_sorted, move_changed_entropy = zip(*sorted(zip(policy_entropy, visits_to_change)))

print entropy_sorted
print move_changed_entropy

plt.figure()
plt.title('Policy of the chosen move and maximum policy')
plt.plot(chosen_sorted)
plt.plot(max_sorted)
plt.xlabel('Move')

plt.figure()
plt.title('Entropy of the value head vs changed move')
plt.scatter(entropy_sorted, move_changed_entropy)
plt.xlabel('Entropy')
plt.show()
