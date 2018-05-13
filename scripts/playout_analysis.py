import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple, defaultdict
import pickle

moves = []

state = 'default'

letters = [i for i in 'ABCDEFGHJKLMNOPQRST']
numbers = [i for i in range(1,20)]
assert len(letters) == 19
assert len(numbers) == 19
coords = [i+str(j) for i in letters for j in numbers]
coord_to_idx = {coords[i]:i for i in range(len(coords))}
coord_to_idx['pass'] = 361

for filename in sys.argv[1:]:
    print filename
    current_move = []
    with open(filename, 'r') as f:
        pols = None
        nns = [0]*362
        visits = 0
        for line in f:
            if state == 'policy':
                # Read policy scores
                if ':' not in line:
                    state = 'default'
                    continue
                line = line.split(':')
                move = line[0].strip()
                val = float(line[1])
                pols[coord_to_idx[move]] = val
            if 'Policy scores' in line:
                state = 'policy'
                pols = [0]*362
                continue
            if 'Thinking at most ' in line:
                # Start a new move
                if current_move != []:
                    moves.append(current_move)
                current_move = {}
            if '->' in line:
                # Add simulation to the current move
                coord = line[:4].strip()
                nn = float(line[45:50].strip())/100.0
                vi = float(line[58:65].strip())
                visits = vi
                nns[coord_to_idx[coord]] = nn
            if 'non leaf nodes' in line:
                # Visits increased
                current_move[visits] = zip(nns, pols)
                nns = [0]*362

    # Add the last move
    moves.append(current_move)

print len(moves), 'moves in game'

# MSE of best move training data change
mses = []
# Change in best moves training data
nn_changes = []
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

# For uniform sampling use visits that the first move got
# Sometimes LZ prints stats at non-specified visits. Leave these out to maintain uniform sampling

ref_visits = sorted(moves[0].keys())
max_visits = max(ref_visits)

#for e, m in enumerate(moves):
#    for v in ref_visits:
#        nn, pol = zip(*m[v])
#        nn = list(nn)
#        for i in range(len(nn)):
#            b = 0.3*(300-v)/300.
#            if b > 0:
#                nn[i] = pol[i]*b + nn[i]*(1-b)
#        moves[e][v] = zip(nn, pol)

for move in moves:
    # Best move
    best_nn = max(move[max_visits][i][0] for i in range(362))
    best_pol = max(move[max_visits][i][1] for i in range(362))
    best_idx = np.argmax([move[max_visits][i][0] for i in range(362)])
    pol_idx = np.argmax([move[max_visits][i][1] for i in range(362)])
    
    nn_change = []
    mse = []
    changed = []
    loss = defaultdict(lambda: 0)

    ref = {}

    entropy = 0
    for s in move[max_visits]:
        if s[0] > 0 and s[1] > 0:
            # Loss of policy head (0 visits)
            loss[0] += -s[0]*np.log(s[1]/s[0])
        elif s[0] > 0:
            raise ValueError("Non-zero visits but 0 policy")

        # Policy head entropy
        # Illegal moves have 0 policy
        if s[1] > 0:
            entropy += -s[1]*np.log(s[1])

    policy_entropy.append(entropy)

    for v in ref_visits:
        chosen_idx = np.argmax([move[v][i][0] for i in range(362)])
        mse_acc = 0
        for e, (nn, pol) in enumerate(move[v]):
            ref = move[max_visits][e]
            ref_nn = ref[0]
            ref_pol = ref[1]
            if ref_nn > 0 and nn > 0:
                loss[v] += -ref_nn*np.log(nn/ref_nn)
            elif ref_nn > 0 and nn == 0:
                loss[v] += -ref_nn*np.log((1.0/max_visits)/ref_nn)
            mse_acc += (ref_nn - nn)**2

        mse.append( mse_acc )
        changed.append( chosen_idx != best_idx )
        nn_change.append( 100*(move[v][best_idx][0] - move[v][best_idx][1]) )

    max_pol = np.max([move[max_visits][i][1] for i in range(362)])

    policy_chosen_move.append(move[max_visits][best_idx][1])
    policy_max.append(max_pol)

    for e, c in enumerate(changed):
        if c == False:
            visits_to_change.append(ref_visits[e])
            break

    policy_loss.append([loss[i] for i in sorted(loss.keys())])
    if changed[0]:
        policy_loss_changed.append([loss[i] for i in sorted(loss.keys())])
    mses.append(mse)
    nn_changes.append(nn_change)
    move_changed.append(changed)

if any(len(policy_loss[i]) != len(policy_loss[0]) for i in range(len(policy_loss))):
    raise ValueError("Moves have different visit counts")

pickle.dump((ref_visits, policy_loss, policy_loss_changed, move_changed, nn_changes, policy_entropy), open('policy.p', 'w'))

policy_loss_avg = np.average(np.array(policy_loss), axis=0)
policy_loss_std = np.std(np.array(policy_loss), axis=0)
policy_loss_avg_changed = np.average(np.array(policy_loss_changed), axis=0)
policy_loss_std_changed = np.std(np.array(policy_loss_changed), axis=0)
move_changed_avg = np.average(np.array(move_changed), axis=0)

avg_mse = np.average(mses, axis=0)
nn_avg = np.average(nn_changes, axis=0)
nn_std = np.std(nn_changes, axis=0)

plt.figure()
plt.title('Best move training data change')
plt.xlabel('Visits')
plt.ylabel('Change (percentage points)')
#for i in nn_changes:
#    plt.plot(ref_visits, i, 'b-', alpha=0.05)
plt.plot(ref_visits, nn_avg, 'b-')#, alpha=1.0, linewidth=3)

plt.figure()
plt.title('Best move training data change per visit')
plt.xlabel('Visits')
plt.ylabel('Change (percentage points)')
plt.plot(ref_visits, np.array(nn_avg)/ref_visits)

if 0:
    plt.figure()
    plt.title('MSE of the best move training data')
    plt.xlabel('Visits')
    plt.ylabel('MSE')
    plt.plot(ref_visits, avg_mse)

    plt.figure()
    plt.title('MSE/visits of the best move training data')
    plt.plot(ref_visits, avg_mse/ref_visits)
    plt.xlabel('Visits')
    plt.ylabel('MSE/visits')

visits0 = [0] + ref_visits

plt.figure()
plt.title('Policy loss')
#for i in policy_loss:
#    plt.plot(visits0, i, 'b-', alpha=0.05)
plt.plot(visits0, policy_loss_avg, 'b-')#, alpha=1.0, linewidth=3)
plt.ylim([0,1])
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
    plt.plot(ref_visits, (policy_loss_avg[0]-np.array(policy_loss_avg[1:]))/ref_visits)
    plt.xlabel('Visits')
    plt.ylabel('Policy loss change')

if 1:
    plt.figure()
    plt.title('Policy loss change per one visit (changed moves only)')
    plt.plot(ref_visits, (policy_loss_avg_changed[0]-np.array(policy_loss_avg_changed[1:]))/ref_visits)
    plt.xlabel('Visits')
    plt.ylabel('Policy loss change')

opt = (policy_loss_avg[0]-np.array(policy_loss_avg[1:]))/ref_visits
opt_changed = (policy_loss_avg_changed[0]-np.array(policy_loss_avg_changed[1:]))/ref_visits

print max(opt), ref_visits[np.argmax(opt)]
print opt[np.searchsorted(ref_visits, 3200)]

print max(opt_changed), ref_visits[np.argmax(opt_changed)]

plt.figure()
plt.title('Move changed')
plt.plot(ref_visits, move_changed_avg)
plt.xlabel('Visits')

if 0:
    max_sorted, chosen_sorted = zip(*sorted(zip(policy_max, policy_chosen_move)))
    entropy_sorted, move_changed_entropy = zip(*sorted(zip(policy_entropy, visits_to_change)))

    plt.figure()
    plt.title('Policy of the chosen move and maximum policy')
    plt.plot(chosen_sorted)
    plt.plot(max_sorted)
    plt.xlabel('Move')

    plt.figure()
    plt.title('Entropy of the value head vs changed move')
    plt.scatter(entropy_sorted, move_changed_entropy)
    plt.xlabel('Entropy')

if 0:
    opts = (np.array(policy_loss)[:,0][:,np.newaxis]-np.array(policy_loss)[:,1:])/ref_visits
    opt_visits = np.array(ref_visits)[np.argmax(opts, axis=1)]

    entropy_sorted, opt_visits_sorted = zip(*sorted(zip(policy_entropy, opt_visits)))
    
    plt.figure()
    plt.scatter(entropy_sorted, opt_visits)
    plt.xlabel('Policy head entropy')
    plt.ylabel('Optimum number of visits')

plt.show()
