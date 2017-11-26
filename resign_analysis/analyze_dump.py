from __future__ import division
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats

def new_game_length(e, r):
    for n, i in enumerate(e):
        if i > 1-r:
            return n+1, False
        if i < r:
            return n+1, True
    return n+1, False

evals = pickle.load(open('evals.p', 'r'))

error_pct = 0.95

#plt.figure()
#plt.title('Winner\'s value network output')
#plt.xlabel('Game length [%]')
#plt.ylabel('NN eval [%]')

min_w = []
min_l = []
for e in evals:
    min_w.append(min(e))
    min_l.append(1-max(e))
    #plt.plot(np.linspace(0, 100, len(e)), e)

mean_w = np.mean(min_w)
std_w = np.std(min_w)

mean_l = np.mean(min_l)
std_l = np.std(min_l)

print 'Dataset size {} games'.format(len(evals))
print 'Minimum observed evaluation for winner {}'.format(min(min_w))
print 'Minimum evaluation in game for winner mean={}, std={}'.format(mean_w, std_w)
print 'Minimum evaluation in game for loser mean={}, std={}'.format(mean_l, std_l)

r = max(0, stats.norm.ppf(1-error_pct, loc=mean_w, scale=std_w))

print '{}% confidence resignation rate {}'.format(error_pct*100, r)

plt.hist(min_w, bins=50)
plt.title('Minimum NN eval for winner')
#plt.hist(min_l, bins=50)

x = np.linspace(0, 1, 100)
plt.plot(x, stats.norm.pdf(x, loc=mean_w, scale=std_w))
#plt.plot(x, stats.norm.pdf(x, loc=mean_l, scale=std_l))

game_eval = []
incorrect_resigns = 0
game_length_before = 0
game_length_after = 0
resigned_games = 0
for e in evals:
    l, change = new_game_length(e, r)
    game_length_before += len(e)
    game_length_after += l
    if len(e) != l:
        resigned_games += 1
    if change:
        incorrect_resigns += 1

game_length_before /= len(evals)
game_length_after /= len(evals)

print 'Incorrect resignations in dataset with the suggested resign rate: {:.2f}%'.format(100*incorrect_resigns/len(evals))
print 'Average game length without resignations: {}, after resignations: {}'.format(int(game_length_before), int(game_length_after))
print 'Average game length reduction: {:.2f}%'.format(100*(game_length_before-game_length_after)/game_length_before)
print '{:.2f}% games resigned'.format(100*resigned_games/len(evals))

plt.show()
