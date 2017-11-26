import sys, os
from subprocess import Popen, PIPE
import sgf2gtp
from StringIO import StringIO
import matplotlib.pyplot as plt
import numpy as np
import pickle

gtp_path = '../src/leelaz'
gtp_cwd = os.path.dirname(gtp_path)
sgf_path = 'sgf'

def usage():
    print 'python sgf-analyze.py "leelaz args"'
    print
    print 'Put sgf files played without resignation to folder \'./sgf\'.'
    print 'ARGS example: "--noponder -g -p 1 -w <weights>"'

def get_nn_eval(cmds):

    gtp_cmd = [gtp_path]
    gtp_cmd.extend(args)

    p = Popen(gtp_cmd, stdout=PIPE, stdin=PIPE, stderr=PIPE, cwd=gtp_cwd)

    setup = ''
    setup_i = 0
    for line in cmds:
        if 'play' in line:
            break
        setup += line+'\n'
        setup_i += 1

    evals = []
    i = 0
    p.stdin.write(setup)

    for move in cmds[setup_i:-1]:
        player = move.split(' ')[1]
        out = 'genmove {}\nundo\n'.format(player)

        p.stdin.write(out)

        nn_eval = None
        for line in iter(p.stderr.readline, ""):
            if 'NN eval' in line:
                nn_eval = float(line.split('=')[1])
                break

        p.stdin.write(move+'\n')

        i += 1

        if player == 'W':
            nn_eval = 1 - nn_eval

        evals.append(nn_eval)
    p.stdin.write('quit\n')
    p.wait()

    return evals

if __name__ == '__main__':

    if len(sys.argv) != 1:
        args = sys.argv[1].strip().split(' ')
    else:
        usage()
        exit()

    plt.figure()
    plt.title('Winner\'s value network output')
    plt.xlabel('Game length [%]')
    plt.ylabel('NN eval [%]')

    evals = []

    i = 0
    for sgf_file in os.listdir(sgf_path):
        if os.path.splitext(sgf_file)[1] != '.sgf':
            continue
        with open(os.path.join(sgf_path, sgf_file), 'r') as f:
            line = f.readline()
            re = line.find('RE[')
            if re > 0:
                winner = line[re+3:re+4]
            else:
                print 'No winner information in file: ', sgf_file
                continue
        i = i+1
        print i
        gtp = StringIO()
        sgf2gtp.process_sgf_file(os.path.join(sgf_path, sgf_file), gtp)
        cmds = gtp.getvalue().split('\n')

        e = get_nn_eval(cmds)
        if winner == 'W':
            e = [1-j for j in e]
        evals.append(e)
        x = np.linspace(0, 100, len(e))

        plt.plot(x, e)

    pickle.dump(evals, open('evals.p', 'wb'))
    plt.show()
