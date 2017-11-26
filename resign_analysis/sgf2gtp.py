#! /usr/bin/env python

import sys
import argparse
import re

import sgf

DEBUG = False

parser = argparse.ArgumentParser( formatter_class=argparse.RawDescriptionHelpFormatter,
description="""
This script converts SGF files to GTP format so that you can feed them
to Pachi, insert genmove at the right places etc. Might not work on
obscure SGF files.

When called with FILENAMES argument, it will create according output
files with .gtp extension instead of .sgf, unless --no-gtp option is
specified.

Otherwise the games are read from standard input. Note that the stdin
in this case is read in at once, so for very large collections, it is
better to run this script separately for each sgf file.

example:
    cat *.sgf | %s -g -n 5
"""%(sys.argv[0]))

parser.add_argument('FILENAMES', help='List of sgf games to process.', nargs='*', default=[])
parser.add_argument('-g', help='Automatically append genmove command for the other color.', action='store_true')
parser.add_argument('-n', help='Output at most first MOVENUM moves.', metavar='MOVENUM', type=int, default=10**10)
parser.add_argument('--stdout-only', help='Do not create the .gtp files from FILENAMES, print everything to stdout.', action='store_true')
args = vars(parser.parse_args())

class UnknownNode(Exception):
    pass

def get_atr(node, atr):
    try:
        return node.properties[atr][0]
    except KeyError:
        return None

def get_setup(node, atr):
    try:
        return node.properties[atr][:]
    except KeyError:
        return None

def col2num(column, board_size):
    a, o, z = map(ord, ['a', column, 'z'])
    if a <= o <= z:
        return a + board_size - o
    raise Exception( "Wrong column character: '%s'"%(column,) )

def is_pass_move(coord, board_size):
    # the pass move is represented either by [] ( = empty coord )
    # OR by [tt] (for boards <= 19 only)
    return len(coord) == 0 or ( board_size <= 19 and coord == 'tt' )

def process_gametree(gametree, fout):
    header = gametree.nodes[0]

    # first node is the header

    handicap = get_atr(header, 'HA')
    try:
        board_size = int(get_atr(header, 'SZ'))
    except:
        board_size = 19
    komi = get_atr(header, 'KM')
    player_next, player_other = "B", "W"
    setup_black = get_setup(header, 'AB')
    setup_white = get_setup(header, 'AW')

    print >>fout, "boardsize", board_size
    print >>fout, "clear_board"
    if komi:
        print >>fout, "komi", komi

    if handicap and handicap != '0':
        print >>fout, "fixed_handicap", handicap
        player_next, player_other = player_other, player_next

    if setup_black:
	for item in setup_black:
	    x, y = item
	    if x >= 'i':
		x = chr(ord(x)+1)
	    y = str(col2num(y, board_size))
	    print >>fout, "play B", x+y
    if setup_white:
	for item in setup_white:
	    x, y = item
	    if x >= 'i':
		x = chr(ord(x)+1)
	    y = str(col2num(y, board_size))
	    print >>fout, "play W", x+y

    def print_game_step(coord):
        if is_pass_move(coord, board_size):
            print >>fout, "play", player_next, "pass"
        else:
            x, y = coord
            # The reason for this incredibly weird thing is that
            # the GTP protocol excludes `i` in the coordinates
            # (do you see any purpose in this??)
            if x >= 'i':
                x = chr(ord(x)+1)
            y = str(col2num(y, board_size))
            print >>fout, "play", player_next, x+y

    movenum = 0

    # cursor for tree traversal
    c = header

    # walk the game tree forward
    while True:
        # sgf2gtp.pl ignores n = 0
        #while c.next_variation != None:
        #    c = c.next_variation
        c = c.next
        if c == None or (args['n'] and movenum >= args['n']):
            break
        movenum += 1

        coord = get_atr(c, player_next)
        if coord != None:
            print_game_step(coord)
        else:
            # MAYBE white started?
            # or one of the players plays two time in a row
            player_next, player_other = player_other, player_next
            coord = get_atr(c, player_next)
            if coord != None:
                print_game_step(coord)
            else:
                # TODO handle weird sgf files better
                print 'Unknown node', c.properties.keys()
                #raise UnknownNode

        player_next, player_other = player_other, player_next

    if args['g']:
        print >>fout, "genmove", player_next

def process_sgf_file(fin, fout):

    with open(fin) as f:
        col = sgf.parse(f.read())

    for gametree in col.children:
        process_gametree(gametree, fout)

if __name__ == "__main__":
    process_sgf_file(sys.argv[1], sys.stdout)
