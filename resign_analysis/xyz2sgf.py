#!/usr/bin/python3


# "Universal" GIB, NGF, UGF --> SGF converter
# Copyright the author: Ask on GitHub if you
# want to redistribute (make a new issue).
#
# Homepage:
# https://github.com/fohristiwhirl/xyz2sgf
#
# This standalone converter is based on my larger Go library at:
# https://github.com/fohristiwhirl/gofish


from __future__ import with_statement
from __future__ import absolute_import
import os, sys
from io import open


class BadBoardSize(Exception): pass
class ParserFail(Exception): pass
class UnknownFormat(Exception): pass

EMPTY, BLACK, WHITE = 0, 1, 2



class Node(object):
    def __init__(self, parent):
        self.properties = dict()
        self.children = []
        self.parent = parent

        if parent:
            parent.children.append(self)

    def safe_commit(self, key, value):      # Note: destroys the key if value is ""
        safe_s = safe_string(value)
        if safe_s:
            self.properties[key] = [safe_s]
        else:
            try:
                self.properties.pop(key)
            except KeyError:
                pass

    def add_value(self, key, value):        # Note that, if improperly used, could lead to odd nodes like ;B[ab][cd]
        if key not in self.properties:
            self.properties[key] = []
        if unicode(value) not in self.properties[key]:
            self.properties[key].append(unicode(value))

    def set_value(self, key, value):        # Like the above, but only allows the node to have 1 value for this key
        self.properties[key] = [unicode(value)]


# ---------------------------------------------------------------------


def string_from_point(x, y):                        # convert x, y into SGF coordinate e.g. "pd"
    if x < 1 or x > 26 or y < 1 or y > 26:
        raise ValueError
    s = u""
    s += unichr(x + 96)
    s += unichr(y + 96)
    return s


def safe_string(s):     # "safe" meaning safely escaped \ and ] characters
    s = unicode(s)
    safe_s = u""
    for ch in s:
        if ch in [u"\\", u"]"]:
            safe_s += u"\\"
        safe_s += ch
    return safe_s


def handicap_points(boardsize, handicap, tygem = False):

    points = set()

    if boardsize < 4:
        return points

    if handicap > 9:
        handicap = 9

    if boardsize < 13:
        d = 2
    else:
        d = 3

    if handicap >= 2:
        points.add((boardsize - d, 1 + d))
        points.add((1 + d, boardsize - d))

    # Experiments suggest Tygem puts its 3rd handicap stone in the top left

    if handicap >= 3:
        if tygem:
            points.add((1 + d, 1 + d))
        else:
            points.add((boardsize - d, boardsize - d))

    if handicap >= 4:
        if tygem:
            points.add((boardsize - d, boardsize - d))
        else:
            points.add((1 + d, 1 + d))

    if boardsize % 2 == 0:      # No handicap > 4 on even sided boards
        return points

    mid = (boardsize + 1) // 2

    if handicap in [5, 7, 9]:
        points.add((mid, mid))

    if handicap in [6, 7, 8, 9]:
        points.add((1 + d, mid))
        points.add((boardsize - d, mid))

    if handicap in [8, 9]:
        points.add((mid, 1 + d))
        points.add((mid, boardsize - d))

    return points


def load(filename):

    # FileNotFoundError is just allowed to bubble up
    # All the parsers below can raise ParserFail

    if filename[-4:].lower() == u".gib":

        # These seem to come in many encodings; just try UTF-8 with replacement:

        with open(filename, encoding=u"utf8", errors=u"replace") as infile:
            contents = infile.read()
        root = parse_gib(contents)

    elif filename[-4:].lower() == u".ngf":

        # These seem to usually be in GB18030 encoding:

        with open(filename, encoding=u"gb18030", errors=u"replace") as infile:
            contents = infile.read()
        root = parse_ngf(contents)

    elif filename[-4:].lower() in [u".ugf", u".ugi"]:

        # These seem to usually be in Shift-JIS encoding:

        with open(filename, encoding=u"shift_jisx0213", errors=u"replace") as infile:
            contents = infile.read()
        root = parse_ugf(contents)

    else:
        print u"Couldn't detect file type -- make sure it has an extension of .gib, .ngf, .ugf or .ugi"
        raise UnknownFormat

    root.set_value(u"FF", 4)
    root.set_value(u"GM", 1)
    root.set_value(u"CA", u"UTF-8")   # Force UTF-8

    if u"SZ" in root.properties:
        size = int(root.properties[u"SZ"][0])
    else:
        size = 19
        root.set_value(u"SZ", u"19")

    if size > 19 or size < 1:
        raise BadBoardSize

    return root


def save_file(filename, root):      # Note: this version of the saver requires the root node
    with open(filename, u"w", encoding=u"utf-8") as outfile:
        write_tree(outfile, root)


def write_tree(outfile, node):      # Relies on values already being correctly backslash-escaped
    outfile.write(u"(")
    while 1:
        outfile.write(u";")
        for key in node.properties:
            outfile.write(key)
            for value in node.properties[key]:
                outfile.write(u"[{}]".format(value))
        if len(node.children) > 1:
            for child in node.children:
                write_tree(outfile, child)
            break
        elif len(node.children) == 1:
            node = node.children[0]
            continue
        else:
            break
    outfile.write(u")\n")
    return


def parse_ugf(ugf):     # Note that the files are often (always?) named .ugi

    root = Node(parent = None)
    node = root

    boardsize = None
    handicap = None

    handicap_stones_set = 0

    coordinate_type = u""

    lines = ugf.split(u"\n")

    section = None

    for line in lines:

        line = line.strip()

        try:
            if line[0] == u"[" and line[-1] == u"]":

                section = line.upper()

                if section == u"[DATA]":

                    # Since we're entering the data section, we need to ensure we have
                    # gotten sane info from the header; check this now...

                    if handicap is None or boardsize is None:
                        raise ParserFail
                    if boardsize < 1 or boardsize > 19 or handicap < 0:
                        raise ParserFail

                continue

        except IndexError:
            pass

        if section == u"[HEADER]":

            if line.upper().startswith(u"HDCP="):
                try:
                    handicap_str = line.split(u"=")[1].split(u",")[0]
                    handicap = int(handicap_str)
                    if handicap >= 2:
                        root.set_value(u"HA", handicap)      # The actual stones are placed in the data section

                    komi_str = line.split(u"=")[1].split(u",")[1]
                    komi = float(komi_str)
                    root.set_value(u"KM", komi)
                except:
                    continue

            elif line.upper().startswith(u"SIZE="):
                size_str = line.split(u"=")[1]
                try:
                    boardsize = int(size_str)
                    root.set_value(u"SZ", boardsize)
                except:
                    continue

            elif line.upper().startswith(u"COORDINATETYPE="):
                coordinate_type = line.split(u"=")[1].upper()

            # Note that the properties that aren't being converted to int/float need to use the .safe_commit() method...

            elif line.upper().startswith(u"PLAYERB="):
                root.safe_commit(u"PB", line[8:])

            elif line.upper().startswith(u"PLAYERW="):
                root.safe_commit(u"PW", line[8:])

            elif line.upper().startswith(u"PLACE="):
                root.safe_commit(u"PC", line[6:])

            elif line.upper().startswith(u"TITLE="):
                root.safe_commit(u"GN", line[6:])

            # Determine the winner...

            elif line.upper().startswith(u"WINNER=B"):
                root.set_value(u"RE", u"B+")

            elif line.upper().startswith(u"WINNER=W"):
                root.set_value(u"RE", u"W+")

        elif section == u"[DATA]":

            line = line.upper()

            slist = line.split(u",")
            try:
                x_chr = slist[0][0]
                y_chr = slist[0][1]
                colour = slist[1][0]
            except IndexError:
                continue

            try:
                node_chr = slist[2][0]
            except IndexError:
                node_chr = u""

            if colour not in [u"B", u"W"]:
                continue

            if coordinate_type == u"IGS":        # apparently "IGS" format is from the bottom left
                x = ord(x_chr) - 64
                y = (boardsize - (ord(y_chr) - 64)) + 1
            else:
                x = ord(x_chr) - 64
                y = ord(y_chr) - 64

            if x > boardsize or x < 1 or y > boardsize or y < 1:    # Likely a pass, "YA" is often used as a pass
                value = u""
            else:
                try:
                    value = string_from_point(x, y)
                except ValueError:
                    continue

            # In case of the initial handicap placement, don't create a new node...

            if handicap >= 2 and handicap_stones_set != handicap and node_chr == u"0" and colour == u"B" and node is root:
                handicap_stones_set += 1
                key = u"AB"
                node.add_value(key, value)      # add_value not set_value
            else:
                node = Node(parent = node)
                key = colour
                node.set_value(key, value)

    if len(root.children) == 0:     # We'll assume we failed in this case
        raise ParserFail

    return root


def parse_ngf(ngf):

    ngf = ngf.strip()
    lines = ngf.split(u"\n")

    try:
        boardsize = int(lines[1])
        handicap = int(lines[5])
        pw = lines[2].split()[0]
        pb = lines[3].split()[0]
        rawdate = lines[8][0:8]
        komi = float(lines[7])

        if handicap == 0 and int(komi) == komi:
            komi += 0.5

    except (IndexError, ValueError):
        boardsize = 19
        handicap = 0
        pw = u""
        pb = u""
        rawdate = u""
        komi = 0

    re = u""
    try:
        if u"hite win" in lines[10]:
            re = u"W+"
        elif u"lack win" in lines[10]:
            re = u"B+"
    except:
        pass

    if handicap < 0 or handicap > 9:
        raise ParserFail

    root = Node(parent = None)
    node = root

    # Set root values...

    root.set_value(u"SZ", boardsize)

    if handicap >= 2:
        root.set_value(u"HA", handicap)
        stones = handicap_points(boardsize, handicap, tygem = True)     # While this isn't Tygem, uses same layout I think
        for point in stones:
            root.add_value(u"AB", string_from_point(point[0], point[1]))

    if komi:
        root.set_value(u"KM", komi)

    if len(rawdate) == 8:
        ok = True
        for n in xrange(8):
            if rawdate[n] not in u"0123456789":
                ok = False
        if ok:
            date = rawdate[0:4] + u"-" + rawdate[4:6] + u"-" + rawdate[6:8]
            root.set_value(u"DT", date)

    if pw:
        root.safe_commit(u"PW", pw)
    if pb:
        root.safe_commit(u"PB", pb)

    if re:
        root.set_value(u"RE", re)

    # Main parser...

    for line in lines:
        line = line.strip().upper()

        if len(line) >= 7:
            if line[0:2] == u"PM":
                if line[4] in [u"B", u"W"]:

                    key = line[4]

                    # Coordinates are from 1-19, but with "B" representing
                    # the digit 1. (Presumably "A" would represent 0.)

                    x = ord(line[5]) - 65       # Therefore 65 is correct
                    y = ord(line[6]) - 65

                    try:
                        value = string_from_point(x, y)
                    except ValueError:
                        continue

                    node = Node(parent = node)
                    node.set_value(key, value)

    if len(root.children) == 0:     # We'll assume we failed in this case
        raise ParserFail

    return root


def parse_gib(gib):

    root = Node(parent = None)
    node = root

    lines = gib.split(u"\n")

    for line in lines:
        line = line.strip()

        if line.startswith(u"\\[GAMEBLACKNAME=") and line.endswith(u"\\]"):
            s = line[16:-2]
            root.safe_commit(u"PB", s)

        if line.startswith(u"\\[GAMEWHITENAME=") and line.endswith(u"\\]"):
            s = line[16:-2]
            root.safe_commit(u"PW", s)

        if line.startswith(u"\\[GAMERESULT="):
            score = None
            strings = line.split()
            for s in strings:           # This is very crude
                try:
                    score = float(s)
                except:
                    pass
            if u"white" in line.lower() and u"black" not in line.lower():
                if u"resignation" in line.lower():
                    root.set_value(u"RE", u"W+R")
                elif score:
                    root.set_value(u"RE", u"W+{}".format(score))
                else:
                    root.set_value(u"RE", u"W+")
            if u"black" in line.lower() and u"white" not in line.lower():
                if u"resignation" in line.lower():
                    root.set_value(u"RE", u"B+R")
                elif score:
                    root.set_value(u"RE", u"B+{}".format(score))
                else:
                    root.set_value(u"RE", u"B+")

        if line.startswith(u"\\[GAMECONDITION="):
            if u"black 6.5 dum" in line.lower():     # Just hard-coding the typical case; we should maybe extract komi by regex
                root.set_value(u"KM", 6.5)
            elif u"black 7.5 dum" in line.lower():   # Perhaps komi becomes 7.5 in the future...
                root.set_value(u"KM", 7.5)
            elif u"black 0.5 dum" in line.lower():   # Do these exist on Tygem?
                root.set_value(u"KM", 0.5)

        if line[0:3] == u"INI":

            if node is not root:
                raise ParserFail

            setup = line.split()

            try:
                handicap = int(setup[3])
            except IndexError:
                continue

            if handicap < 0 or handicap > 9:
                raise ParserFail

            if handicap >= 2:
                node.set_value(u"HA", handicap)
                stones = handicap_points(19, handicap, tygem = True)
                for point in stones:
                    node.add_value(u"AB", string_from_point(point[0], point[1]))

        if line[0:3] == u"STO":

            move = line.split()

            key = u"B" if move[3] == u"1" else u"W"

            # Although one source claims the coordinate system numbers from the bottom left in range 0 to 18,
            # various other pieces of evidence lead me to believe it numbers from the top left (like SGF).
            # In particular, I tested some .gib files on http://gokifu.com

            try:
                x = int(move[4]) + 1
                y = int(move[5]) + 1
            except IndexError:
                continue

            try:
                value = string_from_point(x, y)
            except ValueError:
                continue

            node = Node(parent = node)
            node.set_value(key, value)

    if len(root.children) == 0:     # We'll assume we failed in this case
        raise ParserFail

    return root


def main():

    if len(sys.argv) == 1:
        print u"Usage: {} <list of input files>".format(os.path.basename(sys.argv[0]))
        return

    for filename in sys.argv[1:]:
        try:
            root = load(filename)
            outfilename = filename + u".sgf"
            save_file(outfilename, root)
        except:
            try:
                print u"Conversion failed for {}".format(filename)
            except:
                print u"Conversion failed for file with unprintable filename"


if __name__ == u"__main__":
    main()

