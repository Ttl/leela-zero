/*
    This file is part of Leela Zero.
    Copyright (C) 2018 Henrik Forsten

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <vector>
#include <algorithm>
#include <cassert>

#include "config.h"
#include "Ladder.h"

#include "Utils.h"

using namespace Utils;


Ladder::LadderStatus Ladder::ladder_status(const FastState &state) {
    const auto board = state.board;

    Ladder::LadderStatus status;

    for (auto i = 0; i < BOARD_SIZE; i++) {
        for (auto j = 0; j < BOARD_SIZE; j++) {
            auto vertex = board.get_vertex(i, j);
            status[i][j] = NO_LADDER;
            if (ladder_capture(state, vertex)) {
                status[i][j] = CAPTURE;
            }
            if (ladder_escape(state, vertex)) {
                status[i][j] = ESCAPE;
            }
        }
    }
    return status;
}

bool Ladder::ladder_capture(const FastState &state, int vertex, int group, int depth) {

    const auto &board = state.board;
    const auto capture_player = board.get_to_move();
    const auto escape_player = board.get_not_to_move();

    if (!state.is_move_legal(capture_player, vertex)) {
        return false;
    }

    // Assume that capture succeeds if it takes this long
    if (depth >= 100) {
        return true;
    }

    std::vector<int> groups_in_ladder;

    if (group == FastBoard::PASS) {
        // Check if there are nearby groups with 2 liberties
        for (int d = 0; d < 4; d++) {
            int n_vtx = board.get_square_neighbor(vertex, d);
            int n = board.get_state(n_vtx);
            if ((n == escape_player) && (board.get_liberties(n_vtx) == 2)) {
                auto parent = board.m_parent[n_vtx];
                if (std::find(groups_in_ladder.begin(), groups_in_ladder.end(), parent) == groups_in_ladder.end()) {
                    groups_in_ladder.emplace_back(parent);
                }
            }
        }
    } else {
        groups_in_ladder.emplace_back(group);
    }

    for (auto& group : groups_in_ladder) {
        auto state_copy = std::make_unique<FastState>(state);
        auto &board_copy = state_copy->board;

        state_copy->play_move(vertex);

        int escape = FastBoard::PASS;
        int newpos = group;
        do {
            for (int d = 0; d < 4; d++) {
                int stone = newpos + board_copy.m_dirs[d];
                // If the surrounding stones are in atari capture fails
                if (board_copy.m_state[stone] == capture_player) {
                    if (board_copy.get_liberties(stone) == 1) {
                        return false;
                    }
                }
                // Possible move to escape
                if (board_copy.m_state[stone] == FastBoard::EMPTY) {
                    escape = stone;
                }
            }
            newpos = board_copy.m_next[newpos];
        } while (newpos != group);
        
        assert(escape != FastBoard::PASS);

        // If escaping fails the capture was successful
        if (!ladder_escape(*state_copy, escape, group, depth + 1)) {
            return true;
        }
    }

    return false;
}

bool Ladder::ladder_escape(const FastState &state, const int vertex, int group, int depth) {
    const auto &board = state.board;
    const auto escape_player = board.get_to_move();

    if (!state.is_move_legal(escape_player, vertex)) {
        return false;
    }

    // Assume that escaping failed if it takes this long
    if (depth >= 100) {
        return false;
    }

    std::vector<int> groups_in_ladder;

    if (group == FastBoard::PASS) {
        // Check if there are nearby groups with 1 liberties
        for (int d = 0; d < 4; d++) {
            int n_vtx = board.get_square_neighbor(vertex, d);
            int n = board.get_state(n_vtx);
            if ((n == escape_player) && (board.get_liberties(n_vtx) == 1)) {
                auto parent = board.m_parent[n_vtx];
                if (std::find(groups_in_ladder.begin(), groups_in_ladder.end(), parent) == groups_in_ladder.end()) {
                    groups_in_ladder.emplace_back(parent);
                }
            }
        }
    } else {
        groups_in_ladder.emplace_back(group);
    }

    for (auto& group : groups_in_ladder) {
        auto state_copy = std::make_unique<FastState>(state);
        auto &board_copy = state_copy->board;

        state_copy->play_move(vertex);

        if (board_copy.get_liberties(group) >= 3) {
            // Opponent can't atari on the next turn
            return true;
        }

        if (board_copy.get_liberties(group) == 1) {
            // Will get captured on the next turn
            return false;
        }

        // Still two liberties left, check for possible captures
        int newpos = group;
        do {
            for (int d = 0; d < 4; d++) {
                int empty = newpos + board_copy.m_dirs[d];
                if (board_copy.m_state[empty] == FastBoard::EMPTY) {
                    if (ladder_capture(*state_copy, empty, group, depth + 1)) {
                        // Got captured
                        return false;
                    }
                }
            }
            newpos = board_copy.m_next[newpos];
        } while (newpos != group);

        // Ladder capture failed, escape succeeded
        return true;
    }

    return false;
}

static void print_columns() {
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (i < 25) {
            myprintf("%c ", (('a' + i < 'i') ? 'a' + i : 'a' + i + 1));
        }
        else {
            myprintf("%c ", (('A' + (i - 25) < 'I') ? 'A' + (i - 25) : 'A' + (i - 25) + 1));
        }
    }
    myprintf("\n");
}

void Ladder::display_ladders(const LadderStatus &status) {
    myprintf("\n   ");
    print_columns();
    for (int j = BOARD_SIZE-1; j >= 0; j--) {
        myprintf("%2d", j+1);
        myprintf(" ");
        for (int i = 0; i < BOARD_SIZE; i++) {
            if (status[i][j] == CAPTURE) {
                myprintf("C");
            } else if (status[i][j] == ESCAPE) {
                myprintf("E");
            } else if (FastBoard::starpoint(BOARD_SIZE, i, j)) {
                myprintf("+");
            } else {
                myprintf(".");
            }
            myprintf(" ");
        }
        myprintf("%2d\n", j+1);
    }
    myprintf("   ");
    print_columns();
    myprintf("\n");
}

void Ladder::display_ladders(const FastState &state) {
    display_ladders(ladder_status(state));
}
