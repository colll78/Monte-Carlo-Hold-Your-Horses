#!/usr/bin/env python
""" A Monte Carlo Search Tree utilizing Rapid Action Value Estimation
This code was based on the following publications:
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.86.5248&rep=rep1&type=pdf
https://users.soe.ucsc.edu/~dph/mypubs/AMAFpaperWithRef.pdf
By Sylvain Gelly", "David Silver", and "David P. Helmbold", "Aleatha Parker-Wood respectively.

This program implements only a fraction of the Monte Carlo Search Tree improvements detailed in published works.
Originally, I had planned to implement a much more significant portion of the publications, however, while I
was able to understand the majority of the theory behind these improvements, implementing them proved much more
difficult than I had imagined. I was particularly interested in implementing value-based reinforcement learning,
and spent a great deal of time attempting to do so to no avail, and I ultimately gave up after many hours of
debugging. If you would like to see this additional work, please let me know. Additionally, I have implemented
a version of the suggested alpha-beta pruning minimax search, and while it is significantly stronger than
knight_rider, it is no-where near the playing strength of this program. In an 100-game match, this program managed to
beat dark_knight in 80 out of 120 (66.67%) of matches.

Future improvements I hope to include are:
root node parallelization
use of transposition table

UPDATE:
I have implemented root node parallelization without the need for locks or any other bottle-necks, the number of
simulations ran is therefore doubled for each new process.
"""

from collections import defaultdict
import concurrent.futures
from datetime import datetime, timedelta
import numpy as np
import queue as qu
import math

__author__ = "Philip DiSarro"
__credits__ = ["Sylvain Gelly", "David Silver", "David P. Helmbold", "Aleatha Parker-Wood"]
__version__ = "1.0"
__maintainer__ = "Philip DiSarro"
__email__ = "philip.disarro@phabulous.org"
__status__ = "Production"


boardWidth = 7            # These 3 global variables are set by the getMove function. This is not...
boardHeight = 6           # ... an elegant solution but an efficient one.
timeLimit = 2.0           # Maximum thinking time (in seconds) for each move

startState = None         # Initial state, provided to the initPlayer function
assignedPlayer = 1        # 1 -> player MAX; -1 -> player MIN (in terms of the MiniMax algorithm)
startTime = 0

victoryPoints = 0         # Number of points for the winner
moveLimit = 0             # Maximum number of moves
                          # If exceeded, game is a tie; otherwise, number of remaining moves is added to winner's score.

pointMultiplier = 10       # Muliplier for winner's points in getScore function
pieceValue = 20            # Score value of a single piece in getScore function
victoryScoreThresh = 1000  # An absolute score exceeds this value if and only if one player has won
minLookAhead = 1           # Initial search depth for iterative deepening
maxLookAhead = 20          # Maximum search depth
apple_loc = None
mating_squares = {1: [(2, 1), (1, 2)], -1: [(4, 4), (5, 3)]}
reachable = {(0, 0): [(2, 1), (1, 2)], (0, 1): [(2, 0), (2, 2), (1, 3)], (0, 2): [(1, 0), (2, 1), (2, 3), (1, 4)], (0, 3): [(1, 1), (2, 2), (2, 4), (1, 5)], (0, 4): [(1, 2), (2, 3), (2, 5)], (0, 5): [(1, 3), (2, 4)], (1, 0): [(3, 1), (2, 2), (0, 2)], (1, 1): [(3, 0), (3, 2), (2, 3), (0, 3)], (1, 2): [(2, 0), (3, 1), (3, 3), (2, 4), (0, 4), (0, 0)], (1, 3): [(2, 1), (3, 2), (3, 4), (2, 5), (0, 5), (0, 1)], (1, 4): [(2, 2), (3, 3), (3, 5), (0, 2)], (1, 5): [(2, 3), (3, 4), (0, 3)], (2, 0): [(4, 1), (3, 2), (1, 2), (0, 1)], (2, 1): [(4, 0), (4, 2), (3, 3), (1, 3), (0, 2), (0, 0)], (2, 2): [(3, 0), (4, 1), (4, 3), (3, 4), (1, 4), (0, 3), (0, 1), (1, 0)], (2, 3): [(3, 1), (4, 2), (4, 4), (3, 5), (1, 5), (0, 4), (0, 2), (1, 1)], (2, 4): [(3, 2), (4, 3), (4, 5), (0, 5), (0, 3), (1, 2)], (2, 5): [(3, 3), (4, 4), (0, 4), (1, 3)], (3, 0): [(5, 1), (4, 2), (2, 2), (1, 1)], (3, 1): [(5, 0), (5, 2), (4, 3), (2, 3), (1, 2), (1, 0)], (3, 2): [(4, 0), (5, 1), (5, 3), (4, 4), (2, 4), (1, 3), (1, 1), (2, 0)], (3, 3): [(4, 1), (5, 2), (5, 4), (4, 5), (2, 5), (1, 4), (1, 2), (2, 1)], (3, 4): [(4, 2), (5, 3), (5, 5), (1, 5), (1, 3), (2, 2)], (3, 5): [(4, 3), (5, 4), (1, 4), (2, 3)], (4, 0): [(6, 1), (5, 2), (3, 2), (2, 1)], (4, 1): [(6, 0), (6, 2), (5, 3), (3, 3), (2, 2), (2, 0)], (4, 2): [(5, 0), (6, 1), (6, 3), (5, 4), (3, 4), (2, 3), (2, 1), (3, 0)], (4, 3): [(5, 1), (6, 2), (6, 4), (5, 5), (3, 5), (2, 4), (2, 2), (3, 1)], (4, 4): [(5, 2), (6, 3), (6, 5), (2, 5), (2, 3), (3, 2)], (4, 5): [(5, 3), (6, 4), (2, 4), (3, 3)], (5, 0): [(6, 2), (4, 2), (3, 1)], (5, 1): [(6, 3), (4, 3), (3, 2), (3, 0)], (5, 2): [(6, 0), (6, 4), (4, 4), (3, 3), (3, 1), (4, 0)], (5, 3): [(6, 1), (6, 5), (4, 5), (3, 4), (3, 2), (4, 1)], (5, 4): [(6, 2), (3, 5), (3, 3), (4, 2)], (5, 5): [(6, 3), (3, 4), (4, 3)], (6, 0): [(5, 2), (4, 1)], (6, 1): [(5, 3), (4, 2), (4, 0)], (6, 2): [(5, 4), (4, 3), (4, 1), (5, 0)], (6, 3): [(5, 5), (4, 4), (4, 2), (5, 1)], (6, 4): [(4, 5), (4, 3), (5, 2)], (6, 5): [(4, 4), (5, 3)]}
knight_distance = {}
tile_value = {}
executor = None
guard_tiles = {1: {(3, 1), (0, 2), (3, 3), (4, 0), (1, 3), (2, 0), (2, 4), (0, 4), (4, 2)}, -1: {(3, 2), (4, 1), (2, 3), (6, 1), (4, 5), (2, 5), (6, 3), (3, 4), (5, 2)}}


class GameState(object):
    __slots__ = ['board', 'playerToMove', 'gameOver', 'movesRemaining', 'points', 'winner', 'curr_move']


def p1_apple(state):
    for xStart in range(boardWidth):  # Search board for player's pieces
        for yStart in range(boardHeight):
            if state.board[xStart, yStart] == 2:
                return (xStart, yStart)


def p2_apple(state):
    for xStart in range(boardWidth):  # Search board for player's pieces
        for yStart in range(boardHeight):
            if state.board[xStart, yStart] == -2:
                return (xStart, yStart)


def getMoveOptions(state):
    direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]    # Possible (dx, dy) moves
    moves = []
    for xStart in range(boardWidth):                                    # Search board for player's pieces
        for yStart in range(boardHeight):
            # print("PLAYER TO MOVE", state.playerToMove)
            if state.board[xStart, yStart] == state.playerToMove:       # Found a piece!
                for (dx, dy) in direction:                              # Check all potential move vectors
                    (xEnd, yEnd) = (xStart + dx, yStart + dy)
                    if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight and not (state.board[xEnd, yEnd] in [state.playerToMove, 2 * state.playerToMove]):
                        moves.append((xStart, yStart, xEnd, yEnd))      # If square is empty or occupied by the opponent, then we have a legal move.
    return moves


def attackers_defenders(state, x, y):
    direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]
    attack_defend = 0
    for (dx, dy) in direction:
        (xEnd, yEnd) = (x + dx, y + dy)
        if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight:
            if state.board[xEnd, yEnd] == state.playerToMove:
                attack_defend += 1
            elif state.board[xEnd, yEnd] == -state.playerToMove:
                attack_defend -= 1
    return attack_defend-1


def get_weighted_moves(state):
    direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]    # Possible (dx, dy) moves
    moves = []
    for xStart in range(boardWidth):                                    # Search board for player's pieces
        for yStart in range(boardHeight):
            if state.board[xStart, yStart] == state.playerToMove:       # Found a piece!
                for (dx, dy) in direction:                              # Check all potential move vectors
                    (xEnd, yEnd) = (xStart + dx, yStart + dy)
                    if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight and not (state.board[xEnd, yEnd] in [state.playerToMove, 2 * state.playerToMove]):
                        moves.append((xStart, yStart, xEnd, yEnd))      # If square is empty or occupied by the opponent, then we have a legal move.
    return moves


# def get_simulation_moves(state):
#     moves = getMoveOptions(state)
#     remove_guard = True
#     filtered_moves = []
#     atk_moves = []
#     for x in moves:
#         if state.board[x[2], x[3]] == state.playerToMove * -2:
#             return [x]
#         elif (x[2], x[3]) in mating_squares[-state.playerToMove]:
#             if all(state.board[square] != -1 * state.playerToMove for square in mating_squares[state.playerToMove]):
#                 atk_def = attackers_defenders(state, x[2], x[3])
#                 if atk_def >= 0:
#                     return [x]
#                 # elif remove_guard and atk_def == -1:
#                 #     remove_guard = False
#                 #     for tile in guard_tiles[-state.playerToMove]:
#                 #         if state.board[tile[0], tile[1]] == -state.playerToMove:
#                 #             for rtile in reachable[(tile[0], tile[1])]:
#                 #                 if state.board[rtile[0], rtile[1]] == state.playerToMove:
#                 #                     # print(state.board)
#                 #                     # print((rtile[0], rtile[1], tile[0], tile[1]))
#                 #                     return [(rtile[0], rtile[1], tile[0], tile[1])]
#         elif (x[2], x[3]) in mating_squares[state.playerToMove] and state.board[x[2], x[3]] == -state.playerToMove:
#             return [x]
#         # elif (x[0], x[1]) in guard_tiles[state.playerToMove]:
#         #     for tile in mating_squares[state.playerToMove]:
#         #         atk_def = attackers_defenders(state, tile[0], tile[1])
#         #         if atk_def <= -1:
#         #             continue
#         # if attackers_defenders(state, x[2], x[3]) >= 0 or state.board[x[2], x[3]] == -state.playerToMove:
#         #     filtered_moves.append(x)
#         if state.board[x[2], x[3]] == -state.playerToMove or attackers_defenders(state, x[2], x[3]) >= 0:
#             filtered_moves.append(x)
#         # elif attackers_defenders(state, x[2], x[3]) >= 0 or state.board[x[2], x[3]] == -state.playerToMove:
#         #     atk_moves.append(x)
#     # if not filtered_moves:
#     #     print("NO GOOD MOVES")
#     #     print(moves)
#     #     print(state.board)
#     return filtered_moves or moves

def get_simulation_moves(state):
    moves = getMoveOptions(state)
    filtered_moves = []
    atk_moves = []
    for x in moves:
        if state.board[x[2], x[3]] == state.playerToMove * -2:
            return [x]
        elif (x[2], x[3]) in mating_squares[-state.playerToMove]:
            if all(state.board[square] != -1 * state.playerToMove for square in mating_squares[state.playerToMove]):
                if attackers_defenders(state, x[2], x[3]) >= 0:
                    return [x]
        elif (x[2], x[3]) in mating_squares[state.playerToMove] and state.board[x[2], x[3]] == -state.playerToMove:
            return [x]
        # if attackers_defenders(state, x[2], x[3]) >= 0 or state.board[x[2], x[3]] == -state.playerToMove:
        #     filtered_moves.append(x)
        if state.board[x[2], x[3]] == -state.playerToMove and attackers_defenders(state, x[2], x[3]) >= 0:
            filtered_moves.append(x)
        elif attackers_defenders(state, x[2], x[3]) >= 0 or state.board[x[2], x[3]] == -state.playerToMove:
            atk_moves.append(x)
    # if not filtered_moves:
    #     print("NO GOOD MOVES")
    #     print(moves)
    #     print(state.board)
    return filtered_moves or atk_moves or moves

def makeMove(state, move):
    (xStart, yStart, xEnd, yEnd) = move
    newState = GameState()
    newState.board = np.copy(state.board)  # The new board configuration is a copy of the current one except that...
    newState.board[xStart, yStart] = 0  # ... we remove the moving piece from its start position...
    newState.board[xEnd, yEnd] = state.playerToMove  # ... and place it at the end position
    newState.playerToMove = -state.playerToMove  # After this move, it will be the opponent's turn
    newState.movesRemaining = state.movesRemaining - 1
    newState.gameOver = False
    newState.winner = None
    newState.curr_move = move
    newState.points = 0

    if state.board[xEnd, yEnd] == -2 * state.playerToMove or not (-state.playerToMove in newState.board):
        newState.gameOver = True  # If the opponent lost the apple or all horses, the game is over...
        newState.points = state.playerToMove * (
                    victoryPoints + newState.movesRemaining)  # ... and more remaining moves result in more points
        newState.winner = state.playerToMove
    elif newState.movesRemaining == 0:  # Otherwise, if there are no more moves left, the game is drawn
        newState.gameOver = True
        newState.winner = 0

    return newState


# wiki says sqrt(2) is theoretical best, but research shows higher performance engines benefit from lower value
exploration_constant = .25
rave_constant = 0.001
#rave_constant = 0.2
result_rewards = {0: 0.002, 1: 1, -1: 0}


class HorseHoldRaveSearchNode():
    move: object
    parent: object
    children: list
    visit_count: int
    score: float
    _moves_to_try: list

    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.move = move
        self.visit_count = 0
        self._moves_to_try = None
        self.score = 0.0
        self._n_sims_with_move = 0
        self._move_win_count = 0.0

    @property
    def move_win_count(self):
        return self._move_win_count

    @property
    def num_simulations_containing_move(self):
        return self._n_sims_with_move

    @property
    def moves_to_try(self):
        if self._moves_to_try is None:
            self._moves_to_try = get_simulation_moves(self.state)
        return self._moves_to_try

    @property
    def num_visits(self):
        return self.visit_count

    @property
    def node_score(self):
        return self.score

    def expand(self):
        action = self.moves_to_try.pop()
        next_state = makeMove(self.state, action)
        child_node = HorseHoldRaveSearchNode(next_state, parent=self, move=action)
        self.children.append(child_node)
        return child_node

    def is_expansion_complete(self):
        return len(self.moves_to_try) == 0

    def is_terminal_leaf(self):
        return self.state.gameOver

    def run_simulation(self):
        possible_moves = set()
        current_rollout_state = self.state
        while not current_rollout_state.gameOver:
            possible_moves = get_simulation_moves(current_rollout_state)
            action = self.simulation_move_policy(possible_moves)
            current_rollout_state = makeMove(current_rollout_state, action)
        return current_rollout_state.winner, possible_moves

    def backpropagate(self, simulation_result):
        outcome, possible_moves = simulation_result
        self.visit_count += 1.0
        result_for_self = -1
        if self.parent:
            result_for_self = outcome * self.parent.state.playerToMove
        self.score += result_rewards[result_for_self]
        if self.parent:
            for child in self.parent.children:
                if child.move in possible_moves:
                    child_result = outcome * self.state.playerToMove
                    child._move_win_count += result_rewards[child_result]
                    child._n_sims_with_move += 1
            self.parent.backpropagate(simulation_result)

    def get_consolidation_data(self):
        return self.children, self.num_visits

    def get_child_weights(self, exploration_c=exploration_constant):
        def decaying_weight(child_node):
            return child_node.num_simulations_containing_move / (child_node.num_visits + child_node.num_simulations_containing_move + 4 * rave_constant ** 2 * child_node.num_visits * child_node.num_simulations_containing_move)

        choices_weights = []
        for c in self.children:
            decaying_w = decaying_weight(c)
            if decaying_w:
                weight = (1 - decaying_w) * (c.node_score / c.num_visits) + decaying_w * (c.move_win_count / c.num_simulations_containing_move) + exploration_c * np.sqrt((2 * np.log(self.num_visits) / c.num_visits))
            else:
                weight = c.node_score / c.num_visits + exploration_c * np.sqrt((2 * np.log(self.num_visits) / c.num_visits))
            choices_weights.append(weight)
        return choices_weights

    def best_child(self, exploration_c=exploration_constant):
        """
        Rapid Action Value Estimation should be used heavily initially, however, the more times a node is visited,
        the less impact RAVE should have on the node's score. As the limit of visit approaches infinity, the score
        function should approach UCT. I achieve this by using this decaying weight.
        :return:
        """
        def decaying_weight(child_node):
            return child_node.num_simulations_containing_move / (child_node.num_visits + child_node.num_simulations_containing_move + 4 * rave_constant ** 2 * child_node.num_visits * child_node.num_simulations_containing_move)

        choices_weights = []
        for c in self.children:
            decaying_w = decaying_weight(c)
            if decaying_w:
                weight = (1 - decaying_w) * (c.node_score / c.num_visits) + decaying_w * (c.move_win_count / c.num_simulations_containing_move) + exploration_c * np.sqrt((2 * np.log(self.num_visits) / c.num_visits))
            else:
                weight = c.node_score / c.num_visits + exploration_c * np.sqrt((2 * np.log(self.num_visits) / c.num_visits))
            choices_weights.append(weight)
        return self.children[np.argmax(choices_weights)]

    @staticmethod
    def simulation_move_policy(possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]


def time_out():
    duration = datetime.now() - startTime
    return duration.seconds + duration.microseconds * 1e-6 >= timeLimit


def time_out_close():
    duration = datetime.now() - (startTime - timedelta(milliseconds=200))
    return duration.seconds + duration.microseconds * 1e-6 >= timeLimit


from queue import Queue


class HorseHoldSearchTree:
    def __init__(self, node: HorseHoldRaveSearchNode):
        self.root = node

    def multi_threaded_best_action(self, num_simulations=2000):
        # TODO: store results in queue so that the search can be parallelize
        # for _ in range(0, 100):
        #     c = self.expansion()
        #     points = c.run_simulation()
        #     c.backpropagate(points)
        # weights = self.root.get_child_weights(exploration_c=0)
        #i = 0
        while not time_out_close():
            #i+= 1
            c = self.expansion()
            points = c.run_simulation()
            c.backpropagate(points)

        #print("Number of simulations: ", i)
        weights = self.root.get_child_weights(exploration_c=0)
        return weights

    def multi_threaded_consolidate(self):
        while not time_out_close():
            c = self.expansion()
            points = c.run_simulation()
            c.backpropagate(points)
        return self.root.get_consolidation_data()

    def best_action(self, num_simulations=2000, queue=None):
        # TODO: store results in queue so that the search can be parallelize
        for _ in range(0, 400):
            c = self.expansion()
            points = c.run_simulation()
            c.backpropagate(points)

        best = self.root.best_child(exploration_c=0)
        for _ in range(401, num_simulations):
            c = self.expansion()
            points = c.run_simulation()
            c.backpropagate(points)
            if _ % 100 == 0:
                best = self.root.best_child(exploration_c=0)
            if _ % 20 == 0 and time_out():
                print("ParallelRave TimeOut at simulation :", _)
                return best
        best = self.root.best_child(exploration_c=0)
        return best

    def expansion(self):
        """
        choose node to run simulations on.
        wiki refers to this as the selection phase
        :return: the chosen node
        """
        curr_node = self.root
        while not curr_node.is_terminal_leaf():
            if not curr_node.is_expansion_complete():
                return curr_node.expand()
            else:
                curr_node = curr_node.best_child()
        return curr_node


def knight_dist(start_x, start_y, goal_x, goal_y):
    if start_x == goal_x and start_y == goal_y:
        return 0
    x = abs(goal_x - start_x)
    y = abs(goal_y - start_y)
    # print("Horizontal Distance: ", x)
    # print("Vertical Distance: ", y)
    if x == y == 1 and ((start_x, start_y) == (0, 0) or (start_x, start_y) == (boardWidth-1, boardHeight-1) or (start_x, start_y) == (0, boardHeight-1) or (start_x, start_y) == (boardWidth-1, 0)):
        return 4
    if x + y == 1:
        return 3
    if x == y == 2:
        return 4
    else:
        m = math.ceil(max((x / 2), y / 2, (x + y) / 3))
        result = m + ((m + x + y) % 2)
        return int(result)


def around(x, y):
    direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]
    result = []
    for (dx, dy) in direction:
        (xEnd, yEnd) = (x + dx, y + dy)
        if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight:
            result.append((xEnd, yEnd))
    return result


# Set global variables and initialize any data structures that the player will need
def initPlayer(_startState, _timeLimit, _victoryPoints, _moveLimit, _assignedPlayer):
    global startState, timeLimit, victoryPoints, moveLimit, assignedPlayer, boardWidth, boardHeight, apple_loc, mating_squares, knight_distance, reachable, tile_value, executor, guard_tiles
    startState, timeLimit, victoryPoints, moveLimit, assignedPlayer = _startState, _timeLimit, _victoryPoints, _moveLimit, _assignedPlayer
    (boardWidth, boardHeight) = startState.board.shape
    p1_apple_loc = p1_apple(startState)
    p2_apple_loc = p2_apple(startState)
    apple_loc = {1: p1_apple_loc, -1: p2_apple_loc}
    reachable = {}

    direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]
    p1_mating_squares = []
    for (dx, dy) in direction:
        (xEnd, yEnd) = (p1_apple_loc[0] + dx, p1_apple_loc[1] + dy)
        if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight:
            p1_mating_squares.append((xEnd, yEnd))

    p1_guard_tiles = set()
    for tile in p1_mating_squares:
        for (dx, dy) in direction:
            (xEnd, yEnd) = (tile[0] + dx, tile[1] + dy)
            if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight and startState.board[
                xEnd, yEnd] != 2:
                p1_guard_tiles.add((xEnd, yEnd))

    p2_mating_squares = []
    for (dx, dy) in direction:
        (xEnd, yEnd) = (p2_apple_loc[0] + dx, p2_apple_loc[1] + dy)
        if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight:
            p2_mating_squares.append((xEnd, yEnd))

    p2_guard_tiles = set()
    for tile in p2_mating_squares:
        for (dx, dy) in direction:
            (xEnd, yEnd) = (tile[0] + dx, tile[1] + dy)
            if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight and startState.board[
                xEnd, yEnd] != -2:
                p2_guard_tiles.add((xEnd, yEnd))

    guard_tiles = {1: p1_guard_tiles, -1: p2_guard_tiles}

    mating_squares = {1: p1_mating_squares, -1: p2_mating_squares}

    for x in range(0, boardWidth):
        for y in range(0, boardHeight):
            if not reachable.get((x,y)):
                reachable[(x,y)] = around(x,y)
            if not knight_distance.get((x, y, 1)) and not knight_distance.get((x, y, -1)):
                knight_distance[(x, y, 1)] = knight_dist(x, y, p1_apple_loc[0], p1_apple_loc[1])
                knight_distance[(x, y, -1)] = knight_dist(x, y, p2_apple_loc[0], p2_apple_loc[1])

    # Piece-Square Policy Values: https://www.chessprogramming.org/Piece-Square_Tables
    p1_tile_values_scaled = np.array([[0, 1, 1.5, 1.2, 1.4, 0],
                                      [1, 1.3, 1.5, 1.6, 1.2, 1.2],
                                      [1.5, 1.5, 1.5, 1.7, 1.6, 1.5],
                                      [1.2, 1.6, 1.7, 1.7, 1.6, 1.3],
                                      [1.5, 1.6, 1.7, 1.4, 1.9, 1.5],
                                      [1, 1.3, 1.6, 1.9, 1.2, 1.2],
                                      [1, 1.4, 1.3, 1.5, 1.2, 0]])

    p2_tile_values_scaled = np.array([[0., 1.2, 1.5, 1.3, 1.4, 1.],
                                      [1.2, 1.2, 1.9, 1.6, 1.3, 1.],
                                      [1.5, 1.9, 1.4, 1.7, 1.6, 1.5],
                                      [1.3, 1.6, 1.7, 1.7, 1.6, 1.2],
                                      [1.5, 1.6, 1.7, 1.5, 1.5, 1.5],
                                      [1.2, 1.2, 1.6, 1.5, 1.3, 1.],
                                      [0., 1.4, 1.2, 1.5, 1., 0.]])

    tile_value = {1: p1_tile_values_scaled, -1: p2_tile_values_scaled}
    #executor = concurrent.futures.ProcessPoolExecutor(6)


def exitPlayer():
    return


def maux(hhst, time=None):
    global startTime
    startTime = time
    return hhst.multi_threaded_best_action(2000)


def compute_tree_data(hhst, time=None):
    global startTime
    startTime = time
    return hhst.multi_threaded_consolidate()

from timeit import default_timer as timer


def merge_nodes(node_0, node):
    node_0.visit_count += node.visit_count
    node_0.score += node.score
    node_0._n_sims_with_move += node._n_sims_with_move
    node_0._move_win_count += node._move_win_count


def consolidate(futures):
    def decaying_weight(child_node):
        return child_node.num_simulations_containing_move / (
                child_node.num_visits + child_node.num_simulations_containing_move + 4 * rave_constant ** 2 * child_node.num_visits * child_node.num_simulations_containing_move)

    children_l = []
    visits_l = []
    for args in futures:
        children_l.append(args[0])
        visits_l.append(args[1])

    num_child = len(children_l[0])
    children_same = [[children[i] for children in children_l] for i in range(0, num_child)]

    merged_children = []
    visits_l = sum(visits_l)

    for children in children_same:
        child = children.pop()
        while children:
            merge_nodes(child, children.pop())
        merged_children.append(child)

    choices_weights = []
    for c in merged_children:
        decaying_w = decaying_weight(c)
        if decaying_w:
            weight = (1 - decaying_w) * (c.node_score / c.num_visits) + decaying_w * (
                    c.move_win_count / c.num_simulations_containing_move) + exploration_constant * np.sqrt(
                (2 * np.log(visits_l) / c.num_visits))
        else:
            weight = c.node_score / c.num_visits + exploration_constant * np.sqrt(
                (2 * np.log(visits_l) / c.num_visits))
        choices_weights.append(weight)

    print(max(choices_weights))
    return merged_children[np.argmax(choices_weights)]




def multi_thread_best_action(state, num_simulations=2000, queue=None):
    if(len(np.where(state.board == 1)[0])) < 5:
        roots = [HorseHoldRaveSearchNode(state=state, parent=None),
                 HorseHoldRaveSearchNode(state=state, parent=None),
                 HorseHoldRaveSearchNode(state=state, parent=None),
                 HorseHoldRaveSearchNode(state=state, parent=None),
                 HorseHoldRaveSearchNode(state=state, parent=None),
                 HorseHoldRaveSearchNode(state=state, parent=None)]
        hhst = [HorseHoldSearchTree(roots[0]),
                HorseHoldSearchTree(roots[1]),
                HorseHoldSearchTree(roots[2]),
                HorseHoldSearchTree(roots[3]),
                HorseHoldSearchTree(roots[4]),
                HorseHoldSearchTree(roots[5])]
    else:
        roots = [HorseHoldRaveSearchNode(state=state, parent=None),
                 HorseHoldRaveSearchNode(state=state, parent=None),
                 HorseHoldRaveSearchNode(state=state, parent=None),
                 HorseHoldRaveSearchNode(state=state, parent=None)]
        hhst = [HorseHoldSearchTree(roots[0]),
                HorseHoldSearchTree(roots[1]),
                HorseHoldSearchTree(roots[2]),
                HorseHoldSearchTree(roots[3])]

    children = []
    for move in reversed(get_simulation_moves(state)):
        next_state = makeMove(state, move)
        children.append(HorseHoldRaveSearchNode(next_state, parent=None, move=move))

    exec = concurrent.futures.ProcessPoolExecutor(12)
    futures = [exec.submit(compute_tree_data, x, startTime) for x in hhst]
    results = concurrent.futures.wait(futures)
    futures = [x.result() for x in results[0]]
    # with open("game_file.txt", "w+") as game_file:
    #         game_file.write(str(state.board))
    #         game_file.write("\n")

    return consolidate(futures)




def getMove(state):
    global startTime
    startTime = datetime.now()
    best_node = multi_thread_best_action(state, 2000)
    #print(best_node.state.board)
    # with open("game_file.txt", "w+") as game_file:
    #     game_file.write(str(state.board))
    #     game_file.write("\n")
    return best_node.state.curr_move
