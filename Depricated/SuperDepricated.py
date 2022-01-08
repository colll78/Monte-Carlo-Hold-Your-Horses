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
leaf node parallelization
use of transposition table
"""

from collections import defaultdict
import random
from datetime import datetime
import numpy as np
import queue as qu
import math

__author__ = "Philip DiSarro"
__credits__ = ["Sylvain Gelly", "David Silver", "David P. Helmbold", "Aleatha Parker-Wood"]
__version__ = "1.0"
__maintainer__ = "Philip DiSarro"
__email__ = "philip.disarro@phabulous.org"
__status__ = "Production"


boardWidth = 0            # These 3 global variables are set by the getMove function. This is not...
boardHeight = 0           # ... an elegant solution but an efficient one.
timeLimit = 0.0           # Maximum thinking time (in seconds) for each move

startState = None         # Initial state, provided to the initPlayer function
assignedPlayer = 0        # 1 -> player MAX; -1 -> player MIN (in terms of the MiniMax algorithm)
startTime = 0

victoryPoints = 0       # Number of points for the winner
moveLimit = 0           # Maximum number of moves
                          # If exceeded, game is a tie; otherwise, number of remaining moves is added to winner's score.

pointMultiplier = 10      # Muliplier for winner's points in getScore function
pieceValue = 20           # Score value of a single piece in getScore function
victoryScoreThresh = 1000 # An absolute score exceeds this value if and only if one player has won
minLookAhead = 1          # Initial search depth for iterative deepening
maxLookAhead = 20          # Maximum search depth
apple_loc = None
mating_squares = None
reachable = None
knight_distance = {}
tile_value = {}


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
    weights = []
    for xStart in range(boardWidth):  # Search board for player's pieces
        for yStart in range(boardHeight):
            if state.board[xStart, yStart] == state.playerToMove:  # Found a piece!
                for (dx, dy) in direction:  # Check all potential move vectors
                    (xEnd, yEnd) = (xStart + dx, yStart + dy)
                    if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight and not (state.board[xEnd, yEnd] in [state.playerToMove, 2 * state.playerToMove]):
                        if state.board[xEnd, yEnd] == state.playerToMove * -2:
                            return [(xStart, yStart, xEnd, yEnd)], False
                        elif (xEnd, yEnd) in mating_squares[-state.playerToMove] and all(state.board[square] != -1 * state.playerToMove for square in mating_squares[state.playerToMove]) and attackers_defenders(state, xEnd, yEnd) >= 0:
                            return [(xStart, yStart, xEnd, yEnd)], False
                        elif (xEnd, yEnd) in mating_squares[state.playerToMove] and state.board[xEnd, yEnd] == -state.playerToMove:
                            return [(xStart, yStart, xEnd, yEnd)], False
                        if state.board[xEnd, yEnd] == -state.playerToMove and (xEnd, yEnd) in mating_squares[state.playerToMove]:
                            return [(xStart, yStart, xEnd, yEnd)], False
                        atk_def = attackers_defenders(state, xEnd, yEnd)
                        if atk_def >= 0 and state.board[xEnd, yEnd] == -state.playerToMove:
                            weights.append(tile_value[state.playerToMove][xEnd, yEnd]*20)
                        elif atk_def >= 0 or state.board[xEnd, yEnd] == -state.playerToMove:
                            weights.append(tile_value[state.playerToMove][xEnd, yEnd]*5)
                        else:
                            weights.append(tile_value[state.playerToMove][xEnd, yEnd])
                        moves.append((xStart, yStart, xEnd, yEnd))
    return moves, weights


def get_simulation_moves(state):
    moves = getMoveOptions(state)
    filtered_moves = []
    for x in moves:
        if state.board[x[2], x[3]] == state.playerToMove * -2:
            return [x]
        if (x[2], x[3]) in mating_squares[-state.playerToMove]:
            if all(state.board[square] != -1 * state.playerToMove for square in mating_squares[state.playerToMove]):
                if attackers_defenders(state, x[2], x[3]) >= 0:
                    return [x]
        if state.board[x[2], x[3]] == -state.playerToMove and (x[2], x[3]) in mating_squares[state.playerToMove]:
            return [x]
        if attackers_defenders(state, x[2], x[3]) >= 0 or state.board[x[2], x[3]] == -state.playerToMove:
            filtered_moves.append(x)
    # if not filtered_moves:
    #     print("NO GOOD MOVES")
    #     print(moves)
    #     print(state.board)
    return filtered_moves or moves


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
exploration_constant = .05
#rave_constant = 1
rave_constant = 0.2
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
        self._move_policy = None
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
            possible_moves = get_weighted_moves(current_rollout_state)
            if possible_moves[1]:
                action = self.simulation_move_policy(possible_moves)
            else:
                action = possible_moves[0][0]
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
        """
        This policy is based on the premise that moves to higher value squares (as defined by the piece-square tables)
        should have a higher likelihood to be chosen in the simulations.
        :return:
        """
        return random.choices(possible_moves[0], weights=possible_moves[1], k=1)[0]


def time_out():
    duration = datetime.now() - startTime
    return duration.seconds + duration.microseconds * 1e-6 >= timeLimit


class HorseHoldSearchTree:
    def __init__(self, node: HorseHoldRaveSearchNode):
        self.root = node

    def best_action(self, num_simulations=2000, queue=None):
        # TODO: store results in queue so that the search can be parallelized
        for _ in range(0, num_simulations):
            c = self.expansion()
            points = c.run_simulation()
            c.backpropagate(points)
            if _ % 200 == 0:
                best = self.root.best_child(exploration_c=0)
            if _ % 20 == 0 and time_out():
                #print("HoldHorsesRave TimeOut at simulation :", _)
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
    global startState, timeLimit, victoryPoints, moveLimit, assignedPlayer, boardWidth, boardHeight, apple_loc, mating_squares, knight_distance, reachable, tile_value
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

    p2_mating_squares = []
    for (dx, dy) in direction:
        (xEnd, yEnd) = (p2_apple_loc[0] + dx, p2_apple_loc[1] + dy)
        if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight:
            p2_mating_squares.append((xEnd, yEnd))

    mating_squares = {1: p1_mating_squares, -1: p2_mating_squares}

    for x in range(0, boardWidth):
        for y in range(0, boardHeight):
            if not reachable.get((x,y)):
                reachable[(x,y)] = around(x,y)
            if not knight_distance.get((x, y, 1)) and not knight_distance.get((x, y, -1)):
                knight_distance[(x, y, 1)] = knight_dist(x, y, p1_apple_loc[0], p1_apple_loc[1])
                knight_distance[(x, y, -1)] = knight_dist(x, y, p2_apple_loc[0], p2_apple_loc[1])

    p1_piece_square_table_policy = np.array([[0, 1.61803399,  8.09016994,  3.23606798,  6.47213595,  0.],
                                          [1.61803399,  4.85410197,  8.09016994,  9.70820393,  3.23606798,  3.23606798],
                                          [8.09016994,  8.09016994,  8.09016994, 11.32623792,  9.70820393,  8.09016994],
                                          [3.23606798,  9.70820393, 11.32623792, 11.32623792,  9.70820393,  4.85410197],
                                          [8.09016994,  9.70820393, 11.32623792,  6.47213595, 14.5623059,  8.09016994],
                                          [1.61803399,  4.85410197, 9.70820393,   14.5623059,   3.23606798,  3.23606798],
                                          [1.61803399,  6.47213595,  4.85410197,  8.09016994,  3.23606798,  0.]])
    p2_piece_square_table_policy = np.array([[ 0.,          3.23606798,  8.09016994,  4.85410197,  6.47213595,  1.61803399],
                                             [ 3.23606798,  3.23606798, 14.5623059,   9.70820393,  4.85410197,  1.61803399],
                                             [ 8.09016994, 14.5623059,   6.47213595, 11.32623792,  9.70820393,  8.09016994],
                                             [ 4.85410197,  9.70820393, 11.32623792, 11.32623792,  9.70820393,  3.23606798],
                                             [ 8.09016994,  9.70820393, 11.32623792,  8.09016994,  8.09016994,  8.09016994],
                                             [ 3.23606798,  3.23606798,  9.70820393,  8.09016994,  4.85410197,  1.61803399],
                                             [ 0.,          6.47213595,  3.23606798,  8.09016994,  1.61803399,  0.        ]])
    tile_value = {1: p1_piece_square_table_policy, -1: p2_piece_square_table_policy}



def exitPlayer():
    return


def getMove(state):
    global startTime
    startTime = datetime.now()
    queue = qu.Queue()
    root = HorseHoldRaveSearchNode(state=state, parent=None)
    #root.expand()
    hhst = HorseHoldSearchTree(root)
    best_node = hhst.best_action(2000)
    #print(best_node.state.board)
    return best_node.state.curr_move