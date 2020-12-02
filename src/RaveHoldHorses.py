from collections import defaultdict
from datetime import datetime
import numpy as np
import queue as qu
import math


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


def filter_possible_moves(state):
    moves = getMoveOptions(state)
    filtered_moves = []
    for x in moves:
        if state.board[x[2], x[3]] == state.playerToMove * -2:
            return [x]
        if (x[2], x[3]) in mating_squares[-state.playerToMove]:
            if all(state.board[square] != -1 * state.playerToMove for square in mating_squares[state.playerToMove]):
                if attackers_defenders(state, x[2], x[3]) >= 0:
                    return [x]
        if attackers_defenders(state, x[2], x[3]) >= 0 or state.board[x[2], x[3]] == -state.playerToMove:
            filtered_moves.append(x)
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


# Set global variables and initialize any data structures that the player will need
def initPlayer(_startState, _timeLimit, _victoryPoints, _moveLimit, _assignedPlayer):
    global startState, timeLimit, victoryPoints, moveLimit, assignedPlayer, boardWidth, boardHeight, apple_loc, mating_squares
    startState, timeLimit, victoryPoints, moveLimit, assignedPlayer = _startState, _timeLimit, _victoryPoints, _moveLimit, _assignedPlayer
    (boardWidth, boardHeight) = startState.board.shape
    p1_apple_loc = p1_apple(startState)
    p2_apple_loc = p2_apple(startState)
    apple_loc = {1: p1_apple_loc, -1: p2_apple_loc}

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


def exitPlayer():
    return


# RAVE_C_FACTOR = 1.4
RAVE_C_FACTOR = .05
#RAVE_B_FACTOR = 1
RAVE_B_FACTOR = 0.2
REWARD = {0: 0.002, 1: 1, -1: 0}


class MonteCarloTreeSearchNode():
    action: object
    parent: object
    children: list
    _number_of_visits: int
    _wins: float
    _untried_actions: list

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.action = action
        self._number_of_visits = 0
        self._untried_actions = None
        self._wins = 0.0

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = filter_possible_moves(self.state)
        return self._untried_actions

    @property
    def n(self):
        return self._number_of_visits

    @property
    def w(self):
        return self._wins

    def expand(self):
        action = self.untried_actions.pop()
        next_state = makeMove(self.state, action)
        # be prepared to instantiate of inherited class
        child_node = self.__class__(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def expand_action(self, action):
        self.untried_actions.remove(action)
        next_state = self.state.move(action)
        # be prepared to instantiate of inherited class
        child_node = self.__class__(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def get_child(self, action):
        if action in self.untried_actions:
            self.expand_action(action)
        for c in self.children:
            if action == c.action:
                return c
        return None

    def is_terminal_node(self):
        return self.state.gameOver

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.gameOver:
            possible_moves = filter_possible_moves(current_rollout_state)
            action = self.rollout_policy(possible_moves)
            current_rollout_state = makeMove(current_rollout_state, action)
        return current_rollout_state.winner

    def update_stats(self, result):
        self._number_of_visits += 1.0
        result_for_self = -1  # assume self lost and gets zero reward
        if self.parent:
            result_for_self = result * self.parent.state.playerToMove
        self._wins += REWARD[result_for_self]

    def backpropagate(self, result):
        self.update_stats(result)
        if self.parent:
            self.parent.backpropagate(result)

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.w / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]


class MonteCarloRaveNode(MonteCarloTreeSearchNode):
    _number_of_visits_rave: int
    _wins_rave: float

    def __init__(self, state, parent=None, action=None):
        super().__init__(state, parent, action)
        self._number_of_visits_rave = 0
        self._wins_rave = 0.0

    @property
    def w_rave(self):
        return self._wins_rave

    @property
    def n_rave(self):
        return self._number_of_visits_rave

    def best_child(self, c_param=RAVE_C_FACTOR):
        def calc_beta(c):
            return c.n_rave / (c.n + c.n_rave + 4 * RAVE_B_FACTOR ** 2 * c.n * c.n_rave)

        choices_weights = []
        for c in self.children:
            beta = calc_beta(c)
            if beta != 0:
                weight = (1 - beta) * (c.w / c.n) + beta * (c.w_rave / c.n_rave) + \
                         c_param * np.sqrt((2 * np.log(self.n) / c.n))
            else:
                weight = c.w / c.n + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            choices_weights.append(weight)
        return self.children[np.argmax(choices_weights)]

    def rollout(self):
        actions = set()
        current_rollout_state = self.state
        while not current_rollout_state.gameOver:
            possible_moves = filter_possible_moves(current_rollout_state)
            action = self.rollout_policy(possible_moves)
            actions.add(action)
            current_rollout_state = makeMove(current_rollout_state, action)
        return current_rollout_state.winner, actions

    def backpropagate(self, rollout_output):
        result, rollout_actions = rollout_output
        self.update_stats(result)
        if self.parent:
            for c in self.parent.children:
                if c.action in rollout_actions:
                    result_for_c = result * self.state.playerToMove
                    c._wins_rave += REWARD[result_for_c]
                    c._number_of_visits_rave += 1
            self.parent.backpropagate(rollout_output)

def time_out():
    duration = datetime.now() - startTime
    return duration.seconds + duration.microseconds * 1e-6 >= timeLimit

class MonteCarloTreeSearch(object):

    def __init__(self, node):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """
        self.root = node

    def best_action(self, simulations_number):
        """
        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action
        Returns
        -------
        """
        for _ in range(0, simulations_number):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
            if _ % 200 == 0:
                best = self.root.best_child(c_param=0)
            if _ % 20 == 0 and time_out():
                print("RaveHoldHorse TimeOut at simulation: ", _)
                return best

        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)

    def _tree_policy(self):
        """
        selects node to run rollout/playout for
        Returns
        -------
        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node


def getMove(state):
    global startTime
    startTime = datetime.now()
    queue = qu.Queue()
    root = MonteCarloRaveNode(state=state, parent=None, action=None)
    #root = MonteCarloTreeSearchNode(state=state, parent=None, action=None)
    #root.expand()
    hhst = MonteCarloTreeSearch(root)
    best_node = hhst.best_action(2000)
    #print(best_node.state.board)
    return best_node.state.curr_move