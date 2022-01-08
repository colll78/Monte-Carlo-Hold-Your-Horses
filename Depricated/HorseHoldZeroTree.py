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


def get_simulation_moves(state):
    moves = getMoveOptions(state)
    filtered_moves = []
    atk_moves = []
    for x in moves:
        if state.board[x[2], x[3]] == state.playerToMove * -2:
            return [x]
        if (x[2], x[3]) in mating_squares[-state.playerToMove]:
            if all(state.board[square] != -1 * state.playerToMove for square in mating_squares[state.playerToMove]):
                if attackers_defenders(state, x[2], x[3]) >= 0:
                    return [x]
        if attackers_defenders(state, x[2], x[3]) >= 0 and state.board[x[2], x[3]] == -state.playerToMove:
            return [x]
        if state.board[x[2], x[3]] == -state.playerToMove and (x[2], x[3]) in mating_squares[state.playerToMove]:
            return [x]
        if state.board[x[2], x[3]] == -state.playerToMove:
            atk_moves.append(x)
        if attackers_defenders(state, x[2], x[3]) >= 0:
            filtered_moves.append(x)
    # if not filtered_moves:
    #     print("NO GOOD MOVES")
    #     print(moves)
    #     print(state.board)
    return filtered_moves or moves

# def get_simulation_moves(state):
#     moves = getMoveOptions(state)
#     filtered_moves = []
#     for x in moves:
#         if state.board[x[2], x[3]] == state.playerToMove * -2:
#             return [x]
#         if (x[2], x[3]) in mating_squares[-state.playerToMove]:
#             if all(state.board[square] != -1 * state.playerToMove for square in mating_squares[state.playerToMove]):
#                 if attackers_defenders(state, x[2], x[3]) >= 0:
#                     return [x]
#         if attackers_defenders(state, x[2], x[3]) >= 0 or state.board[x[2], x[3]] == -state.playerToMove:
#             filtered_moves.append(x)
#     return filtered_moves or moves


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


# based on the search tree employed by AlphaZero and LeelaZero
class HorseHoldSearchNode:
    def __init__(self, state, parent=None):
        self._outcomes = defaultdict(int)
        self.upper_bound = None
        self.lower_bound = None
        self.state = state
        self.parent = parent
        self.children = []
        self.to_process = []
        self.score = 0
        self.visit_count = 0

    def victory_score(self):
        # TODO: potentially adjust to only consider wins, and not subtract loses, since if the move results in many
        #       loses then it will be considered anyway since W[i] (wins for the node considered after the i-th move)
        win_count = self._outcomes[self.parent.state.playerToMove]
        lose_count = self._outcomes[-1 * self.parent.state.playerToMove]
        #print(self.state.board, " ", win_count)
        #return win_count
        return win_count-lose_count

    @property
    def moves_to_try(self):
        if not hasattr(self, '_moves_to_try'):
            #self._moves_to_try = getMoveOptions(self.state)
            self._moves_to_try = get_simulation_moves(self.state)
            if not self._moves_to_try:
                self._moves_to_try = getMoveOptions(self.state)
        return self._moves_to_try

    def num_visits(self):
        return self.visit_count

    def expand(self):
        # print("expanding")
        # print("move options ", getMoveOptions(self.state))
        # print(self.state.board)
        # print(self.state.playerToMove)
        move = self.moves_to_try.pop()
        new_state = makeMove(self.state, move)
        child_node = HorseHoldSearchNode(new_state, parent=self)
        self.children.append(child_node)
        return child_node

    def at_final_node(self):
        return self.state.gameOver

    def simulation(self):
        current_simulation = self.state
        while not current_simulation.gameOver:
            #possible_moves = getMoveOptions(current_simulation)
            possible_moves = get_simulation_moves(current_simulation)
            # Complete one random playout by choosing uniform random moves until the game is decided
            move = possible_moves[np.random.randint(len(possible_moves))]
            current_simulation = makeMove(current_simulation, move)
        return current_simulation.winner

    def backpropagation(self, outcome):
        #print("Outcome: ", outcome)
        self.visit_count += 1
        self._outcomes[outcome] += 1
        if self.parent:
            self.parent.backpropagation(outcome)

    def expansion_complete(self):
        return len(self.moves_to_try) == 0

    # c_param recommended = 1.4
    def best_child(self, exploration_param=0.05):
        # TODO: potentially add weights to choices to incorporate domain knowledge
        weights = []
        best = self.children[0]
        for child in self.children:
            node_score = 0
            if child.num_visits() != 0:
                node_score = (child.victory_score() / float(child.num_visits())) + exploration_param * math.sqrt((2 * float(math.log(self.num_visits())) / float(child.num_visits())))
                #node_score = child.victory_score() / float(child.num_visits())
            child.score = node_score
            if best.score < child.score:
                best = child
        return best


def time_out():
    duration = datetime.now() - startTime
    return duration.seconds + duration.microseconds * 1e-6 >= timeLimit


class HorseHoldSearchTree:
    def __init__(self, node: HorseHoldSearchNode):
        self.root = node

    def best_action(self, num_simulations=2000, queue=None):
        for _ in range(0, num_simulations):
            c = self.expansion()
            points = c.simulation()
            c.backpropagation(points)
            if _ % 200 == 0:
                best = self.root.best_child(exploration_param=0)
            if time_out():
                print("HorseHoldZeroTree TimeOut at simulation :", _)
                return best
        # for child in self.root.children:
        #     print(child.state.curr_move, " = ", child.score)
        #print()
        #start_t = datetime.now()
        best = self.root.best_child(exploration_param=0)
        #end_t = datetime.now()
        #print("Time taken: ", end_t-start_t)
        print("HorseHoldZeroTree BEST MOVE: ", best.state.curr_move, " Score: ", best.score)

        if queue:
            queue.put(best)
        else:
            return best

    def expansion(self):
        curr_node = self.root
        while not curr_node.at_final_node():
            if not curr_node.expansion_complete():
                return curr_node.expand()
            else:
                #
                curr_node = curr_node.best_child()
        return curr_node


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


def getMove(state):
    global startTime
    startTime = datetime.now()
    queue = qu.Queue()
    root = HorseHoldSearchNode(state=state, parent=None)
    # if(len(get_simulation_moves(state)) == 0):
    #     print("NO GOOD MOVES IN REAL")
    #     input()
    #root.expand()
    hhst = HorseHoldSearchTree(root)
    best_node = hhst.best_action(2000)
    #print(best_node.state.board)
    return best_node.state.curr_move