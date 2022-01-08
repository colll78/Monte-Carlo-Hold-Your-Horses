import numpy as np
import random
from timeit import default_timer as timer

class GameState(object):
    __slots__ = ['board', 'playerToMove', 'gameOver', 'movesRemaining', 'points', 'winner', 'curr_move']

# TODO: modify filter_possible_moves to return
#  moves that sacrifice pieces if it results in a winning sequence

def simulation_move_policy(possible_moves):
    move = random.choices(possible_moves, cum_weights=(5, 15, 30, 50), k=1)
    # or alternatively
    move = np.random.choice(possible_moves, 1, p=[0.1, 0.6, 0.3])


#This neural network takes the board position as input and outputs position evaluation (QValue) and a vector of move probabilities (PValue, policy).
def predict(state):
    # a list of probabilities for each move (Pvalue, policy)
    move_probabilities = []

    # The QValue is the position evaluation
    q_value = 0

    return q_value, move_probabilities

boardWidth = 7
boardHeight = 6

class GameState(object):
    __slots__ = ['board', 'playerToMove', 'gameOver', 'movesRemaining', 'points']


def getMoveOptions(state):
    direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]  # Possible (dx, dy) moves
    moves = []
    for xStart in range(boardWidth):  # Search board for player's pieces
        for yStart in range(boardHeight):
                for (dx, dy) in direction:  # Check all potential move vectors
                    (xEnd, yEnd) = (xStart + dx, yStart + dy)
                    if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight:
                        moves.append((xStart, yStart, xEnd,
                                      yEnd))  # If square is empty or occupied by the opponent, then we have a legal move.
    return moves

horseCoords = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
appleCoords = (0, 0)

timeLimit = 5.0  # Limit for computer players' thinking time (seconds)
timeTolerance = 0.1  # Additional wait time (seconds) before timeout is called

victoryPoints = 100  # Number of points for the winner
moveLimit = 40  # Maximum number of moves

state = GameState()
state.board = np.zeros((boardWidth, boardHeight), dtype=int)
state.board[appleCoords] = 2
state.board[boardWidth - appleCoords[0] - 1, boardHeight - appleCoords[1] - 1] = -2

for (x, y) in horseCoords:
    state.board[x, y] = 1
    state.board[boardWidth - x - 1, boardHeight - y - 1] = -1

state.playerToMove = 1
state.movesRemaining = moveLimit
state.gameOver = False
state.points = 0

board = np.array([[ 2,  1,  1,  0,  0,  0],
                  [ 1,  1,  1,  0,  0,  0,],
                  [ 1,  1,  0,  0,  0,  0],
                  [ 0,  0,  0,  0,  0,  0],
                  [ 0,  0,  0,  0, -1, -1],
                  [ 0,  0,  0, -1, -1, -1],
                  [ 0,  0,  0, -1, -1, -2]])
reverse_board = np.flip(np.flip(board, 1), 0)
# print([str(x) for x in getMoveOptions(state)])
# print(len(getMoveOptions(state)))
import hashlib
print(hashlib.md5(np.array_str(np.array(board)).encode("utf-8")).hexdigest())
print(hashlib.md5(np.array_str(np.array(reverse_board)).encode("utf-8")).hexdigest())

board = np.array([[ 2,  1,  1,  0,  0,  0],
                  [ 1,  1,  1,  0,  0,  0,],
                  [ 1,  0,  0,  0,  0,  0],
                  [ 0,  0,  0,  0,  0,  0],
                  [ 0,  0,  1,  0, -1, -1],
                  [ 0,  0,  0, -1, -1, -1],
                  [ 0,  0,  0, -1, -1, -2]])

start = timer()

end = timer()
print("copy time ", start-end)
#print(np.flip(np.flip(board, 1), 0))


### GRAVE YARD ###
# newState.horse_coords = state.horse_coords.copy()
# print(newState.horse_coords[state.playerToMove])
# print(state.playerToMove)
# print(move)
# print(state.board)
# horse_index = newState.horse_coords[state.playerToMove].index((xStart, yStart))
# newState.horse_coords[state.playerToMove][horse_index] = (xEnd, yEnd)

# def getScore(state):
#     global get_score_time
#     start = timer()
#     score = pointMultiplier * state.points
#     if state.gameOver:
#         return score
#     p1_horses = np.where(state.board == 1)
#     p2_horses = np.where(state.board == -1)
#     direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]
#     for square in mating_squares[1]:
#         atk_def = 0
#         for (dx, dy) in direction:
#             (xEnd, yEnd) = (square[0] + dx, square[1] + dy)
#             if 0 <= xEnd < boardWidth and 0 <= yEnd < boardHeight:
#                 if state.board[xEnd, yEnd] == 1:
#                     atk_def += 1
#                 elif state.board[xEnd, yEnd] == -1:
#                     atk_def -= 1
#         if atk_def <= 0:
#             if state.board[square[0], square[1]] == -1:
#                 if not any(state.board[square] == 1 for square in mating_squares[-1]):
#                     score -= 1000
#                     # print("CURRENT")
#                     # print(state.board)
#             else:
#                 score -= pieceValue/4
#
# for square in mating_squares[-1]:
#     atk_def = 0
#     for (dx, dy) in direction:
#         (xEnd, yEnd) = (square[0] + dx, square[1] + dy)
#         if 0 <= xEnd < boardWidth and 0 <= yEnd < boardHeight:
#             if state.board[xEnd, yEnd] == 1:
#                 atk_def += 1
#             elif state.board[xEnd, yEnd] == -1:
#                 atk_def -= 1
#     if atk_def > 0:
#         if state.board[square[0], square[1]] == 1:
#             if not any(state.board[square] == -1 for square in mating_squares[1]):
#                 score += 1000
#         else:
#             score += pieceValue/4

# for i in range(0, len(p1_horses[0])):
#     appleDistance = knight_distance.get((p1_horses[0][i], p1_horses[1][i], -1))
#     score += pieceValue - appleDistance
#
# for i in range(0, len(p2_horses[0])):
#     appleDistance = knight_distance.get((p2_horses[0][i], p2_horses[1][i], 1))
#     score -= pieceValue - appleDistance
# end = timer()
# # print("getScore time: %.20f" % (end-start))
# # if score > 1000 or score < -1000:
# #     print("Previous Score: ", prev_score)
# #     print(score)
# #     print(state.board)
# #     print(state.playerToMove)
# return score






#mcts_rave
# end_t = datetime.now()
# print("Time taken: ", end_t-start_t)
# print("BEST MOVE: ", best.state.curr_move, " Score: ", best.score)
# if best.state.playerToMove == 1:
#     print("Current Player 1")
# else:
#     print("Current Player 2")
# for child in self.root.children:
#     print(child.state.curr_move, " = ", child.score)
# print()
# start_t = datetime.now()