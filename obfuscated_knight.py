# uncompyle6 version 3.7.4
# Python bytecode 3.8 (3413)
# Decompiled from: Python 3.8.2 (tags/v3.8.2:7b3ab59, Feb 25 2020, 22:45:29) [MSC v.1916 32 bit (Intel)]
# Embedded file name: Dark_Knight.py
# Compiled at: 2020-11-30 22:56:03
# Size of source mod 2**32: 13154 bytes
import sys
l1ll_secret_ = sys.version_info[0] == 2
l11lll_secret_ = 2048
l1ll11_secret_ = 7
#
# def l11l_secret_(l1l111_secret_):
#     l11ll_secret_ = ord(l1l111_secret_[(-1)])
#     l1l_secret_ = l1l111_secret_[:-1]
#     l111_secret_ = l11ll_secret_ % len(l1l_secret_)
#     l11l1l_secret_ = l1l_secret_[:l111_secret_] + l1l_secret_[l111_secret_:]
#     if l1ll_secret_:
#         l1ll1_secret_ = unicode().join([l1lll_secret_(ord(char) - l11lll_secret_ - (l1llll_secret_ + l11ll_secret_) % l1ll11_secret_) for l1llll_secret_, char in enumerate(l11l1l_secret_)])
#     else:
#         l1ll1_secret_ = str().join([chr(ord(char) - l11lll_secret_ - (l1llll_secret_ + l11ll_secret_) % l1ll11_secret_) for l1llll_secret_, char in enumerate(l11l1l_secret_)])
#     return eval(l1ll1_secret_)


import numpy as np
from datetime import datetime

class GameState(object):
    __slots__ = [
     'board', 'playerToMove', 'gameOver', 'movesRemaining', 'points']


boardWidth = 0
boardHeight = 0
timeLimit = 0.0
victoryPoints = 0
moveLimit = 0
startState = None
assignedPlayer = 0
startTime = 0
pointMultiplier = 10
pieceValue = 20
victoryScoreThresh = 1000
l1lll1l_secret_ = 3
l111l1_secret_ = 20
l11ll11_secret_ = 0
l111lll_secret_ = 0
knight_distances_to_apple = None
tile_usefulness_scores = None
direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]

def getMoveOptions(state):
    global boardWidth
    global boardHeight
    moves = []
    for xStart in range(boardWidth):
        for yStart in range(boardHeight):
            if state.board[(xStart, yStart)] == state.playerToMove:
                for dx, dy in direction:
                    xEnd, yEnd = xStart + dx, yStart + dy
                    if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight and state.board[(xEnd, yEnd)] not in (state.playerToMove, 2 * state.playerToMove):
                        moves.append((xStart, yStart, xEnd, yEnd))

    return moves


def makeMove(state, move):
    global victoryPoints
    xStart, yStart, xEnd, yEnd = move
    newState = GameState()
    newState.board = np.copy(state.board)
    newState.board[(xStart, yStart)] = 0
    newState.board[(xEnd, yEnd)] = state.playerToMove
    newState.playerToMove = -state.playerToMove
    newState.movesRemaining = state.movesRemaining - 1
    newState.gameOver = False
    newState.points = 0
    if state.board[(xEnd, yEnd)] == -2 * state.playerToMove or -state.playerToMove not in newState.board:
        newState.gameOver = True
        newState.points = state.playerToMove * (victoryPoints + newState.movesRemaining)
    else:
        if newState.movesRemaining == 0:
            newState.gameOver = True
    return newState


def getScore(state):
    global knight_distances_to_apple
    global tile_usefulness_scores
    if state.gameOver:
        return pointMultiplier * state.points
    score = 0
    zero_if_player_one_else_one_if_player_two = (1 - state.playerToMove) // 2
    total_num_of_knights = [0, 0]
    num_knights_with_enemies_in_range = [0, 0]
    num_knights_with_more_enemies_than_friends_as_neighbors = [0, 0]
    total_num_of_knights_with_friendly_neighbors = [0, 0]
    is_winning = [10000, 10000]
    for x in range(boardWidth):
        for y in range(boardHeight):
            if state.board[(x, y)] in (-1, 1):
                negative_or_positive_knight = state.board[(x, y)]
                # 0 if player one, 1 if player two
                is_player_two = (1 - negative_or_positive_knight) // 2
                total_num_of_knights[is_player_two] += 1
                friendly_neighbors, enemy_neighbors = (0, 0)
                score += negative_or_positive_knight * tile_usefulness_scores[(is_player_two, x, y)]
                for dx, dy in direction:
                    xEnd, yEnd = x + dx, y + dy
                    if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight:
                        if state.board[(xEnd, yEnd)] == negative_or_positive_knight:
                            friendly_neighbors += 1
                        elif state.board[(xEnd, yEnd)] == -negative_or_positive_knight:
                            enemy_neighbors += 1
                        if enemy_neighbors > 0:
                            num_knights_with_enemies_in_range[is_player_two] += 1
                        if enemy_neighbors > friendly_neighbors:
                            num_knights_with_more_enemies_than_friends_as_neighbors[is_player_two] += 1
                        total_num_of_knights_with_friendly_neighbors[is_player_two] += friendly_neighbors
                        if knight_distances_to_apple[(1 - is_player_two, x, y)] == 1:
                            if state.playerToMove == negative_or_positive_knight:
                                is_winning[zero_if_player_one_else_one_if_player_two] = 1
                        elif friendly_neighbors >= enemy_neighbors:
                            is_winning[1 - zero_if_player_one_else_one_if_player_two] = min(is_winning[(1 - zero_if_player_one_else_one_if_player_two)], 2 * enemy_neighbors + 2)
    score += pieceValue * (total_num_of_knights[0] - total_num_of_knights[1]) + 1 * (num_knights_with_more_enemies_than_friends_as_neighbors[1] - num_knights_with_more_enemies_than_friends_as_neighbors[0]) + 4.0 * (total_num_of_knights_with_friendly_neighbors[0] / total_num_of_knights[0] - total_num_of_knights_with_friendly_neighbors[1] / total_num_of_knights[1])
    if (num_knights_with_more_enemies_than_friends_as_neighbors[zero_if_player_one_else_one_if_player_two] < num_knights_with_more_enemies_than_friends_as_neighbors[(1 - zero_if_player_one_else_one_if_player_two)] or num_knights_with_more_enemies_than_friends_as_neighbors[(1 - zero_if_player_one_else_one_if_player_two)]) == 1:
        if num_knights_with_enemies_in_range[(1 - zero_if_player_one_else_one_if_player_two)] == 1:
            score += state.playerToMove * pieceValue
    if state.playerToMove == 1 and total_num_of_knights[1] == 1 and num_knights_with_more_enemies_than_friends_as_neighbors[1] == 1:
        is_winning[zero_if_player_one_else_one_if_player_two] = 1

    elif state.playerToMove == -1:
        if total_num_of_knights[0] == 1:
            if num_knights_with_more_enemies_than_friends_as_neighbors[0] == 1:
                if state.movesRemaining > 0:
                    is_winning[zero_if_player_one_else_one_if_player_two] = 1

    if is_winning[zero_if_player_one_else_one_if_player_two] < 1000 and is_winning[zero_if_player_one_else_one_if_player_two] <= is_winning[(1 - zero_if_player_one_else_one_if_player_two)]:
        score = state.playerToMove * pointMultiplier * (victoryPoints + state.movesRemaining - is_winning[zero_if_player_one_else_one_if_player_two])
    elif is_winning[(1 - zero_if_player_one_else_one_if_player_two)] < is_winning[zero_if_player_one_else_one_if_player_two]:
            score = -state.playerToMove * pointMultiplier * (victoryPoints + state.movesRemaining - is_winning[(1 - zero_if_player_one_else_one_if_player_two)])
    return score

def getScore(state):
    global knight_distances_to_apple
    global tile_usefulness_scores
    if state.gameOver:
        return pointMultiplier * state.points
    l1111l_secret_ = 0
    l111ll1_secret_ = (1 - state.playerToMove) // 2
    l11111_secret_ = [0, 0]
    l1l11ll_secret_ = [0, 0]
    l1llll1l_secret_ = [0, 0]
    l1l11111_secret_ = [0, 0]
    l1ll11ll_secret_ = [10000, 10000]
    for x in range(boardWidth):
        for y in range(boardHeight):
            if state.board[(x, y)] in (-1, 1):
                l111l1l_secret_ = state.board[(x, y)]
                l1111l1_secret_ = (1 - l111l1l_secret_) // 2
                l11111_secret_[l1111l1_secret_] += 1
                l1ll111l_secret_, l1l1l1ll_secret_ = (0, 0)
                l1111l_secret_ += l111l1l_secret_ * tile_usefulness_scores[(l1111l1_secret_, x, y)]
                for dx, dy in direction:
                    l1l1l_secret_, l111ll_secret_ = x + dx, y + dy
                    if l1l1l_secret_ >= 0 and l1l1l_secret_ < boardWidth and l111ll_secret_ >= 0 and l111ll_secret_ < boardHeight:
                        if state.board[(l1l1l_secret_, l111ll_secret_)] == l111l1l_secret_:
                            l1ll111l_secret_ += 1
                        if state.board[(l1l1l_secret_, l111ll_secret_)] == -l111l1l_secret_:
                            l1l1l1ll_secret_ += 1
                        if l1l1l1ll_secret_ > 0:
                            l1l11ll_secret_[l1111l1_secret_] += 1
                        if l1l1l1ll_secret_ > l1ll111l_secret_:
                            l1llll1l_secret_[l1111l1_secret_] += 1
                        l1l11111_secret_[l1111l1_secret_] += l1ll111l_secret_
                        if knight_distances_to_apple[(1 - l1111l1_secret_, x, y)] == 1:
                            if state.playerToMove == l111l1l_secret_:
                                l1ll11ll_secret_[l111ll1_secret_] = 1
                        elif l1ll111l_secret_ >= l1l1l1ll_secret_:
                            l1ll11ll_secret_[1 - l111ll1_secret_] = min(l1ll11ll_secret_[(1 - l111ll1_secret_)], 2 * l1l1l1ll_secret_ + 2)
    l1111l_secret_ += pieceValue * (l11111_secret_[0] - l11111_secret_[1]) + 1 * (l1llll1l_secret_[1] - l1llll1l_secret_[0]) + 4.0 * (l1l11111_secret_[0] / l11111_secret_[0] - l1l11111_secret_[1] / l11111_secret_[1])
    if (l1llll1l_secret_[l111ll1_secret_] < l1llll1l_secret_[(1 - l111ll1_secret_)] or l1llll1l_secret_[(1 - l111ll1_secret_)]) == 1:
        if l1l11ll_secret_[(1 - l111ll1_secret_)] == 1:
            l1111l_secret_ += state.playerToMove * pieceValue
    if state.playerToMove == 1 and l11111_secret_[1] == 1 and l1llll1l_secret_[1] == 1:
        l1ll11ll_secret_[l111ll1_secret_] = 1
    else:
        if state.playerToMove == -1:
            if l11111_secret_[0] == 1:
                if l1llll1l_secret_[0] == 1:
                    if state.movesRemaining > 0:
                        l1ll11ll_secret_[l111ll1_secret_] = 1

    # if l1ll11ll_secret_[l111ll1_secret_] < 1000 and l1ll11ll_secret_[l111ll1_secret_] <= l1ll11ll_secret_[(1 - l111ll1_secret_)]:
    #     l1111l_secret_ = state.playerToMove * 10 * (victoryPoints + state.movesRemaining - l1ll11ll_secret_[l111ll1_secret_])
    # if l1ll11ll_secret_[(1 - l111ll1_secret_)] < l1ll11ll_secret_[l111ll1_secret_]:
    #     l1111l_secret_ = -state.playerToMove * 10 * (victoryPoints + state.movesRemaining - l1ll11ll_secret_[(1 - l111ll1_secret_)])
    return l1111l_secret_
    

def time_out():
    global startTime
    global timeLimit
    l1ll1ll_secret_ = datetime.now() - startTime
    return l1ll1ll_secret_.seconds + l1ll1ll_secret_.microseconds * 1e-06 >= timeLimit


def alpha_beta_search(state, lookAheadDepth, alpha, beta):
    if lookAheadDepth == 0 or state.gameOver:
        return getScore(state)

    if time_out():
        return 0

    a, b = alpha, beta
    for move in getMoveOptions(state):
        projectedState = makeMove(state, move)
        score = alpha_beta_search(projectedState, lookAheadDepth - 1, a, b)
        if state.playerToMove == 1 and score > a:
            a = score
            if a >= beta:
                break
        elif state.playerToMove == -1 and score < b:
            b = score
            if b <= alpha:
                break

    if state.playerToMove == 1:
        return a
    return b


def alpha_beta_adv(state, lookAheadDepth, alpha, beta):
    if lookAheadDepth == 0 or state.gameOver:
        return getScore(state)
    if time_out():
        return 0
    a, b = alpha, beta
    childrenScores = []
    children_states = []
    moves = getMoveOptions(state)
    for move in moves:
        projected_state = makeMove(state, move)
        childrenScores.append(-state.playerToMove * getScore(projected_state))
        children_states.append(projected_state)
    l1ll1111_secret_ = np.argsort(childrenScores)
    if lookAheadDepth == 1:
        return -state.playerToMove * childrenScores[l1ll1111_secret_[0]]

    for l1l1l1l1_secret_ in l1ll1111_secret_:
        if lookAheadDepth > 3:
            score = alpha_beta_adv(children_states[l1l1l1l1_secret_], lookAheadDepth - 1, a, b)
        else:
            score = alpha_beta_search(children_states[l1l1l1l1_secret_], lookAheadDepth - 1, a, b)
        if state.playerToMove == 1 and score > a:
            a = score
            if a >= beta:
                break
        elif state.playerToMove == -1 and score < b:
            b = score
            if b <= alpha:
                break
    if state.playerToMove == 1:
        return a
    return b


def compute_knight_distances_from_apple(zero_for_p1Apple_one_for_p2Apple, xOfApple, yOfApple):
    num_tiles = boardWidth * boardHeight - 1
    knight_distances_to_apple[(zero_for_p1Apple_one_for_p2Apple, xOfApple, yOfApple)] = 0
    l1l1l1l_secret_ = -1
    while num_tiles > 0:
        l1l1l1l_secret_ += 1
        for x in range(boardWidth):
            for y in range(boardHeight):
                if knight_distances_to_apple[(zero_for_p1Apple_one_for_p2Apple, x, y)] == l1l1l1l_secret_:
                    for dx, dy in direction:
                        xEnd, yEnd = x + dx, y + dy
                        if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight and knight_distances_to_apple[(zero_for_p1Apple_one_for_p2Apple, xEnd, yEnd)] == -1:
                            knight_distances_to_apple[(zero_for_p1Apple_one_for_p2Apple, xEnd, yEnd)] = l1l1l1l_secret_ + 1
                            num_tiles -= 1


def initPlayer(_1111_secret_, _111l_secret_, _11_secret_, _1lll1_secret_, _11ll1_secret_):
    global moveLimit
    global knight_distances_to_apple
    global tile_usefulness_scores
    global boardWidth
    global victoryPoints
    global timeLimit
    global assignedPlayer
    global startState
    global boardHeight
    startState, timeLimit, victoryPoints, moveLimit, assignedPlayer = (
     _1111_secret_, _111l_secret_, _11_secret_, _1lll1_secret_, _11ll1_secret_)
    boardWidth, boardHeight = startState.board.shape
    knight_distances_to_apple = -np.ones((2, boardWidth, boardHeight), dtype=int)
    tile_usefulness_scores = np.zeros((2, boardWidth, boardHeight))
    l11lll1l_secret_ = [0, 2, 2, 0]
    l11llll1_secret_ = [0, 6, 3, 1]
    for x in range(boardWidth):
        for y in range(boardHeight):
            if startState.board[(x, y)] == 2:
                compute_knight_distances_from_apple(0, x, y)
            elif startState.board[(x, y)] == -2:
                compute_knight_distances_from_apple(1, x, y)

    for x in range(boardWidth):
        for y in range(boardHeight):
            l1l1111l_secret_ = min(x, boardWidth - 1 - x)
            l11ll1ll_secret_ = min(y, boardHeight - 1 - y)
            l11ll1l1_secret_ = min(2, l1l1111l_secret_) + min(2, l11ll1ll_secret_)
            for pl in range(2):
                tile_usefulness_scores[(pl, x, y)] = l11ll1l1_secret_
                if knight_distances_to_apple[(pl, x, y)] < 4:
                    tile_usefulness_scores[(pl, x, y)] += l11lll1l_secret_[knight_distances_to_apple[(pl, x, y)]]
                if knight_distances_to_apple[(1 - pl, x, y)] < 4:
                    tile_usefulness_scores[(pl, x, y)] += l11llll1_secret_[knight_distances_to_apple[(1 - pl, x, y)]]


def exitPlayer():
    pass

def l11ll1l_secret_(state, l1ll11l_secret_, alpha, beta):
    if l1ll11l_secret_ == 0 or state.gameOver:
        return getScore(state)
    if time_out():
        return 0
    a, b = alpha, beta
    for move in getMoveOptions(state):
        l1lllll1_secret_ = makeMove(state, move)
        l1111l_secret_ = l11ll1l_secret_(l1lllll1_secret_, l1ll11l_secret_ - 1, a, b)
        if state.playerToMove == 1 and l1111l_secret_ > a:
            a = l1111l_secret_
            if a >= beta:
                break
        elif state.playerToMove == -1 and l1111l_secret_ < b:
            b = l1111l_secret_
            if b <= alpha:
                break
    if state.playerToMove == 1:
        return a
    return b


def l1l1lll1_secret_(state, l1ll11l_secret_, alpha, beta):
    if l1ll11l_secret_ == 0 or state.gameOver:
        return getScore(state)
    if time_out():
        return 0
    a, b = alpha, beta
    l1l1l11l_secret_ = []
    l1l1l111_secret_ = []
    l1_secret_ = getMoveOptions(state)
    for move in l1_secret_:
        l1lllll1_secret_ = GameState()
        l1lllll1_secret_ = makeMove(state, move)
        l1l1l11l_secret_.append(-state.playerToMove * getScore(l1lllll1_secret_))
        l1l1l111_secret_.append(l1lllll1_secret_)
    else:
        l1ll1111_secret_ = np.argsort(l1l1l11l_secret_)
        if l1ll11l_secret_ == 1:
            return -state.playerToMove * l1l1l11l_secret_[l1ll1111_secret_[0]]

    for l1l1l1l1_secret_ in l1ll1111_secret_:
        if l1ll11l_secret_ > 3:
            l1111l_secret_ = l1l1lll1_secret_(l1l1l111_secret_[l1l1l1l1_secret_], l1ll11l_secret_ - 1, a, b)
        else:
            l1111l_secret_ = l11ll1l_secret_(l1l1l111_secret_[l1l1l1l1_secret_], l1ll11l_secret_ - 1, a, b)
        if state.playerToMove == 1 and l1111l_secret_ > a:
            a = l1111l_secret_
            if a >= beta:
                break
        elif state.playerToMove == -1 and l1111l_secret_ < b:
            b = l1111l_secret_
            if b <= alpha:
                break

    if state.playerToMove == 1:
        return a
    return b

def getMove(state):
    global startTime
    startTime = datetime.now()  # Remember computation start time
    moveList = getMoveOptions(state)  # Get the list of possible moves
    favoredMove = moveList[0]  # Just choose first move from the list for now, in case we run out of time
    favoredMoveScore = -9e9 * state.playerToMove  # Use this variable to remember the score for the favored move

    # if assignedPlayer == 1:
    #     playerNames = ["philip_bot", "opponent"]
    # else:
    #     playerNames = ['opponent', 'philip_bot']

    # Iterative deepening loop
    for lookAheadDepth in range(3, 20 + 1):
        currBestMove = None  # Best move and score currently found during the current iteration (lookAheadDepth)
        currBestScore = -9e9 * state.playerToMove

        # Try every possible next move, evaluate it using Minimax, and pick the one with best score
        for move in moveList:
            projectedState = makeMove(state, move)
            #score = lookAhead(projectedState, lookAheadDepth - 1)  # Find score through MiniMax for current lookAheadDepth
            score = l1l1lll1_secret_(projectedState, lookAheadDepth - 1, -9e9, 9e9)
            if time_out():
                break

            if (state.playerToMove == 1 and score > currBestScore) or (state.playerToMove == -1 and score < currBestScore):
                currBestMove, currBestScore = move, score  # Found new best move during this iteration
                # displayState(state, playerNames=playerNames, selectedSquare=(move[2], move[3]))
                # print("Move: %s, Score: %s" % (move, score))
                # input("he")

        if not time_out():  # Pick the move from the last lookahead depth as new favorite, unless the lookahead was incomplete
            favoredMove, favoredMoveScore = currBestMove, currBestScore
            duration = datetime.now() - startTime
            print('obfuscated_knight: Depth %d finished at %.4f s, favored move (%d,%d)->(%d,%d), score = %.2f'
                  % (lookAheadDepth, duration.seconds + duration.microseconds * 1e-6,
                     favoredMove[0], favoredMove[1], favoredMove[2], favoredMove[3], favoredMoveScore))
        else:
            print('obfuscated_knight: Timeout!')

        if time_out() or abs(
                favoredMoveScore) > victoryScoreThresh:  # Stop computation if timeout or certain victory/defeat predicted
            break

    return favoredMove
