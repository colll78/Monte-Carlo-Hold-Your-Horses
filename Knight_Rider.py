# Player Knight_Rider

import numpy as np
from datetime import datetime


class GameState(object):
    __slots__ = ['board', 'playerToMove', 'gameOver', 'movesRemaining', 'points']


# Global variables
boardWidth = 0  # Board dimensions
boardHeight = 0
timeLimit = 0.0  # Maximum thinking time (in seconds) for each move
victoryPoints = 0  # Number of points for the winner
moveLimit = 0  # Maximum number of moves
# If exceeded, game is a tie with victoryPoints being split between players.
# Otherwise, number of remaining moves is added to winner's score.
startState = None  # Initial state, provided to the initPlayer function
assignedPlayer = 0  # 1 -> player MAX; -1 -> player MIN (in terms of the MiniMax algorithm)
startTime = 0  # Remember the time stamp when move computation started

# Local parameters for player's algorithm. Can be modified, deleted, or extended in any conceivable way
pointMultiplier = 10  # Muliplier for winner's points in getScore function
pieceValue = 20  # Score value of a single piece in getScore function
victoryScoreThresh = 1000  # An absolute score exceeds this value if and only if one player has won
minLookAhead = 1  # Initial search depth for iterative deepening
maxLookAhead = 4  # Maximum search depth


# Compute list of legal moves for a given GameState and the player moving next
def getMoveOptions(state):
    direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]  # Possible (dx, dy) moves
    moves = []
    for xStart in range(boardWidth):  # Search board for player's pieces
        for yStart in range(boardHeight):
            if state.board[xStart, yStart] == state.playerToMove:  # Found a piece!
                for (dx, dy) in direction:  # Check all potential move vectors
                    (xEnd, yEnd) = (xStart + dx, yStart + dy)
                    if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight and not (
                            state.board[xEnd, yEnd] in [state.playerToMove, 2 * state.playerToMove]):
                        moves.append((xStart, yStart, xEnd,
                                      yEnd))  # If square is empty or occupied by the opponent, then we have a legal move.
    return moves


# For a given GameState and move to be executed, return the GameState that results from the move
def makeMove(state, move):
    (xStart, yStart, xEnd, yEnd) = move
    newState = GameState()
    newState.board = np.copy(state.board)  # The new board configuration is a copy of the current one except that...
    newState.board[xStart, yStart] = 0  # ... we remove the moving piece from its start position...
    newState.board[xEnd, yEnd] = state.playerToMove  # ... and place it at the end position
    newState.playerToMove = -state.playerToMove  # After this move, it will be the opponent's turn
    newState.movesRemaining = state.movesRemaining - 1
    newState.gameOver = False
    newState.points = 0

    if state.board[xEnd, yEnd] == -2 * state.playerToMove or not (-state.playerToMove in newState.board):
        newState.gameOver = True  # If the opponent lost the apple or all horses, the game is over...
        newState.points = state.playerToMove * (
                    victoryPoints + newState.movesRemaining)  # ... and more remaining moves result in more points
    elif newState.movesRemaining == 0:  # Otherwise, if there are no more moves left, the game is drawn
        newState.gameOver = True

    return newState


# Return the evaluation score for a given GameState; higher score indicates a better situation for Player MAX.
# Knight_Rider's evaluation function is based on the number of remaining horses and their proximity to the
# opponent's apple (the latter factor is not too useful in its current form but at least motivates Knight_Rider
# to move horses toward the opponent's apple).
def getScore(state):
    score = pointMultiplier * state.points
    for x in range(boardWidth):  # Search board for any pieces
        for y in range(boardHeight):
            if state.board[x, y] == 1:
                appleDistance = (boardWidth - 1) - x + (boardHeight - 1) - y
                score += pieceValue - appleDistance
            elif state.board[x, y] == -1:
                appleDistance = x + y
                score -= pieceValue - appleDistance
    return score


# Check whether time limit has been reached
def timeOut():
    duration = datetime.now() - startTime
    return duration.seconds + duration.microseconds * 1e-6 >= timeLimit


# Use the minimax algorithm to look ahead <depthRemaining> moves and return the resulting score
def lookAhead(state, depthRemaining):
    if depthRemaining == 0 or state.gameOver:
        return getScore(state)

    if timeOut():
        return 0

    bestScore = -9e9 * state.playerToMove

    for move in getMoveOptions(state):
        projectedState = makeMove(state, move)  # Try out every possible move...
        score = lookAhead(projectedState, depthRemaining - 1)  # ... and score the resulting state

        if (state.playerToMove == 1 and score > bestScore) or (state.playerToMove == -1 and score < bestScore):
            bestScore = score  # Update bestScore if we have a new highest/lowest score for MAX/MIN

    return bestScore


# Set global variables and initialize any data structures that the player will need
def initPlayer(_startState, _timeLimit, _victoryPoints, _moveLimit, _assignedPlayer):
    global startState, timeLimit, victoryPoints, moveLimit, assignedPlayer, boardWidth, boardHeight
    startState, timeLimit, victoryPoints, moveLimit, assignedPlayer = _startState, _timeLimit, _victoryPoints, _moveLimit, _assignedPlayer
    (boardWidth, boardHeight) = startState.board.shape
    # TODO: Put any initialization code here


# Free up memory if player used huge data structures
def exitPlayer():
    return


# Compute the next move to be played; keep updating <favoredMove> until computation finished or time limit reached
def getMove(state):
    global startTime
    startTime = datetime.now()  # Remember computation start time
    moveList = getMoveOptions(state)  # Get the list of possible moves
    favoredMove = moveList[0]  # Just choose first move from the list for now, in case we run out of time
    favoredMoveScore = -9e9 * state.playerToMove  # Use this variable to remember the score for the favored move

    # Iterative deepening loop
    for lookAheadDepth in range(minLookAhead, maxLookAhead + 1):
        currBestMove = None  # Best move and score currently found during the current iteration (lookAheadDepth)
        currBestScore = -9e9 * state.playerToMove

        # Try every possible next move, evaluate it using Minimax, and pick the one with best score
        for move in moveList:
            projectedState = makeMove(state, move)
            score = lookAhead(projectedState,
                              lookAheadDepth - 1)  # Find score through MiniMax for current lookAheadDepth

            if timeOut():
                break

            if (state.playerToMove == 1 and score > currBestScore) or (
                    state.playerToMove == -1 and score < currBestScore):
                currBestMove, currBestScore = move, score  # Found new best move during this iteration

        if not timeOut():  # Pick the move from the last lookahead depth as new favorite, unless the lookahead was incomplete
            favoredMove, favoredMoveScore = currBestMove, currBestScore
            duration = datetime.now() - startTime
            print('Knight_Rider: Depth %d finished at %.4f s, favored move (%d,%d)->(%d,%d), score = %.2f'
                  % (lookAheadDepth, duration.seconds + duration.microseconds * 1e-6,
                     favoredMove[0], favoredMove[1], favoredMove[2], favoredMove[3], favoredMoveScore))
        else:
            print('Knight_Rider: Timeout!')

        if timeOut() or abs(
                favoredMoveScore) > victoryScoreThresh:  # Stop computation if timeout or certain victory/defeat predicted
            break

    return favoredMove
