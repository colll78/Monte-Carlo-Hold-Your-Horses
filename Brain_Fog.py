# Player Brain_Fog

import random as rnd
import numpy as np

class GameState(object):
    __slots__ = ['board', 'playerToMove', 'gameOver', 'movesRemaining', 'points']

# Global variables
boardWidth = 0            # These 3 global variables are set by the getMove function. This is not...
boardHeight = 0           # ... an elegant solution but an efficient one.

# Compute list of legal moves for a given GameState and the player moving next 
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

# Set global variables and initialize any data structures that the player will need
def initPlayer(_startState, _timeLimit, _victoryPoints, _moveLimit, _assignedPlayer):
    global boardWidth, boardHeight
    (boardWidth, boardHeight) = _startState.board.shape

# Free up memory if player used huge data structures
def exitPlayer():
    return

# Compute the next move to be played; Brain_Fog simply picks a random legal move
def getMove(state):
    moveList = getMoveOptions(state)                # Get the list of possible moves
    return rnd.choice(moveList)                     # Randomly select one of them 
