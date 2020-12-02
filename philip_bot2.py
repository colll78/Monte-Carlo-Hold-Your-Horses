# Player Knight_Rider

import numpy as np
import math
from datetime import datetime

class GameState(object):
    __slots__ = ['board', 'playerToMove', 'gameOver', 'movesRemaining', 'points']


# Draw the board, pieces, and player names indicating active and winning players
# If <currentMove>, show currently moving piece somewhere between its start position (moveProgress = 0)
# and its end position (moveProgress = 1) for move animation
def displayState(state, playerNames, selectedSquare, currentMove=None, moveProgress=0):
    def drawPiece(pieceCode, x, y):
        if pieceCode == 1:
            color, mirror, shape, xEye = pieceColors[0], -1, horseShape, 60
        elif pieceCode == 2:
            color, mirror, shape, xEye = pieceColors[0], 1, appleShape, 0
        elif pieceCode == -1:
            color, mirror, shape, xEye = pieceColors[1], 1, horseShape, 40
        else:
            color, mirror, shape, xEye = pieceColors[1], 1, appleShape, 0

        poly = Polygon(
            [Point(x + 50 + mirror * (xPoint - 50) * squareSize / 100, y + yPoint * squareSize / 100) for
             (xPoint, yPoint)
             in shape])
        poly.setFill(color)
        poly.setOutline("black")
        poly.setWidth(2)
        poly.draw(win)

        if xEye > 0:
            eye = Circle(Point(x + xEye * squareSize / 100, y + 30), 3)
            eye.setFill("black")
            eye.setOutline("black")
            eye.setWidth(1)
            eye.draw(win)


    global boardWidth, boardHeight
    from graphics import GraphWin, Text, Point, Rectangle, Circle, Line, Polygon, update, color_rgb
    # Polygon coordinates for game pieces - awkward implementation but keeps graphics library use to a minimum
    horseShape = [(24, 87), (24, 78), (30, 73), (30, 68), (34, 60), (49, 45), (40, 44), (36, 43), (27, 47), (21, 47),
                  (16, 43), (16, 38), (15, 37), (31, 26), (35, 22), (38, 19), (41, 17), (47, 16), (47, 8), (54, 16),
                  (61, 18), (68, 22), (74, 30), (78, 38), (77, 51), (70, 73), (76, 78), (76, 87)]

    appleShape = [(52, 36), (60, 34), (67, 34), (75, 38), (81, 45), (83, 51), (83, 62), (80, 71), (74, 79), (67, 87),
                  (62, 89), (57, 89), (52, 87), (47, 87), (42, 89), (37, 89), (32, 87), (25, 79), (19, 71), (16, 62),
                  (16, 51), (18, 45), (24, 38), (32, 34), (39, 34), (47, 36), (44, 25), (36, 18), (40, 15), (46, 22),
                  (48, 26), (49, 17), (57, 9), (65, 6), (66, 14), (62, 21), (58, 25), (50, 28)]
    squareSize = 100  # Size of each square (pixels)
    textHeight = 50  # Height of the text display at the top of the window (pixels)
    pieceColors = [color_rgb(230, 20, 20), color_rgb(20, 200, 20)]
    squareColors = [color_rgb(40, 40, 210), color_rgb(50, 50, 255)]
    horseCoords = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
    appleCoords = (0, 0)
    playerCode = [1, -1]
    win = GraphWin("Hold Your Horses!", boardWidth * squareSize, textHeight + boardHeight * squareSize, autoflush=False)
    win.setBackground("black")
    win.delete('all')
    textPos = [boardWidth * squareSize / 4, boardWidth * squareSize * 3 / 4]

    for p in range(2):
        if state.gameOver == False:
            if state.playerToMove == playerCode[p]:
                t = Text(Point(textPos[p], textHeight / 2), '<< ' + playerNames[p] + ' >>')
            else:
                t = Text(Point(textPos[p], textHeight / 2), playerNames[p])
        else:
            if np.sign(state.points) == playerCode[p]:
                t = Text(Point(textPos[p], textHeight / 2), '!!! ' + playerNames[p] + ' !!!')
                t.setStyle("bold")
            else:
                t = Text(Point(textPos[p], textHeight / 2), playerNames[p])
        t.setFace("arial")
        t.setSize(min([int(textHeight / 3), 36]))
        t.setTextColor(pieceColors[p])
        t.draw(win)

    # Show squares and pieces
    for x in range(boardWidth):
        for y in range(boardHeight):
            r = Rectangle(Point(squareSize * x, textHeight + squareSize * y),
                          Point(squareSize * (x + 1), textHeight + squareSize * (y + 1)))
            if selectedSquare == (x, y):
                r.setFill("white")
            else:
                r.setFill(squareColors[(x + y) % 2])
            r.setWidth(0)
            r.draw(win)

            if state.board[x, y] != 0 and (currentMove == None or currentMove[:2] != (x, y)):
                drawPiece(state.board[x, y], squareSize * x, textHeight + squareSize * y)

    # Show moving piece somewhere between its start and end points (moveProgress between 0 and 1)
    if currentMove != None:
        x = moveProgress * (currentMove[2] - currentMove[0]) + currentMove[0]
        y = moveProgress * (currentMove[3] - currentMove[1]) + currentMove[1]
        drawPiece(state.playerToMove, squareSize * x, textHeight + squareSize * y)

    if currentMove == None:
        update()
    else:
        update(60)

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
maxLookAhead = 22  # Maximum search depth

apple_loc = None
mating_squares = None
knight_distance = {}

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


def attackers_defenders_2(state, player_number, x, y):
    direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]
    attack_defend = 0
    for (dx, dy) in direction:
        (xEnd, yEnd) = (x + dx, y + dy)
        if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight:
            if state.board[xEnd, yEnd] == player_number:
                attack_defend += 1
            elif state.board[xEnd, yEnd] == -player_number:
                attack_defend -= 1
    return attack_defend


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
        if attackers_defenders(state, x[2], x[3]) >= 0:
            #or state.board[x[2],x[3]] == -state.playerToMove:
            filtered_moves.append(x)
    return filtered_moves or moves


def is_loss(state):
    for square in mating_squares[state.playerToMove]:
        if all(state.board[square] != state.playerToMove for square in mating_squares[-state.playerToMove]):
            if attackers_defenders(state, square[0], square[1]) < 0:
                return True
    return False


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


def knight_dist(start_x, start_y, goal_x, goal_y):
    x = abs(goal_x - start_x)
    y = abs(goal_y - start_y)
    # print("Horizontal Distance: ", x)
    # print("Vertical Distance: ", y)
    if x == y == 1 and (start_x, start_y) == (0, 0) or (start_x, start_y) == (boardWidth-1, boardHeight-1) or (start_x, start_y) == (0, boardHeight-1) or (start_x, start_y) == (boardWidth-1, 0):
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


# Return the evaluation score for a given GameState; higher score indicates a better situation for Player MAX.
# Knight_Rider's evaluation function is based on the number of remaining horses and their proximity to the
# opponent's apple (the latter factor is not too useful in its current form but at least motivates Knight_Rider
# to move horses toward the opponent's apple).
def getScore(state):
    score = pointMultiplier * state.points
    for square in mating_squares[1]:
        atk_def = attackers_defenders_2(state, 1, square[0], square[1])
        if atk_def <= 0:
            if state.board[square[0], square[1]] == -1:
                score -= 500
            else:
                score -= pieceValue/2

    # for square in mating_squares[state.playerToMove]:
    #     atk_def = attackers_defenders(state, square[0], square[1])
    #     if atk_def >= 0:
    #         if state.board[square[0], square[1]] == state.playerToMove:
    #             score += 2
    #         else:
    #             score += 1

    if assignedPlayer == 1:
        playerNames = ["philip_bot", "opponent"]
    else:
        playerNames = ['opponent', 'philip_bot']

    for x in range(boardWidth):  # Search board for any pieces
        for y in range(boardHeight):
            if state.board[x, y] == 1:
                # appleDistance = (boardWidth - 1) - x + (boardHeight - 1) - y
                #print("Piece: ", x, " ", y)
                # if attackers_defenders(state, x, y) + 1 < 0:
                #     score -= pieceValue/3
                # displayState(state, playerNames=playerNames, selectedSquare=(x, y))
                #apple_point = apple_loc[-1]
                appleDistance = knight_distance.get((x, y, -1))

                # print("Apple distance: ", appleDistance)
                # print(state.board)
                # print("Apple location: ", apple_point)
                # print("Horse location: (%s, %s)" % (x, y))
                # input("hellO")
                #print()
                #score += pieceValue - appleDistance
                score += pieceValue - appleDistance
            elif state.board[x, y] == -1:
                # if attackers_defenders(state, x, y) > 0:
                #     score += pieceValue/4
                # appleDistance = x + y
                appleDistance = knight_distance.get((x, y, 1))
                #score -= pieceValue - appleDistance
                score -= pieceValue - appleDistance
    return score


# Check whether time limit has been reached
def timeOut():
    duration = datetime.now() - startTime
    return duration.seconds + duration.microseconds * 1e-6 >= timeLimit


def lookAhead(state, depthRemaining, alpha, beta):
    if depthRemaining == 0 or state.gameOver:
        return getScore(state)

    if timeOut():
        return 0

    bestScore = -9e9 * state.playerToMove

    for move in getMoveOptions(state):
        projectedState = makeMove(state, move)  # Try out every possible move...
        score = lookAhead(projectedState, depthRemaining - 1, alpha, beta)  # ... and score the resulting state
        if state.playerToMove == 1:
            bestScore = max(score, bestScore)
            alpha = max(alpha, score)
            if beta <= alpha:
                break

        if (state.playerToMove == -1):
            bestScore = min(bestScore, score)
            beta = min(beta, score)
            if beta <= alpha:
                break

    return bestScore

#Use the minimax algorithm to look ahead <depthRemaining> moves and return the resulting score
# def lookAhead(state, depthRemaining):
#     if depthRemaining == 0 or state.gameOver:
#         return getScore(state)
#
#     if timeOut():
#         return 0
#
#     bestScore = -9e9 * state.playerToMove
#
#     for move in getMoveOptions(state):
#         projectedState = makeMove(state, move)  # Try out every possible move...
#         score = lookAhead(projectedState, depthRemaining - 1)  # ... and score the resulting state
#
#         if (state.playerToMove == 1 and score > bestScore) or (state.playerToMove == -1 and score < bestScore):
#             bestScore = score  # Update bestScore if we have a new highest/lowest score for MAX/MIN
#
#     return bestScore


def initPlayer(_startState, _timeLimit, _victoryPoints, _moveLimit, _assignedPlayer):
    global startState, timeLimit, victoryPoints, moveLimit, assignedPlayer, boardWidth, boardHeight, apple_loc, mating_squares, knight_distance
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

    for x in range(0, boardWidth):
        for y in range(0, boardHeight):
            if not knight_distance.get((x, y, 1)) and not knight_distance.get((x, y, -1)):
                knight_distance[(x, y, 1)] = knight_dist(x, y, p1_apple_loc[0], p1_apple_loc[1])
                knight_distance[(x, y, -1)] = knight_dist(x, y, p2_apple_loc[0], p2_apple_loc[1])



    mating_squares = {1: p1_mating_squares, -1: p2_mating_squares}


# Free up memory if player used huge data structures
def exitPlayer():
    return


# Compute the next move to be played; keep updating <favoredMove> until computation finished or time limit reached
def getMove(state):
    global startTime
    startTime = datetime.now()  # Remember computation start time
    moveList = filter_possible_moves(state)  # Get the list of possible moves
    favoredMove = moveList[0]  # Just choose first move from the list for now, in case we run out of time
    favoredMoveScore = -9e9 * state.playerToMove  # Use this variable to remember the score for the favored move

    # Iterative deepening loop
    for lookAheadDepth in range(minLookAhead, maxLookAhead + 1):
        currBestMove = None  # Best move and score currently found during the current iteration (lookAheadDepth)
        currBestScore = -9e9 * state.playerToMove

        # Try every possible next move, evaluate it using Minimax, and pick the one with best score
        for move in moveList:
            projectedState = makeMove(state, move)
            #score = lookAhead(projectedState, lookAheadDepth - 1)  # Find score through MiniMax for current lookAheadDepth
            score = lookAhead(projectedState, lookAheadDepth - 1, -9e9, 9e9)
            if timeOut():
                break

            if (state.playerToMove == 1 and score > currBestScore) or (
                    state.playerToMove == -1 and score < currBestScore):
                currBestMove, currBestScore = move, score  # Found new best move during this iteration

        if not timeOut():  # Pick the move from the last lookahead depth as new favorite, unless the lookahead was incomplete
            favoredMove, favoredMoveScore = currBestMove, currBestScore
            duration = datetime.now() - startTime
            # print('philip_bot: Depth %d finished at %.4f s, favored move (%d,%d)->(%d,%d), score = %.2f'
            #       % (lookAheadDepth, duration.seconds + duration.microseconds * 1e-6,
            #          favoredMove[0], favoredMove[1], favoredMove[2], favoredMove[3], favoredMoveScore))
        else:
            print('philip_bot: Timeout!')

        if timeOut() or abs(
                favoredMoveScore) > victoryScoreThresh:  # Stop computation if timeout or certain victory/defeat predicted
            break

    return favoredMove