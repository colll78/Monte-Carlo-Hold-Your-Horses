# Hold Your Horses!
# Game tournament interface for the course CS 470/670 at UMass Boston
# Version 1.1 on 11/20/2020 by Marc Pomplun

import importlib
import numpy as np
import random as rnd
import time
from datetime import datetime
from graphics import GraphWin, Text, Point, Rectangle, Circle, Line, Polygon, update, color_rgb

# Polygon coordinates for game pieces - awkward implementation but keeps graphics library use to a minimum
horseShape = [(24, 87), (24, 78), (30, 73), (30, 68), (34, 60), (49, 45), (40, 44), (36, 43), (27, 47), (21, 47),
              (16, 43), (16, 38), (15, 37), (31, 26), (35, 22), (38, 19), (41, 17), (47, 16), (47, 8), (54, 16),
              (61, 18), (68, 22), (74, 30), (78, 38), (77, 51), (70, 73), (76, 78), (76, 87)]

appleShape = [(52, 36), (60, 34), (67, 34), (75, 38), (81, 45), (83, 51), (83, 62), (80, 71), (74, 79), (67, 87),
              (62, 89), (57, 89), (52, 87), (47, 87), (42, 89), (37, 89), (32, 87), (25, 79), (19, 71), (16, 62),
              (16, 51), (18, 45), (24, 38), (32, 34), (39, 34), (47, 36), (44, 25), (36, 18), (40, 15), (46, 22),
              (48, 26), (49, 17), (57, 9), (65, 6), (66, 14), (62, 21), (58, 25), (50, 28)]

playerCode = [1, -1]  # Used to translate array indices 0 and 1 to player codes 1 and -1, resp.

boardWidth = 7
boardHeight = 6
squareSize = 100  # Size of each square (pixels)
textHeight = 50  # Height of the text display at the top of the window (pixels)
pieceColors = [color_rgb(230, 20, 20), color_rgb(20, 200, 20)]
squareColors = [color_rgb(40, 40, 210), color_rgb(50, 50, 255)]
horseCoords = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
appleCoords = (0, 0)

timeLimit = 3.0  # Limit for computer players' thinking  (seconds)
timeTolerance = 0.1  # Additional wait time (seconds) before timeout is called

victoryPoints = 100  # Number of points for the winner
moveLimit = 40  # Maximum number of moves


# If exceeded, game is a tie; otherwise, number of remaining moves is added to winner's score.

class GameState(object):
    __slots__ = ['board', 'playerToMove', 'gameOver', 'movesRemaining', 'points']


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
        [Point(x + 50 + mirror * (xPoint - 50) * squareSize / 100, y + yPoint * squareSize / 100) for (xPoint, yPoint)
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


# Draw the board, pieces, and player names indicating active and winning players
# If <currentMove>, show currently moving piece somewhere between its start position (moveProgress = 0)
# and its end position (moveProgress = 1) for move animation
def displayState(state, playerNames, selectedSquare, currentMove=None, moveProgress=0):
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


# Get coordinates of the square selected by the human player
def getClickedSquare():
    while True:
        clickPos = win.getMouse()
        squareX = int(clickPos.x / squareSize)
        squareY = int((clickPos.y - textHeight) / squareSize)
        if squareX >= 0 and squareX < boardWidth and squareY >= 0 and squareY < boardHeight:
            return (squareX, squareY)


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

# def playGame(indexPlayer1, indexPlayer2):
#     # Create initial game state
#     state = GameState()
#     state.board = np.zeros((boardWidth, boardHeight), dtype=int)
#     state.board[appleCoords] = 2
#     state.board[boardWidth - appleCoords[0] - 1, boardHeight - appleCoords[1] - 1] = -2
#
#     for (x, y) in horseCoords:
#         state.board[x, y] = 1
#         state.board[boardWidth - x - 1, boardHeight - y - 1] = -1
#
#     state.playerToMove = 1
#     state.movesRemaining = moveLimit
#     state.gameOver = False
#     state.points = 0
#
#     # Init players
#     moduleIndices = [indexPlayer1, indexPlayer2]
#     playerNames = [playerModuleList[indexPlayer1], playerModuleList[indexPlayer2]]
#     isHuman = [True, True]
#
#     #players[11].initPlayer(state, timeLimit, victoryPoints, moveLimit, playerCode[0])
#     for i in range(2):
#         if moduleIndices[i] >= 0:
#             players[moduleIndices[i]].initPlayer(state, timeLimit, victoryPoints, moveLimit, playerCode[i])
#             isHuman[i] = False
#
#     displayState(state, playerNames, None)
#
#     if not isHuman[0]:  # Brief delay to show the initial game state before a computer's first move
#         time.sleep(1)
#
#     while state.gameOver == False:
#         displayState(state, playerNames, None)
#         moveList = getMoveOptions(state)
#         playerIndex = (1 - state.playerToMove) // 2  # Index (0 or 1) of player to move
#
#         if isHuman[playerIndex]:  # Human player
#             repeatEntry = True
#             while repeatEntry:
#                 displayState(state, playerNames, None)
#                 legalStart = False  # Get start position for human's move
#                 while not legalStart:
#                     (xStart, yStart) = getClickedSquare()
#                     for (xS, yS, _, _) in moveList:
#                         if (xS, yS) == (xStart, yStart):
#                             legalStart = True
#                             break
#
#                 displayState(state, playerNames, (xStart, yStart))
#                 legalEnd = False  # Get end position for human's move
#                 while not legalEnd:
#                     (xEnd, yEnd) = getClickedSquare()
#                     for move in moveList:
#                         if move == (xStart, yStart, xEnd, yEnd):
#                             legalEnd = True
#                             repeatEntry = False
#                             break
#                     if (xStart, yStart) == (xEnd, yEnd):
#                         legalEnd = True
#             move = (xStart, yStart, xEnd, yEnd)
#         else:  # Computer player
#             startTime = datetime.now()
#             move = players[moduleIndices[playerIndex]].getMove(state)
#             # startTime = datetime.now()
#             # players[11].getMove(state)
#
#             duration = datetime.now() - startTime
#
#             if duration.seconds + duration.microseconds * 1e-6 >= timeLimit + timeTolerance:
#                 print("Time violation by player " + playerNames[playerIndex])
#                 move = moveList[
#                     0]  # If computatiomn took too long or illegal move is returned, just pick first move from list
#             else:
#                 if not (move in moveList):
#                     print("Illegal move by player " + playerNames[playerIndex])
#                     move = moveList[0]
#
#         for i in range(1, 15):
#             displayState(state, playerNames, None, move, i / 14)
#
#         state = makeMove(state, move)
#         displayState(state, playerNames, None)
#
#     for i in range(2):
#         if not isHuman[i]:
#             players[moduleIndices[i]].exitPlayer()
#
#     if state.points > 0:
#         return (state.points, 0)
#     if state.points < 0:
#         return (0, -state.points)
#     return (victoryPoints // 2, victoryPoints // 2)

# Play a game with players indicated by their indices in playerList. Human player is indicated by index == None.
# Return the points for each player (winner: move advantage over opponent; loser: 0)
def playGame(indexPlayer1, indexPlayer2):
    # Create initial game state
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

    # Init players
    moduleIndices = [indexPlayer1, indexPlayer2]
    playerNames = [playerModuleList[indexPlayer1], playerModuleList[indexPlayer2]]
    isHuman = [True, True]

    #players[11].initPlayer(state, timeLimit, victoryPoints, moveLimit, playerCode[0])
    for i in range(2):
        if moduleIndices[i] >= 0:
            players[moduleIndices[i]].initPlayer(state, timeLimit, victoryPoints, moveLimit, playerCode[i])
            isHuman[i] = False

    displayState(state, playerNames, None)

    if not isHuman[0]:  # Brief delay to show the initial game state before a computer's first move
        time.sleep(1)

    while state.gameOver == False:
        displayState(state, playerNames, None)
        moveList = getMoveOptions(state)
        playerIndex = (1 - state.playerToMove) // 2  # Index (0 or 1) of player to move

        if isHuman[playerIndex]:  # Human player
            repeatEntry = True
            while repeatEntry:
                displayState(state, playerNames, None)
                legalStart = False  # Get start position for human's move
                while not legalStart:
                    (xStart, yStart) = getClickedSquare()
                    for (xS, yS, _, _) in moveList:
                        if (xS, yS) == (xStart, yStart):
                            legalStart = True
                            break

                displayState(state, playerNames, (xStart, yStart))
                legalEnd = False  # Get end position for human's move
                while not legalEnd:
                    (xEnd, yEnd) = getClickedSquare()
                    for move in moveList:
                        if move == (xStart, yStart, xEnd, yEnd):
                            legalEnd = True
                            repeatEntry = False
                            break
                    if (xStart, yStart) == (xEnd, yEnd):
                        legalEnd = True
            move = (xStart, yStart, xEnd, yEnd)
        else:  # Computer player
            startTime = datetime.now()
            move = players[moduleIndices[playerIndex]].getMove(state)
            # startTime = datetime.now()
            # players[11].getMove(state)

            duration = datetime.now() - startTime

            if duration.seconds + duration.microseconds * 1e-6 >= timeLimit + timeTolerance:
                print("Time violation by player " + playerNames[playerIndex])
                move = moveList[
                    0]  # If computatiomn took too long or illegal move is returned, just pick first move from list
            else:
                if not (move in moveList):
                    print("Illegal move by player " + playerNames[playerIndex])
                    move = moveList[0]

        for i in range(1, 15):
           displayState(state, playerNames, None, move, i / 14)

        state = makeMove(state, move)
        #print(state.board)
        #displayState(state, playerNames, None)

    for i in range(2):
        if not isHuman[i]:
            players[moduleIndices[i]].exitPlayer()

    if state.points > 0:
        return (state.points, 0)
    if state.points < 0:
        return (0, -state.points)
    return (victoryPoints // 2, victoryPoints // 2)


# Play a game by picking computer players by their index in <players> or putting <None> for a human player
def singleGame(indexPlayer1, indexPlayer2):
    (points1, points2) = playGame(indexPlayer1, indexPlayer2)
    playerNames = [playerModuleList[indexPlayer1], playerModuleList[indexPlayer2]]
    print(playerNames[0] + ' vs. ' + playerNames[1] + ' ' + str(points1) + ' - ' + str(points2))
    time.sleep(2)
    return (points1, points2)


# Play a round-robin computer player tournament in which any two players compete against each other twice (to play each side once)
# Afterwards, rank players by number of victories. If victories are identical, rank by number of points.
def computerTournament(playerIndexList):
    gameList = []
    playersList = []
    for player1 in range(len(playerIndexList)):
        for player2 in range(len(playerIndexList)):
            if player1 != player2:
                if playerModuleList[playerIndexList[player1]] != playerModuleList[playerIndexList[player2]] :
                    playersList.append(playerModuleList[playerIndexList[player1]])
                    playersList.append(playerModuleList[playerIndexList[player2]])
                    gameList.append((player1, player2))

    rnd.shuffle(gameList)
    victories = np.zeros(len(playerIndexList), dtype=float)
    points = np.zeros(len(playerIndexList), dtype=int)

    name_vics = {}
    for (player1, player2) in gameList:
        (points1, points2) = singleGame(playerIndexList[player1], playerIndexList[player2])
        points[player1] += points1
        points[player2] += points2
        if points1 > points2:
            victories[player1] += 1
            name_vics[playerModuleList[playerIndexList[player1]]] = name_vics.get(playerModuleList[playerIndexList[player1]], 0) + 1
        elif points1 < points2:
            victories[player2] += 1
            name_vics[playerModuleList[playerIndexList[player2]]] = name_vics.get(playerModuleList[playerIndexList[player2]], 0) + 1
        else:
            name_vics[playerModuleList[playerIndexList[player1]]] = name_vics.get(playerModuleList[playerIndexList[player1]], 0) + 0.5
            name_vics[playerModuleList[playerIndexList[player2]]] = name_vics.get(playerModuleList[playerIndexList[player2]], 0) + 0.5
            victories[player1] += 0.5
            victories[player2] += 0.5

    rankingScore = 1e6 * victories + points
    ranking = np.argsort(rankingScore)

    print('\nFinal Standings:\n')
    print('Name                          Victories Points\n')
    print(playersList)
    print(playerIndexList)
    playersList = set(playersList)
    if len(playersList) == 2 and len(playerIndexList) > 2:

        name1 = playerModuleList[playerIndexList[0]]
        name2 = playerModuleList[playerIndexList[1]]
        spaces1 = ' ' * (30 - len(name1))
        spaces2 = ' ' * (30 - len(name2))
        score1 = 0
        score2 = 0
        for i in range(0, len(victories)):
            if playerModuleList[playerIndexList[i]] == playerModuleList[playerIndexList[0]]:
                score1 += victories[i]
            elif playerModuleList[playerIndexList[i]] == playerModuleList[playerIndexList[1]]:
                score2 += victories[i]

        print(name1 + spaces1 + str(score1))
        print(name2 + spaces2 + str(score2))
    else:
        for r in ranking[::-1]:
            name = playerModuleList[playerIndexList[r]]
            spaces = ' ' * (30 - len(name))
            print(name + spaces + str(victories[r]) + '\t\t' + str(points[r]))

    print('')
    print('')
    print('')
    time.sleep(4)


# Main script
if __name__ == "__main__":
    win = GraphWin("Hold Your Horses!", boardWidth * squareSize, textHeight + boardHeight * squareSize, autoflush=False)
    win.setBackground("black")

    # Names of player files (without '.py' extension).
    #                         0                 1                  2            3              4                5               6               7                    8                    9                        10                      11                 12                 13             14
    playerModuleList = ['Knight_Rider', 'HorseHoldZeroTree', 'Brain_Fog', 'philip_bot', 'one_horse_bot', 'RaveHoldHorses', "Dark_Knight", "HoldHorsesRave", "obfuscated_knight", "HoldHorsesRaveV2", "HoldHorsesRaveV2Depricated", "HoldHorseZeroV2", "SuperDepricated", "ParallelRave", "HoldHorseZeroV3"]
    name_to_ind = {name: i for i, name in enumerate(playerModuleList)}

    players = []  # Import player modules
    for player in playerModuleList:
        players.append(importlib.import_module(player))

    playerModuleList.append('Human Player')
    #singleGame(-1, 3)       # Play single game (refer to players by index in playerModuleList; -1 means human player)
    #computerTournament([1, 1, 1])  # Play a tournament with any number of computer players (list of indices as above)
    #computerTournament([3,0])
    #computerTournament([0,4,3])
    #computerTournament([3,4])
    #computerTournament([5, 6, 1])
    #computerTournament([6, 1])
    # computerTournament([7, 5])
    # computerTournament([7, 5])
    #computerTournament([10, 9, 9, 9])
    knight_rider = name_to_ind.get("Knight_Rider")
    hold_horse_zero_v2 = name_to_ind.get("HoldHorseZeroV2")
    hold_horse_rave = name_to_ind.get("HoldHorsesRave")
    dark_knight = name_to_ind.get("Dark_Knight")
    rave_v2_depricated = name_to_ind.get("HoldHorsesRaveV2Depricated")
    super_depricated = name_to_ind.get("SuperDepricated")
    parallel_rave = name_to_ind.get("ParallelRave")
    one_horse_bot = name_to_ind.get("one_horse_bot")
    hold_horses_zero_v3 = name_to_ind.get("HoldHorseZeroV3")
    #computerTournament([parallel_rave, hold_horse_rave, hold_horse_rave])
    #computerTournament([parallel_rave, dark_knight, parallel_rave, dark_knight, parallel_rave, dark_knight, parallel_rave, dark_knight])
    #computerTournament([dark_knight, hold_horses_zero_v3, dark_knight, hold_horses_zero_v3])
    #computerTournament([dark_knight, one_horse_bot, dark_knight, one_horse_bot])
    computerTournament([dark_knight, hold_horse_rave, dark_knight, hold_horse_rave])
    #computerTournament([hold_horse_rave, hold_horses_zero_v3, hold_horse_rave, hold_horses_zero_v3])
    #computerTournament([hold_horse_rave, dark_knight, hold_horse_zero_v2])  #, dark_knight, hold_horse_rave, hold_horse_rave, dark_knight, hold_horse_rave])
    #computerTournament([dark_knight, hold_horse_zero_v2, dark_knight, hold_horse_zero_v2, dark_knight, hold_horse_zero_v2, dark_knight, hold_horse_zero_v2])
    #computerTournament([hold_horse_rave, hold_horse_zero_v2, hold_horse_rave, hold_horse_zero_v2, hold_horse_rave, hold_horse_zero_v2, hold_horse_rave, hold_horse_zero_v2])
    #computerTournament([super_depricated, dark_knight, super_depricated, dark_knight, super_depricated])

    #computerTournament([rave_v2_depricated, dark_knight, rave_v2_depricated, dark_knight, rave_v2_depricated, dark_knight])
    #computerTournament([hold_horse_zero_v2, dark_knight, hold_horse_zero_v2, dark_knight, hold_horse_zero_v2, dark_knight])
    #computerTournament([6, 7, 7, 7, 7, 7, 7])
    #computerTournament([7, 11, 11, 11, 11, 11])
    #computerTournament([6, 8])
    #computerTournament([7, 6, 6, 6, 6, 6])

    win.close()
