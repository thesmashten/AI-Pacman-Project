# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        minNum = float('inf')
        currentFood = currentGameState.getFood()
        foodList = currentFood.asList()

        for food in foodList:
            minNum = min((manhattanDistance(newPos, food)), minNum)
            
        minNum = -minNum

        newGhostPos = childGameState.getGhostPositions()
        for ghost in newGhostPos:
            if (manhattanDistance(newPos, ghost) < 1):
                return -float('inf')

        return minNum


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        bestScore, bestMove = self.Mini_Max(gameState, self.index)
        return bestMove
        
    def isTerminal(self, gameState, agentIndex, numIndex, depth):
        childAgentIndex = 0
        lst = []
        if (gameState.isLose() or gameState.isWin() or (depth == self.depth)):
            lst.append(self.evaluationFunction(gameState))
            return lst
        elif agentIndex == numIndex:
            depth += 1
            childAgentIndex = self.index
            lst.append(gameState)
            lst.append(childAgentIndex)
            lst.append(depth)
            return lst
        else:
            childAgentIndex = agentIndex + 1
            lst.append(gameState)
            lst.append(childAgentIndex)
            lst.append(depth)
            return lst
        return 
        
                
    def Mini_Max(self, gameState, agentIndex, depth=0):
        numIndex = gameState.getNumAgents() - 1
            
        retList = self.isTerminal(gameState, agentIndex, numIndex, depth)
        if (len(retList) > 1):
            gameState = retList[0]
            childAgentIndex = retList[1]
            depth = retList[2]
        else:
            return retList
            #If Player == MIN
        if agentIndex != 0:
            return self.min_value(gameState, agentIndex, numIndex, depth, childAgentIndex)
            #Player is MAX
        else:
            return self.max_value(gameState, agentIndex, numIndex, depth, childAgentIndex)
                
    def min_value(self, gameState, agentIndex, numIndex, depth, childAgentIndex):
        min = float("inf")
        bestAction = None
        
        for legalAction in gameState.getLegalActions(agentIndex):
            successorGameState = gameState.getNextState(agentIndex, legalAction)
            newMin = self.Mini_Max(successorGameState, childAgentIndex, depth)[0]
            if newMin == min:
                bestAction = legalAction
            #return minimum of DFMiniMax(c, MAX) over c in ChildList
            elif newMin < min:
                min = newMin
                bestAction = legalAction
        return min, bestAction
    
    def max_value(self, gameState, agentIndex, numIndex, depth, childAgentIndex):
        max = -float("inf")
        bestAction = None
        
        for legalAction in gameState.getLegalActions(agentIndex):
            successorGameState = gameState.getNextState(agentIndex, legalAction)
            newMax = self.Mini_Max(successorGameState, childAgentIndex, depth)[0]
            if newMax == max:
                bestAction = legalAction
            #return maximum of DFMiniMax(c, MIN) over c in ChildList
            elif newMax > max:
                max = newMax
                bestAction = legalAction
        return max, bestAction
        
                
            
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        bestScore, bestMove = self.Alpha_Beta(gameState, 0, 0, -float("inf"), float("inf"))
        return bestMove

    
    def isTerminal(self, gameState, agentIndex, numIndex, depth):
        childAgentIndex = 0
        lst = []
        if (gameState.isLose() or gameState.isWin() or (depth == self.depth)):
            lst.append(self.evaluationFunction(gameState))
            return lst
        elif agentIndex == numIndex:
            depth += 1
            childAgentIndex = self.index
            lst.append(gameState)
            lst.append(childAgentIndex)
            lst.append(depth)
            return lst
        else:
            childAgentIndex = agentIndex + 1
            lst.append(gameState)
            lst.append(childAgentIndex)
            lst.append(depth)
            return lst
        return 
    
    def Alpha_Beta(self, gameState, agentIndex, depth, alpha, beta):
        numIndex = gameState.getNumAgents() - 1
            
        retList = self.isTerminal(gameState, agentIndex, numIndex, depth)
        if (len(retList) > 1):
            gameState = retList[0]
            childAgentIndex = retList[1]
            depth = retList[2]
        else:
            return retList
            
        #If Player == MAX
        if agentIndex == self.index:
            return self.max_value(gameState, agentIndex, numIndex, depth, childAgentIndex, alpha, beta)
        #Player is MIN
        else:
            return self.min_value(gameState, agentIndex, numIndex, depth, childAgentIndex, alpha, beta)
        
                
    def min_value(self, gameState, agentIndex, numIndex, depth, childAgentIndex, alpha, beta):
        bestAction = None        
        for legalAction in gameState.getLegalActions(agentIndex):
            if beta <= alpha:
                return beta, bestAction
            successorGameState = gameState.getNextState(agentIndex, legalAction)
            newBeta = self.Alpha_Beta(successorGameState, childAgentIndex, depth, alpha, beta)[0]
            if newBeta < beta:
                bestAction = legalAction
                beta = newBeta
        return beta, bestAction
    
    
    def max_value(self, gameState, agentIndex, numIndex, depth, childAgentIndex, alpha, beta):       
        bestAction = None
        for legalAction in gameState.getLegalActions(agentIndex):
            if float(beta) <= float(alpha):
                return alpha, bestAction
            successorGameState = gameState.getNextState(agentIndex, legalAction)
            newAlpha = self.Alpha_Beta(successorGameState, childAgentIndex, depth, alpha, beta)[0]
            if newAlpha > alpha:
                bestAction = legalAction
                alpha = newAlpha 
        return alpha, bestAction 

        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    
    # def getAction(self, gameState):
    #     bestScoreActionPair = self.expectimax(gameState, self.index)
    #     bestScore = bestScoreActionPair[0]
    #     bestMove =  bestScoreActionPair[1]
    #     return bestMove
    
    # def expectimax(gameState, agentIndex, depth=0):
    #     legalActionList = gameState.getLegalActions(agentIndex)
    #     numIndex = gameState.getNumAgents() - 1
    #     bestAction = None
    #     # If terminal(pos)
    #     if (gameState.isLose() or gameState.isWin() or depth == self.depth):
    #         return [self.evaluationFunction(gameState)]
    #     elif agentIndex == numIndex:
    #         depth += 1
    #         childAgentIndex = self.index
    #     else:
    #         childAgentIndex = agentIndex + 1

    #     numAction = len(legalActionList)
    #     #if player(pos) == MAX: value = -infinity
    #     if agentIndex == self.index:
    #         value = -float("inf")
    #     #if player(pos) == CHANCE: value = 0
    #     else:
    #         value = 0

    #     for legalAction in legalActionList:
    #         successorGameState = gameState.getNextState(agentIndex, legalAction)
    #         expectedMax = self.expectimax(successorGameState, childAgentIndex, depth)[0]
    #         if agentIndex == self.index:
    #             if expectedMax > value:
    #                 #value, best_move = nxt_val, move
    #                 value = expectedMax
    #                 bestAction = legalAction
    #         else:
    #             #value = value + prob(move) * nxt_val
    #             value = value + ((1.0/numAction) * expectedMax)
    #     return value, bestAction

    def isTerminal(self, gameState, agentIndex, numIndex, depth):
        childAgentIndex = 0
        lst = []
        if (gameState.isLose() or gameState.isWin() or (depth == self.depth)):
            lst.append(self.evaluationFunction(gameState))
            return lst
        elif agentIndex == numIndex:
            depth += 1
            childAgentIndex = self.index
            lst.append(gameState)
            lst.append(childAgentIndex)
            lst.append(depth)
            return lst
        else:
            childAgentIndex = agentIndex + 1
            lst.append(gameState)
            lst.append(childAgentIndex)
            lst.append(depth)
            return lst
        return 

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          This implementation adheres to the guideline on Games Lecture, slide 58
          during Summer 2019 at the University of Toronto.
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        
        def expectimax(gameState, agentIndex, depth=0):
            legalActions = gameState.getLegalActions(agentIndex)
            numIndex = gameState.getNumAgents() - 1
            bestAction = None
            
            retList = self.isTerminal(gameState, agentIndex, numIndex, depth) #check if player is at terminal node as before 
            if (len(retList) > 1):
                gameState = retList[0]
                childAgentIndex = retList[1]
                depth = retList[2]
            else:
                return retList

            #if player = MAX we know that the value = -infinity
            if agentIndex == self.index:
                value = -float("inf") #value represents our current score used to identify the best action
                
            #else if player = MIN we know that the value = 0
            else:
                value = 0

            for legalAction in legalActions:
                nextState = gameState.getNextState(agentIndex, legalAction)
                expectedMax = expectimax(nextState, childAgentIndex, depth)[0]
                if agentIndex == self.index:
                    if expectedMax > value:
                        value = expectedMax  #if expected max is greater than value, value should take precedence
                        bestAction = legalAction #pick the best action from bestAction 
                else:
                    numAction = len(legalActions) #get length of available actions
                    value = value + (numAction ** -1 * expectedMax) #if we know that player is not MAX then value is simply calculated using this equation
            return value, bestAction

        bestScore, bestMove = expectimax(gameState, self.index) #starts the recursive expectimax algorithm 
        return bestMove
        
        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostPositions = currentGameState.getGhostPositions()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    numCapsules = len(currentGameState.getCapsules())
    foodList = currentGameState.getFood().asList()
    numFood = currentGameState.getNumFood()
    badGhost = []
    yummyGhost = []
    total = 0
    win = 0
    lose = 0
    score = 0
    foodScore = 0
    ghost = 0
    if currentGameState.isWin():
        win = 10000000000000000000000000000
    elif currentGameState.isLose():
        lose = -10000000000000000000000000000
    score = 10000 * currentGameState.getScore()
    capsules = 10000000000/(numCapsules+1)
    for food in foodList:
        foodScore += 50/(manhattanDistance(pacmanPosition, food)) * numFood
    for index in range(len(scaredTimes)):
        if scaredTimes[index] == 0:
            badGhost.append(ghostPositions[index])
        else:
            yummyGhost.append(ghostPositions[index])
    for index in range(len(yummyGhost)):
        ghost += 1/(((manhattanDistance(pacmanPosition, yummyGhost[index])) * scaredTimes[index])+1)
    for death in badGhost:
        ghost +=  manhattanDistance(pacmanPosition, death)
    total = win + lose + score + capsules + foodScore + ghost
    return total   

# Abbreviation
better = betterEvaluationFunction
