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
        # Collect legal moves and successor states
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

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        food = currentGameState.getFood().asList()
        position = successorGameState.getPacmanPosition()
        badMove = float("-inf")

        if action == "Stop": # Stopping is usually not to Pacman's benefit
            return badMove

        for state in newGhostStates: # Pacman should never go to where a ghost can be
            if state.getPosition() == tuple(position) and state.scaredTimer is 0:
                return badMove

        maxDistance = badMove
        for f in food: #Finds the closest food relative to Pacman's position
            distance = -manhattanDistance(f, position) # Use Manhattan Distance because it is simple. Won't work well for complex mazes
            if distance > maxDistance:
                maxDistance = distance

        return maxDistance

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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        
        #Recursively finds max value for pacman agents
        def pacmanMax(mostAgents, index, currentState, currentDepth, actions):
            possibilities = []
            for action in actions:
                possibilities.append((miniMax(mostAgents, index + 1, currentState.generateSuccessor(index, action), currentDepth), action))
            if currentDepth == 0:
                return max(possibilities)[1] #Returns best action to find max value
            return max(possibilities)[0] #Returns max value
            
        #Recursively finds min value for ghost agents
        def ghostMin(mostAgents, index, currentState, currentDepth, actions):
            possibilities = []
            for action in actions:
                possibilities.append(miniMax(mostAgents, index + 1, currentState.generateSuccessor(index, action), currentDepth))
            return min(possibilities)
        
        def miniMax(mostAgents, index, currentState, currentDepth):
            if mostAgents == index:
                index = 0
                currentDepth += 1

            legalActions = currentState.getLegalActions(index)
            if currentDepth == self.depth or not legalActions:
                return self.evaluationFunction(currentState)

            #Checks if agent is Pacman or ghost
            if index == 0:
                return pacmanMax(mostAgents, index, currentState, currentDepth, legalActions)
            return ghostMin(mostAgents, index, currentState, currentDepth, legalActions)

        return miniMax(gameState.getNumAgents(), 0, gameState, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        #Tries to maximize Pacman values
        def alphaBetaPacman(mostAgents, index, currentState, currentDepth, alpha, beta, actions):
            possibilities = []
            bestValue = float("-inf")

            for action in actions:
                miniValue = alphaBetaPruning(mostAgents, index + 1, currentState.generateSuccessor(index, action), currentDepth, alpha, beta)
                possibilities.append((miniValue, action))
                bestValue = max(bestValue, miniValue)

                #"Prunes" nodes by checking the beta value
                if bestValue > beta:
                    return bestValue
                alpha = max(alpha, bestValue)

            if currentDepth == 0:
                return max(possibilities)[1] #Returns best action to find max value
            return max(possibilities)[0] #Returns max value
            
        #Tries to minimize ghost values
        def alphaBetaGhost(mostAgents, index, currentState, currentDepth, alpha, beta, actions):
            possibilities = []
            bestValue = float("inf")

            for action in actions:
                nextValue = alphaBetaPruning(mostAgents, index + 1, currentState.generateSuccessor(index, action), currentDepth, alpha, beta)
                possibilities.append(nextValue)
                bestValue = min(bestValue, nextValue)
                
                #"Prunes" nodes by checking the alpha value
                if bestValue < alpha:
                    return bestValue
                beta = min(beta, bestValue)

            return min(possibilities)

        def alphaBetaPruning(mostAgents, index, currentState, currentDepth, alpha, beta):
            if mostAgents == index:
                index = 0
                currentDepth += 1

            legalActions = currentState.getLegalActions(index)
            if currentDepth == self.depth or not legalActions:
                return self.evaluationFunction(currentState)

            #Runs a different method for Pacman and the ghosts
            if index == 0:
                return alphaBetaPacman(mostAgents, index, currentState, currentDepth, alpha, beta, legalActions)
            return alphaBetaGhost(mostAgents, index, currentState, currentDepth, alpha, beta, legalActions)

        return alphaBetaPruning(gameState.getNumAgents(), 0, gameState, 0, float("-inf"), float("inf"))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
       
       #This is copied from the minimax problem above
        def expectiPacman(mostAgents, index, currentState, currentDepth, actions):
            possibilities = []
            for action in actions:
                possibilities.append((expectimax(mostAgents, index + 1, currentState.generateSuccessor(index, action), currentDepth), action))
            if currentDepth == 0:
                return max(possibilities)[1] #Returns best action to find max value
            return max(possibilities)[0] #Returns max value
            
        #Uses a uniform distribution to find expected value
        def expectiGhost(mostAgents, index, currentState, currentDepth, actions):
            probability = 1.0 / len(actions)
            expected = 0

            for action in actions:
                value = expectimax(mostAgents, index + 1, currentState.generateSuccessor(index, action), currentDepth)
                expected += probability * value

            return expected

        def expectimax(mostAgents, index, currentState, currentDepth):
            if mostAgents == index:
                index = 0
                currentDepth += 1

            legalActions = currentState.getLegalActions(index)
            if currentDepth == self.depth or not legalActions:
                return self.evaluationFunction(currentState)

            if index == 0:
                return expectiPacman(mostAgents, index, currentState, currentDepth, legalActions)
            return expectiGhost(mostAgents, index, currentState, currentDepth, legalActions)

        return expectimax(gameState.getNumAgents(), 0, gameState, 0)
           
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Gives a score to Pacman to encourage him to change his position
    depending on the state of the game. How this is done is detailed below.
    """

    returnScore = 0
    position = currentGameState.getPacmanPosition()
    
    #Ghosts being scared is good because Pacman can't die from a scare ghost, so this adds to the score
    for time in [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]:
        returnScore += time

    #Power pellets should be eaten in order to make the ghosts weak, so this takes away from the score until there are no more
    for powerPellet in currentGameState.getCapsules():
        returnScore -= manhattanDistance(position, powerPellet)

    #We want Pacman to be close to food and to eat as much as possible, so this takes away from the score
    for food in currentGameState.getFood().asList():
        returnScore -= manhattanDistance(position, food)

    #Pacman can ignore the ghosts until they become very close, in which case he gets very negative feedback to make sure he avoids them
    ghostPositions = currentGameState.getGhostPositions()
    minDistance = ghostPositions[0]
    for ghostPosition in ghostPositions[1:]:
        if manhattanDistance(position, ghostPosition) < manhattanDistance(position, minDistance):
            minDistance = ghostPosition   
    if manhattanDistance(position, minDistance) <= 1:
        returnScore -= 100

    return returnScore + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
