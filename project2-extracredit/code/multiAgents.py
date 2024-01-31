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

        "*** YOUR CODE HERE ***"
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        minFoodDistance = min(foodDistances) if foodDistances else 0

        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        minGhostDistance = min(ghostDistances) if ghostDistances else 0

    	# Calculate evaluation function
        evaluation = successorGameState.getScore() + 1.0 / (minFoodDistance + 1) - 1.0 / (minGhostDistance + 1)
    
        return evaluation

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
        """
        numAgent = gameState.getNumAgents()
        ActionScore = []

        def _rmStop(List):
            return [x for x in List if x != 'Stop']

        def _miniMax(s, iterCount):
            if iterCount >= self.depth * numAgent or s.isWin() or s.isLose():
                return self.evaluationFunction(s)
            if iterCount % numAgent != 0:  # Ghost min
                result = 1e10
                for a in _rmStop(s.getLegalActions(iterCount % numAgent)):
                    sdot = s.generateSuccessor(iterCount % numAgent, a)
                    result = min(result, _miniMax(sdot, iterCount + 1))
                return result
            else:  # Pacman Max
                result = -1e10
                for a in _rmStop(s.getLegalActions(iterCount % numAgent)):
                    sdot = s.generateSuccessor(iterCount % numAgent, a)
                    result = max(result, _miniMax(sdot, iterCount + 1))
                    if iterCount == 0:
                        ActionScore.append(result)
                return result

        result = _miniMax(gameState, 0)
        return _rmStop(gameState.getLegalActions(0))[ActionScore.index(max(ActionScore))]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numOfAgents = gameState.getNumAgents()
        actionValues = []

        def removeStopActions(actionList):
            return [action for action in actionList if action != 'Stop']

        def alphaBetaSearch(state, iterCount, alpha, beta):
            if iterCount >= self.depth * numOfAgents or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if iterCount % numOfAgents != 0:  # Ghost's turn (min)
                result = float('inf')
                for action in removeStopActions(state.getLegalActions(iterCount % numOfAgents)):
                    nextState = state.generateSuccessor(iterCount % numOfAgents, action)
                    result = min(result, alphaBetaSearch(nextState, iterCount + 1, alpha, beta))
                    beta = min(beta, result)
                    if beta < alpha:
                        break
                return result
            else:  # Pacman's turn (max)
                result = float('-inf')
                for action in removeStopActions(state.getLegalActions(iterCount % numOfAgents)):
                    nextState = state.generateSuccessor(iterCount % numOfAgents, action)
                    result = max(result, alphaBetaSearch(nextState, iterCount + 1, alpha, beta))
                    alpha = max(alpha, result)
                    if iterCount == 0:
                        actionValues.append(result)
                    if beta < alpha:
                        break
                return result

        result = alphaBetaSearch(gameState, 0, -1e20, 1e20)
        return removeStopActions(gameState.getLegalActions(0))[actionValues.index(max(actionValues))]


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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    """
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    
    foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
    minFoodDistance = min(foodDistances) if foodDistances else 0

    ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
    minGhostDistance = min(ghostDistances) if ghostDistances else 0

    
    foodWeight = 1.0
    ghostWeight = -2.0
    scaredGhostWeight = -1.0

    
    evaluation = (
        currentGameState.getScore() +
        foodWeight / (minFoodDistance + 1) +
        ghostWeight / (minGhostDistance + 1)
    )

    
    for scaredTime in newScaredTimes:
        if scaredTime > 0:
            evaluation += scaredGhostWeight / (minGhostDistance + 1)

    return evaluation
 
    

# Abbreviation
better = betterEvaluationFunction

