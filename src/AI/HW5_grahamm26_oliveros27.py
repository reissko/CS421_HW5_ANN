import random
import heapq
import itertools
import sys
import numpy as np
sys.path.append("..")
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *

##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "turnin")
        
        # Initialize ANN with hardcoded weights (4 hidden neurons)
        self.ann = ANN(input_size=4, hidden_size=4, output_size=1)
        self.ann.weights_input_hidden = np.array([[-2.8943945301166183, 0.911838189385934, 0.547105547315571, -0.7021427953476562], [1.2272950959769133, 1.7066117378727172, -0.86160231065697, 1.2161389144294046], [-0.3798226411728106, -0.20591354587763672, -1.1890546860942475, -1.0701304775674598], [-0.9302769937402349, 0.950438612078162, -0.10685757112747142, -0.2971111127598313]])
        self.ann.weights_hidden_output = np.array([[-3.3947272864965368], [2.5350349707020725], [0.09994085951542703], [-0.1927737979074666]])
        # Initialize biases to zeros (matching the 4 hidden neurons)
        self.ann.bias_hidden = np.zeros((1, 4))
        self.ann.bias_output = np.zeros((1, 1))
    
    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        if currentState.phase == SETUP_PHASE_1:
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    x = random.randint(0, 9)
                    y = random.randint(0, 3)
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    x = random.randint(0, 9)
                    y = random.randint(6, 9)
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    #getMove
    #Description: Gets the next move from the AIPlayer
    #
    #
    #Return: The Move with the best fScore 
    ##
    def getMove(self, currentState):
        legalMoves = listAllLegalMoves(currentState)
        
        if not legalMoves:
            return None
        
        nodes = []
        for move in legalMoves:
            nextState = getNextState(currentState, move)
            utility = self.utility(currentState, nextState)
            node = self.node(move, nextState, utility, None, 1)
            nodes.append(node)
        
        bestNode = self.bestMove(nodes)
        return bestNode["move"]
    
    ##
    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        return enemyLocations[0]

    ##
    #registerWin
    #
    def registerWin(self, hasWon):
        pass

    ##
    # utility
    #
    # examines a GameState object and returns a heuristic guess of how "good" that game state is on a scale of 0..1.
    # Start of the game should return 0.5
    # When the game is almost won
    #
    def utility(self, currentState, nextState):
        myId = currentState.whoseTurn
        next_inv = nextState.inventories[myId]
                
        # Food progress (0-1, where 1 = food goal reached)
        food_score = self.compute_food_score(nextState)
        
        # Route efficiency (already in [0,1])
        route_score = self.compute_route_score(nextState)
        normalized_route = route_score  # already in [0,1] range
        
        # Unit composition [0-1]]
        unit_score = self.compute_unit_composition_score(nextState)
        normalized_units = max(0, min(1, unit_score)) 
        
        # 4. Military aggression (normalize to 0-1)
        aggro_score = self.compute_rsoldier_aggression_score(nextState)
        normalized_aggro = max(0, min(1, aggro_score))  
        
        # Use ANN to predict utility
        scores = self.scores_to_nn_input(food_score, normalized_units, normalized_route, normalized_aggro)
        # Flatten the 4x1 array to 1D for feedforward (feedforward expects 1D array of 4 elements)
        scores_flat = scores.flatten()
        # Ensure it's a 1D array with exactly 4 elements
        if len(scores_flat) != 4:
            raise ValueError(f"Expected 4 inputs, got {len(scores_flat)}: {scores_flat}")
        base_utility = self.ann.feedforward(scores_flat).item()
        
        # if food goal reached, should be close to 1.0
        if next_inv.foodCount >= FOOD_GOAL:
            base_utility = max(base_utility, 0.95) 
        
        # if no food left, should be close to 0.0
        if next_inv.foodCount == 0:
            base_utility = min(base_utility, 0.05)  # Near-certain loss
        
        return base_utility


    

    ## compute_unit_composition_score
    # computes a score [0,1] based on having a balanced set of unit types
    # Higher score means more balanced  
    def compute_unit_composition_score(self, state):
        myId = state.whoseTurn
        myInv = state.inventories[myId]

        has_worker = any(a.type == WORKER for a in myInv.ants)
        has_r_soldier = any(a.type == R_SOLDIER for a in myInv.ants)

        score = 0.0
        if has_worker:
            score += 0.5
        if has_r_soldier:
            score += 0.5
        return score

    ## compute_food_score
    # computes a score [0,1] based on food progress toward the goal
    # Higher score means closer to food goal
    def compute_food_score(self, state):
        myId = state.whoseTurn
        myInv = state.inventories[myId]
        food_score = min(myInv.foodCount / float(FOOD_GOAL), 1.0)
        return food_score

    ##
    # compute_route_score
    #
    # computes a score [0,1] based on the delivery potential of current workers
    # Higher score means shorter routes on average
    #
    def compute_route_score(self, state):
        myId = state.whoseTurn
        myInv = state.inventories[myId]
        myWorkers = getAntList(state, myInv.player, (WORKER,))
        foods = getConstrList(state, NEUTRAL, (FOOD,))

        deposit_coords = []
        if myInv.getAnthill() is not None:
            deposit_coords.append(myInv.getAnthill().coords)
        for t in myInv.getTunnels():
            deposit_coords.append(t.coords)

        # helper function to find the shortest distance from a coord to any deposit
        def min_deposit_dist_from(coord):
            best = None
            for dep in deposit_coords:
                d = stepsToReach(state, coord, dep)
                if d >= 0:
                    best = d if best is None else min(best, d)
            return best

        MAX_ROUTE = 20.0
        route_lengths = []

        for w in myWorkers:
            if getattr(w, 'carrying', False):
                ddep = min_deposit_dist_from(w.coords)
                if ddep is not None:
                    route_lengths.append(ddep)
            else:
                best_dtofood = None
                best_food = None
                for f in foods:
                    dtof = stepsToReach(state, w.coords, f.coords)
                    if dtof >= 0 and (best_dtofood is None or dtof < best_dtofood):
                        best_dtofood = dtof
                        best_food = f
                if best_food is not None and best_dtofood is not None:
                    ddep = min_deposit_dist_from(best_food.coords)
                    if ddep is not None:
                        route_lengths.append(best_dtofood + ddep)

        if len(route_lengths) == 0:
            return 0.0

        avg_route = sum(route_lengths) / float(len(route_lengths))
        route_score = 1.0 - clamp(avg_route / MAX_ROUTE)
        return route_score


    ## compute_rsoldier_aggression_score
    # 
    # 
    # makes a score [0.0, 1.0] based on how effectively r_soldiers are positioned to attack enemies
    # Priority: Enemy workers first, then enemy anthill/queen
    def compute_rsoldier_aggression_score(self, state):
        myId = state.whoseTurn
        enemyId = 1 - myId
        myInv = state.inventories[myId]
        enInv = state.inventories[enemyId]
        my_rsoldiers = [a for a in myInv.ants if a.type == R_SOLDIER]
        enemy_workers = getAntList(state, enemyId, (WORKER,))
        enemy_hill = enInv.getAnthill() if enInv is not None else None
        enemy_queen = enInv.getQueen() if enInv is not None else None

        if len(my_rsoldiers) == 0:
            return 0.0

        total_aggression_score = 0.0
        
        for rsoldier in my_rsoldiers:
            soldier_score = 0.0
            
            # PRIORITY 1: Target enemy workers (higher weight)
            if len(enemy_workers) > 0:
                best_worker_distance = float('inf')
                for worker in enemy_workers:
                    distance = stepsToReach(state, rsoldier.coords, worker.coords)
                    if distance >= 0:
                        best_worker_distance = min(best_worker_distance, distance)
                
                if best_worker_distance != float('inf'):
                    MAX_WORKER_DISTANCE = 10.0  # Smaller max for worker targeting
                    worker_score = 1.0 - clamp(best_worker_distance / MAX_WORKER_DISTANCE)
                    soldier_score += 0.7 * worker_score  # 70% weight for worker targeting
            
            # PRIORITY 2: Target enemy anthill/queen (lower weight, only if no workers or as secondary)
            secondary_targets = []
            if enemy_hill:
                secondary_targets.append(enemy_hill.coords)
            if enemy_queen:
                secondary_targets.append(enemy_queen.coords)
            
            if secondary_targets:
                best_secondary_distance = float('inf')
                for target_coords in secondary_targets:
                    distance = stepsToReach(state, rsoldier.coords, target_coords)
                    if distance >= 0:
                        best_secondary_distance = min(best_secondary_distance, distance)
                
                if best_secondary_distance != float('inf'):
                    MAX_SECONDARY_DISTANCE = 15.0
                    secondary_score = 1.0 - clamp(best_secondary_distance / MAX_SECONDARY_DISTANCE)
                    
                    # If no enemy workers exist, give full weight to secondary targets
                    # Otherwise, give reduced weight (30%)
                    weight = 1.0 if len(enemy_workers) == 0 else 0.3
                    soldier_score += weight * secondary_score
            
            total_aggression_score += soldier_score
        
        # Average aggression score across all r_soldiers
        avg_aggression = total_aggression_score / len(my_rsoldiers)
        
        # Bonus for eliminating enemy workers (strategic progress)
        initial_enemy_workers = 2  # Assume enemy starts with ~2 workers typically
        if len(enemy_workers) < initial_enemy_workers:
            elimination_bonus = 0.5 * (initial_enemy_workers - len(enemy_workers))
            avg_aggression += elimination_bonus
        
        return clamp(avg_aggression)
        
    ## scores_to_nn_input
    # Maps the outputs from the four score computation functions to a 4x1 vector
    # suitable for neural network input
    #
    # Parameters:
    #   food_score - output from compute_food_score [0, 1]
    #   unit_score - output from compute_unit_composition_score [0, 1]
    #   route_score - output from compute_route_score [0, 1]
    #   aggro_score - output from compute_rsoldier_aggression_score [0, 1]
    #
    # Returns:
    #   numpy array of shape (4, 1) containing the four scores
    #
    def scores_to_nn_input(self, food_score, unit_score, route_score, aggro_score):
        return np.array([[food_score], [unit_score], [route_score], [aggro_score]])

    ## Node representation
    #
    def node(self, move, state, utility, parent, depth):
        depth += parent['depth'] if parent is not None else 0.0
        g = depth * 1.0
        h = 1.0 - utility
        f = g + h

        return {
            "move": move,
            "state": state,
            "parent": parent,
            "depth": depth,
            "gScore": g,
            "hScore": h,
            "fScore": f,
        }
    
    ## Expand a node to generate child nodes
    #
    def expandNode(self, node, currentState):
        legalMoves = listAllLegalMoves(currentState)
        newNodes = []
        for move in legalMoves:
            nextState = getNextState(currentState, move)
            utility = self.utility(currentState, nextState)
            child = self.node(move, nextState, utility, node, node["depth"] + 1)
            newNodes.append(child)
        return newNodes
    
    ## Best move from a list of nodes
    #
    def bestMove(self, nodes):
        return min(nodes, key=lambda x: x["fScore"])
    

# clamp function for capping min and max values
def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))



# ----------------------
import numpy as np # for matrix multiplication
import random

data = [
    ([0, 0, 0, 0], [0]),
    ([0, 0, 0, 1], [1]),
    ([0, 0, 1, 0], [0]),
    ([0, 0, 1, 1], [1]),
    ([0, 1, 0, 0], [0]),
    ([0, 1, 0, 1], [1]),
    ([0, 1, 1, 0], [0]),
    ([0, 1, 1, 1], [1]),
    ([1, 0, 0, 0], [1]),
    ([1, 0, 0, 1], [1]),
    ([1, 0, 1, 0], [1]),
    ([1, 0, 1, 1], [1]),
    ([1, 1, 0, 0], [0]),
    ([1, 1, 0, 1], [0]),
    ([1, 1, 1, 0], [0]),
    ([1, 1, 1, 1], [1])
]

# class definition adapted from:
# https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/#

class ANN():
  # network layer dimensions 
  INPUT_DIMENSION = 4
  HIDDEN_LAYER_DIMENSION = 4
  OUTPUT_DIMENSION = 1
  
  # network properties
  LEARNING_RATE = 0.1
  
  def __init__(self, input_size=INPUT_DIMENSION, hidden_size=HIDDEN_LAYER_DIMENSION, output_size=OUTPUT_DIMENSION):
    # define the network's layer dimensions
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    
    # initialize random weights as a matrix with proper dimensions
    self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
    self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
    
    # initialize biases for hidden layer and output layer
    self.bias_hidden = np.zeros((1, self.hidden_size))
    self.bias_output = np.zeros((1, self.output_size))
    
  
  """
  Sigmoid activation function. Takes one input and applies the sigmoid function to it.
  """
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  """
  Sigmoid Derivative
  """
  def sigmoid_derivative(self, x):
    return x * (1 - x)
  
  def feedforward(self, X):
    self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
    self.hidden_output = self.sigmoid(self.hidden_activation)
    
    self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
    self.predicted_output = self.sigmoid(self.output_activation)
    
    return self.predicted_output
  
  def backward(self, x, y, learning_rate):
    x = np.array([x]) # convert X to a 2D array from matrix multiplication purposes
    output_error = y - self.predicted_output
    output_delta = output_error * self.sigmoid_derivative(self.predicted_output)
    # print(output_delta, output_delta.shape)

    hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
    hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
    # print(hidden_delta, hidden_delta.shape)

    self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
    self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    # print(X)
    # print(X.shape)
    # print(X.T.shape, hidden_delta.shape)
    self.weights_input_hidden += np.dot(x.T, hidden_delta) * learning_rate
    self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
  
  def train(self, X, Y, epochs, learning_rate):
    best_error = float('inf')
    patience = 500
    min_improvement = 0.0005
    epochs_without_improvement = 0
    
    for epoch in range(epochs+1):
      losses = []
      for x, y in zip(X, Y):
        y_hat = self.feedforward(x)
        losses.append(abs(y - y_hat))
        self.backward(x, y, learning_rate)
      
      avg_loss = sum(losses) / len(losses)
      if isinstance(avg_loss, np.ndarray):
        avg_loss = avg_loss.item()
      
      if avg_loss < best_error - min_improvement:
        best_error = avg_loss
        epochs_without_improvement = 0
      else:
        epochs_without_improvement += 1
      
      if epoch % 100 == 0:
        print("Epoch:\t", epoch, "Error:\t", avg_loss)
      
      if avg_loss < 0.02:
        print(f"Stopped early: error < 0.02 at epoch {epoch}")
        break
      
      if epochs_without_improvement >= patience:
        print(f"Stopped early: no improvement for {patience} epochs (best: {best_error:.4f})")
        break
      
  
def main():
  nn = ANN()
  
  # randomly select 10 data points to train on
  train_set = random.sample(data, 10)
  
  # separate the training data into inputs and outputs
  X = np.array([inputs[0] for inputs in train_set])
  Y = np.array([output[1] for output in train_set])
  
  # train the nueral network
  nn.train(X, Y, epochs=50000, learning_rate=0.1)
  print("Training Complete.")
  
  
if __name__ == "__main__":
  main()






  #------------------------------------------------------------------------------------------------


        

    