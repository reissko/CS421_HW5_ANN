import random
import heapq
import itertools
import sys
import os
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
        super(AIPlayer,self).__init__(inputPlayerId, "Neural Network Agent")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_file = os.path.join(base_dir, "ann_data.npz")
        self.MAX_DATASET_SIZE = 5000
        self.games_trained = 0
        
        if os.path.exists(self.data_file):
            try:
                data = np.load(self.data_file)
                X_all = data['X_all']
                Y_all = data['Y_all']
                self.training_data = [
                    [X_all[i].reshape(3, 1), np.array([[Y_all[i]]])]
                    for i in range(len(X_all))
                ]
                if len(self.training_data) > self.MAX_DATASET_SIZE:
                    self.training_data = self.training_data[-self.MAX_DATASET_SIZE:]
                    print(f"Loaded dataset limited to {self.MAX_DATASET_SIZE} examples")
                
                self.ann = ANN()
                self.ann.weights_input_hidden = data['weights_input_hidden']
                self.ann.weights_hidden_output = data['weights_hidden_output']
                self.ann.bias_hidden = data['bias_hidden']
                self.ann.bias_output = data['bias_output']
                
                if 'games_trained' in data:
                    self.games_trained = int(data['games_trained'].item())
                
                print(f"Loaded {len(self.training_data)} examples and trained ANN (trained on {self.games_trained} games)")
            except Exception as e:
                print(f"Error loading data: {e}. Starting fresh.")
                self.training_data = []
                self.ann = ANN()
        else:
            self.training_data = []
            self.ann = ANN()
    
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
        currentUtility = self.utility(currentState, currentState)
        currentScores = self.scores_to_nn_input(self.compute_unit_composition_score(currentState), self.compute_route_score(currentState), self.compute_rsoldier_aggression_score(currentState))
        
        # X is the input (3x1), Y is the output (1x1)
        X = currentScores  # 3x1 array
        Y = np.array([[currentUtility]])  # 1x1 array
        training_example = [X, Y]  # array of arrays containing both
        
        # Add to persistent training data list
        self.training_data.append(training_example)
        
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
        if len(self.training_data) < 10:
            return
        
        if len(self.training_data) > self.MAX_DATASET_SIZE:
            self.training_data = self.training_data[-self.MAX_DATASET_SIZE:]
            print(f"Dataset limited to {self.MAX_DATASET_SIZE} examples")
        
        shuffled_data = random.sample(self.training_data, len(self.training_data))
        split_index = int(len(shuffled_data) * 0.8)
        train_data = shuffled_data[:split_index]
        test_data = shuffled_data[split_index:]

        X_train = np.array([ex[0].flatten() for ex in train_data])
        Y_train = np.array([ex[1].item() for ex in train_data])
        X_test = np.array([ex[0].flatten() for ex in test_data])
        Y_test = np.array([ex[1].item() for ex in test_data])
        
        self.ann.train(X_train, Y_train, epochs=10000, learning_rate=0.1)
        test_predictions = [self.ann.feedforward(x) for x in X_test]
        test_errors = [abs(y_true - y_pred.item()) for y_true, y_pred in zip(Y_test, test_predictions)]
        avg_test_error = sum(test_errors) / len(test_errors)
        
        self.games_trained += 1
        print(f"Test error: {avg_test_error:.4f} (trained on {len(train_data)}, tested on {len(test_data)}) - Game {self.games_trained}")
        
        self._save_data()
    
    def _save_data(self):
        if len(self.training_data) == 0:
            return
        
        try:
            X_all = np.array([ex[0].flatten() for ex in self.training_data])
            Y_all = np.array([ex[1].item() for ex in self.training_data])
            
            np.savez(
                self.data_file,
                X_all=X_all,
                Y_all=Y_all,
                weights_input_hidden=self.ann.weights_input_hidden,
                weights_hidden_output=self.ann.weights_hidden_output,
                bias_hidden=self.ann.bias_hidden,
                bias_output=self.ann.bias_output,
                num_examples=np.array([len(self.training_data)]),
                games_trained=np.array([self.games_trained])
            )
            print(f"Saved {len(self.training_data)} examples and ANN")
        except Exception as e:
            print(f"Error saving data: {e}")

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
        food_score = min(next_inv.foodCount / float(FOOD_GOAL), 1.0)
        
        # Route efficiency (already in [0,1])
        route_score = self.compute_route_score(nextState)
        normalized_route = route_score  # already in [0,1] range
        
        # Unit composition [0-1]]
        unit_score = self.compute_unit_composition_score(nextState)
        normalized_units = max(0, min(1, unit_score)) 
        
        # 4. Military aggression (normalize to 0-1)
        aggro_score = self.compute_rsoldier_aggression_score(nextState)
        normalized_aggro = max(0, min(1, aggro_score))  
        
        # Weights (should sum to 1.0 for proper scaling)
        food_w = 0.6     # Food is most important for winning
        route_w = 0.1     # efficiency 
        units_w = 0.1     # more ants the better
        aggro_w = 0.2    # Military presence
        
        # Weighted combination (results in 0-1 scale)
        base_utility = (food_w * food_score + 
                    route_w * normalized_route + 
                    units_w * normalized_units + 
                    aggro_w * normalized_aggro)
        
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
    # Maps the outputs from the three score computation functions to a 3x1 vector
    # suitable for neural network input
    #
    # Parameters:
    #   unit_score - output from compute_unit_composition_score [0, 1]
    #   route_score - output from compute_route_score [0, 1]
    #   aggro_score - output from compute_rsoldier_aggression_score [0, 1]
    #
    # Returns:
    #   numpy array of shape (3, 1) containing the three scores
    #
    def scores_to_nn_input(self, unit_score, route_score, aggro_score):
        return np.array([[unit_score], [route_score], [aggro_score]])

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
  INPUT_DIMENSION = 3
  HIDDEN_LAYER_DIMENSION = 3
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
      
      if avg_loss < 0.05:
        print(f"Stopped early: error < 0.05 at epoch {epoch}")
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


        

    