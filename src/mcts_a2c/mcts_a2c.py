#%%
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
sys.path.append('../env')
sys.path.append("../model")
from model import Engine689, EngineHelpers, actor_loss, critic_loss, RandomEngine
from mcts import RandomEngineMCTSNode, MCTS
import numpy as np
import math
import chess
import chess.svg
import tensorflow as tf
import random
# import matplotlib.pyplot as plt


def ucb_score(parent, child, c):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1e-6)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score*c


class Node:
    def __init__(self, prior, to_play, c=1):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.c = c

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action
    
    def select_action_util(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action, actions, visit_count_distribution

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child, self.c)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, to_play, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.to_play = to_play
        self.state = state
        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(prior=prob, to_play=self.to_play * -1)

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())


class MCTS_Model:
    "Monte Carlo Tree Search with model"

    def __init__(self, engine, n_sim):
        self.engine = engine
        self.n_sim = n_sim
    
    def run(self, board, to_play):
        root = Node(0, to_play)

        # EXPAND root
        chosen_move, action_probs, value, valid_model_output_indices = self.engine.make_move_util(board)
        if np.isnan(action_probs).any():
            action_probs = np.array([1/4184]*4184)
        action_probs = action_probs * np.array([1.0 if i in valid_model_output_indices else 0.0 for i in range(4184)])
        if not np.any(action_probs):
            action_probs = np.array([1.0 if i in valid_model_output_indices else 0.0 for i in range(4184)])
        action_probs /= np.sum(action_probs)
        root.expand(chess.Board(board.fen()), to_play, action_probs)

        for _ in range(self.n_sim):
            node = root
            search_path = [node]

            # select
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)
            
            parent = search_path[-2]
            state = parent.state

            # engh = EngineHelpers()
            if parent.to_play == 1: # white
                action_uci = self.engine.helpers.white_output_to_uci_mapping[action]
            else:
                action_uci = self.engine.helpers.black_output_to_uci_mapping[action]

            tmp_board = chess.Board(state.fen())
            tmp_board.push(chess.Move.from_uci(action_uci))
            next_board = tmp_board

            # get reward
            value = chess_reward(next_board, player=parent.to_play)

            if value is None:
                # if game not ended
                # expand
                chosen_move, action_probs, value, valid_model_output_indices = self.engine.make_move_util(next_board)
                value = value[0]
                action_probs = action_probs * np.array([1.0 if i in valid_model_output_indices else 0.0 for i in range(4184)])
                if not np.any(action_probs):
                    action_probs = np.array([1.0 if i in valid_model_output_indices else 0.0 for i in range(4184)]) / sum(np.array([1.0 if i in valid_model_output_indices else 0.0 for i in range(4184)]))
                action_probs /= np.sum(action_probs)
                node.expand(next_board, parent.to_play*-1, action_probs)
            self.backpropagate(search_path, value, parent.to_play*-1)
        return root

    def backpropagate(self, search_path, value, to_play):
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1


def chess_reward(board, player):
    if board.is_game_over():
        if(board.result()=='1-0'):  # white won
            result = 1
        elif(board.result()=='0-1'): #black won
            result = -1
        else:
            # return 0.5
            return 0
        return player * result
    return None

        
class MCTS_Agent:
    def __init__(self, engine, args):
        self.engine = engine
        self.args = args
        self.mcts = MCTS_Model(engine=self.engine, n_sim=self.args['num_simulations'])

    def make_move(self, board, player, temperature=0):
        root_node = self.mcts.run(board, to_play=player)

        action_probs = [0] * 4184
        val = [0] * 4184
        for k, v in root_node.children.items():
            action_probs[k] = v.visit_count
            val[k] = v.value_sum
        
        action = root_node.select_action(0)

        engh = self.engine.helpers
        if player == 1: # white
            action_uci = engh.white_output_to_uci_mapping[action]
        else:
            action_uci = engh.black_output_to_uci_mapping[action]
        
        return chess.Move.from_uci(action_uci)


class Trainer:
    def __init__(self, game, engine, args):
        self.game = game
        self.engine = engine
        self.args = args
        self.mcts = MCTS_Model(engine=self.engine, n_sim=self.args['num_simulations'])
        self.stockfish = chess.engine.SimpleEngine.popen_uci("engines/stockfish_15_win_x64_avx2/stockfish_15_x64_avx2.exe")
        self.stockfish.configure({"UCI_Elo": 1350})
        self.train_with_stockfish = self.args['train_with_stockfish']

    def execute_episode(self):
        train_examples = []
        current_player = 1  #1 for white, -1 for black
        board = chess.Board()   # new board

        steps = 0
        while True:
            self.mcts = MCTS_Model(engine=self.engine, n_sim=self.args['num_simulations'])
            root_node = self.mcts.run(board, to_play=current_player)

            action_probs = [0] * 4184
            for k, v in root_node.children.items():
                action_probs[k] = v.visit_count
            
            action_probs = action_probs / np.sum(action_probs)
            if self.train_with_stockfish:
                # change action_probs to be action stockfish takes
                result = self.stockfish.play(board, chess.engine.Limit(time=0.01))
                action_probs_tmp = np.zeros(4184)
                if current_player == 1:
                    action_index = self.engine.helpers.white_uci_to_output_mapping[str(result.move)]
                else:
                    action_index = self.engine.helpers.black_uci_to_output_mapping[str(result.move)]
                action_probs_tmp[action_index] = 1
                action_probs *= action_probs_tmp   # hot encode original action_probs

            action = None
            if self.train_with_stockfish:
                action = action_index
            else:
                action = root_node.select_action(0)

            engh = self.engine.helpers
            if current_player == 1: # white
                action_uci = engh.white_output_to_uci_mapping[action]
            else:
                action_uci = engh.black_output_to_uci_mapping[action]

            tmp_board = chess.Board(root_node.state.fen())
            tmp_board.push(chess.Move.from_uci(action_uci))
            next_board = tmp_board

            # add to training set
            train_examples.append((board.fen(), current_player, action_probs))
            # train_examples.append((board, current_player, one_hot_delta))
            
            board = next_board
            current_player *= -1
            reward = chess_reward(board, current_player)

            # limit run time
            if steps >= self.args['max_moves']:
                reward = 0

            if reward is not None:
                ret = []
                for hist_board, hist_current_player, hist_action_probs in train_examples:
                    ret.append((hist_board, hist_action_probs, reward * ((-1) ** (hist_current_player != current_player))))
                
                return ret
            steps += 1
            print(f'At step {steps}                ', end='\r')
            print(board)

            # experimental
            del root_node
            del self.mcts
            
    def random_metric(self):
        random_engine = RandomEngine()
        random_board = chess.Board()
        num_moves = 0
        while True:  
            # trained engine move
            # move = engine689.make_move(board)
            move = self.engine.make_move(random_board)   # 1 white, -1 black
            random_board.push(move)
            num_moves += 1
            if random_board.is_game_over():
                break

            # random engine move
            move = random_engine.make_move(random_board)
            random_board.push(move)
            if random_board.is_game_over():
                break
        
        return random_board.fen(), num_moves, random_board.result()

    def learn(self):
        history = []
        success_metric = []
        for i in range(1, self.args['num_iterations']+1):
            train_examples = []

            for eps in range(self.args['num_episodes']):
                print(f'training iteration {i} episode {eps}')
                iteration_train_examples = self.execute_episode()
                train_examples.extend(iteration_train_examples)
                print()
            
            random.shuffle(train_examples)
            hist = self.train(train_examples)
            history.append(hist)
            success_metric.append(self.random_metric())
        self.engine.model.save('models/full_model4')
        return history, success_metric
    
    def train(self, training):
        pi_losses = []
        v_losses = []

        boards = [i[0] for i in training]
        target_policy = [i[1] for i in training]
        target_value = [i[2] for i in training]

        board_fens = boards
        board_input = list(map(self.engine.fen_to_input, board_fens))

        hist = self.engine.model.fit(x=np.array(board_input),
                              y={'actor_output': np.array(target_policy), 'critic_output': np.array(target_value, dtype=float)},
                              epochs=self.args['epochs'], batch_size=self.args['batch_size'], verbose=0)
        
        self.engine.model.save('models/prelim_model')

        return hist


#%%
if __name__ == "__main__":
    args = {
        'batch_size' : 64,
        'num_iterations' : 20,
        'num_simulations' : 50,
        'num_episodes' : 5,
        'epochs' : 2,
        'model_n_resid_layers' : 50,
        'model_n_resid_filter' : 256,
        'max_moves' : 125,
        'train_with_stockfish' : False
    }
    continue_training = False
    trainer = None
    if continue_training:
        engine = Engine689(args['model_n_resid_layers'], args['model_n_resid_filter'])
        load_saved_model = tf.keras.models.load_model('models/prelim_model', custom_objects={'loss': {'actor_output_loss' : actor_loss(), 'critic_output_loss' : critic_loss()}})
        engine.model = load_saved_model
        trainer = Trainer(None, engine, args)

    else:
        trainer = Trainer(None, Engine689(args['model_n_resid_layers'], args['model_n_resid_filter']), args)
        
    hist, success_metric = trainer.learn()
    np.save("Model_4_Results", success_metric, allow_pickle=True)
