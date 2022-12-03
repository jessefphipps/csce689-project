#%%
import sys
sys.path.append('/home/bryant/csce689-project/src/env')
sys.path.append("/home/bryant/csce689-project/src/model")
print(sys.path)
from model import Engine689, EngineHelpers
from mcts import RandomEngineMCTSNode, MCTS
import numpy as np
import math
import chess
import chess.svg
import tensorflow as tf
import random
import matplotlib.pyplot as plt


def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

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
            score = ucb_score(self, child)
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


class MCTS_Model(MCTS):
    "Monte Carlo Tree Search with model"

    def __init__(self, engine, n_sim):
        super().__init__()
        self.engine = engine
        self.n_sim = n_sim
    
    def run(self, board, to_play):
        root = Node(0, to_play)

        # EXPAND root
        chosen_move, action_probs, value, valid_model_output_indices = self.engine.make_move_util(board)
        if np.isnan(action_probs).any():
            action_probs = np.array([1/4184]*4184)
        action_probs = action_probs * np.array([1 if i in valid_model_output_indices else 0 for i in range(4184)])
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
                action_probs = action_probs * np.array([1 if i in valid_model_output_indices else 0 for i in range(4184)])
                if np.isnan(action_probs).any():
                    action_probs = np.array([1/4184]*4184)
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
        
        # action_probs = action_probs / np.sum(action_probs)
        # u_val = action_probs * np.sqrt(np.sum(action_probs ** (1/temperature))) / (1 + action_probs ** (1/temperature))

        # # _, _, value, _ = self.engine.make_move_util(board)
        # action = np.argmax(val + u_val)
        # print(action_probs)
        # print(val)
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
        self.stockfish = chess.engine.SimpleEngine.popen_uci("/home/bryant/csce689-project/engines/stockfish_15_win_x64_avx2/stockfish_15_x64_avx2.exe")
        self.stockfish.configure({"UCI_Elo": 1350})
        self.train_with_stockfish = True

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
                
            # _, _, value, _ = self.engine.make_move_util(board)

            action = None
            if self.train_with_stockfish:
                action = action_index
            else:
                action = root_node.select_action(1)

            engh = self.engine.helpers
            if current_player == 1: # white
                action_uci = engh.white_output_to_uci_mapping[action]
            else:
                action_uci = engh.black_output_to_uci_mapping[action]

            tmp_board = chess.Board(root_node.state.fen())
            tmp_board.push(chess.Move.from_uci(action_uci))
            next_board = tmp_board

            # _, _, next_value, _ = self.engine.make_move_util(board)
            # deltas = next_value - value
            # action_one_hot = np.zeros(4184)
            # action_one_hot[action] = 1
            # one_hot_delta = deltas * action_one_hot
            # one_hot_delta = action_probs * action_one_hot


            # add to training set
            train_examples.append((board, current_player, action_probs))
            # train_examples.append((board, current_player, one_hot_delta))
            
            board = next_board
            current_player *= -1
            reward = chess_reward(board, current_player)

            # limit run time
            if steps >= 100:
                reward = 0

            if reward is not None:
                ret = []
                for hist_board, hist_current_player, hist_action_probs in train_examples:
                    ret.append((hist_board, hist_action_probs, reward * ((-1) ** (hist_current_player != current_player))))
                
                return ret
            steps += 1
            print(f'At step {steps}')
            print(board)

    def learn(self):
        history = []
        for i in range(1, self.args['num_iterations']+1):
            train_examples = []

            for eps in range(self.args['num_episodes']):
                iteration_train_examples = self.execute_episode()
                train_examples.extend(iteration_train_examples)
            
            random.shuffle(train_examples)
            hist = self.train(train_examples)
            history.append(hist)
        self.engine.model.save('/home/bryant/csce689-project/models/full_model')
        return history
    
    def train(self, training):
        pi_losses = []
        v_losses = []

        boards = [i[0] for i in training]
        target_policy = [i[1] for i in training]
        target_value = [i[2] for i in training]

        board_fens = list(map(self.engine.helpers.generate_board_fen, boards))
        board_input = list(map(self.engine.fen_to_input, board_fens))

        hist = self.engine.model.fit(x=np.array(board_input),
                              y={'actor_output': np.array(target_policy), 'critic_output': np.array(target_value, dtype=float)},
                              epochs=self.args['epochs'], batch_size=self.args['batch_size'], verbose=0)
        
        self.engine.model.save('/home/bryant/csce689-project/models/prelim_model')

        return hist


#%%
if __name__ == "__main__":
    args = {
        'batch_size' : 64,
        'num_iterations' : 10,
        'num_simulations' : 50,
        'num_episodes' : 5,
        'epochs' : 2,
        'model_n_resid_layers' : 100,
        'model_n_resid_filter' : 256
    }
    trainer = Trainer(None, Engine689(args['model_n_resid_layers'], args['model_n_resid_filter']), args)
    hist = trainer.learn()
    # print(hist.history.keys())
    # fig, axs = plt.subplots(1, 2)
    # axs[0].plot(hist.history['actor_output_loss'])
    # axs[1].plot(hist.history['critic_output_loss'])
    plt.show()
    
# %%
