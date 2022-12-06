#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
sys.path.append('src/env')
sys.path.append("src/model")
sys.path.append('src/mcts_a2c')
from model import Engine689, EngineHelpers, actor_loss, critic_loss, RandomEngine
from mcts import RandomEngineMCTSNode, MCTS
from mcts_a2c import *
import numpy as np
import math
import chess
import chess.svg
import tensorflow as tf
import random

if __name__ == "__main__":
    args = {
        'batch_size' : 64,
        'num_iterations' : 10,
        'num_simulations' : 100,
        'num_episodes' : 2,
        'epochs' : 2,
        'model_n_resid_layers' : 10,
        'model_n_resid_filter' : 128,
        'max_moves' : 200,
        'train_with_stockfish' : False
    }

    if 'models' not in os.listdir():
        os.mkdir('models')
    
    continue_training = False
    print(tf.config.list_physical_devices('GPU'))
    # tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    trainer = None
    if continue_training:
        engine = Engine689(args['model_n_resid_layers'], args['model_n_resid_filter'])
        load_saved_model = tf.keras.models.load_model('../../models/prelim_model10', custom_objects={'loss': {'actor_output_loss' : actor_loss(), 'critic_output_loss' : critic_loss()}})
        engine.model = load_saved_model
        trainer = Trainer(None, engine, args)

    else:
        trainer = Trainer(None, Engine689(args['model_n_resid_layers'], args['model_n_resid_filter']), args)
        
    hist, success_metric = trainer.learn()