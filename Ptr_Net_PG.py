import argparse

import gym
import numpy as np
import tensorflow as tf

import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false')
args = parser.parse_args()

###########################  hyper parameters  ###########################

ENV_NAME = 'PtrNet_PG-v0'   # define the environment
RANDOMSEED = 754            # set the random seed to reappear the result

##################################  PG  ##################################

class PolicyGradient:
    """
    PG class
    """
    
    def __init__(self, num_features, num_actions, learning_rate=0.01, reward_decay=0.95):
        # define related parameters
        self.num_features = num_features
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        
        # save data from each episode
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        
        def get_first_layer_model(inputs_shape):
            """
            Generate a Recurrent Nerual Network using LSTM cell

            Args:
                inputs_shape (nD_tensor): to describe the shape of input data
            """
            # with tf.name_scope('inputs'):
            #     self.tf_obs = tl.layers.Input(inputs_shape, tf.float32, name="observations")
            #     self.tf_acts = tl.layers.Input([None,], tf.int32, name="actions_num")
            #     self.tf_vt = tl.layers.Input([None,], tf.float32, name="actions_value")
            
        def get_first_layer_params():
            return (10, 9)