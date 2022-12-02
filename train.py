import os
import numpy as np
from ReinforceAgent import Agent
import glob
import pickle
from datetime import datetime
import requests

def train():

    batch = [[], []]
    rew_path = []
    rew_path1 = []

    files = glob.glob("../ludo_rl_bot_v1_data/train_data/*.pkl")

    if len(files) > 1:

        for file in files:

            with open(file, 'rb') as f:
                data = pickle.load(f)

            states = []
            actions = []
            rewards = []
            rewards1 = []

            for i, s_ in enumerate(data['states']):
                states.append(np.reshape(agent0.preprocess(s_), [1, 17]))
                actions.append(data['actions'][i])
                rewards.append(np.sum(s_[9:13]))
                rewards1.append(np.sum(s_[9:13]) - np.sum(s_[13:17]))
                if i > 0:
                    rewards[i] -= rewards[i-1]
                    rewards1[i] -= rewards1[i-1]

            states = states[:-1]
            actions = actions[:-1]
            rewards = rewards[1:]
            rewards1 = rewards1[1:]

            batch[0].extend(states)
            batch[1].extend(actions)
            rew_path.append(rewards)
            rew_path1.append(rewards1)

            os.system("mv " + file + " ../ludo_rl_bot_v1_data/trained_data/")

        agent0 = Agent(17, 4)
        agent1 = Agent(17, 4)

        if os.path.exists('../ludo_rl_bot_v1_data/saved_models/v1/current_model.hdf5'):
            agent0.load('../ludo_rl_bot_v1_data/saved_models/v1/current_model.hdf5')
        else:
            agent0.load_model('./data_model_84.hdf5')

        if os.path.exists('../ludo_rl_bot_v1_data/saved_models/v2/current_model.hdf5'):
            agent1.load('../ludo_rl_bot_v1_data/saved_models/v2/current_model.hdf5')
        else:
            agent1.load_model('./data_model_84.hdf5')

        agent0.learn(batch, agent0.get_returns(rew_path))
        agent1.learn(batch, agent1.get_returns(rew_path1))

        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")

        agent0.save_model("../ludo_rl_bot_v1_data/saved_models/v1/model_" + dt_string + ".hdf5")
        agent0.save_model('../ludo_rl_bot_v1_data/saved_models/v1/current_model.hdf5')

        agent1.save_model("../ludo_rl_bot_v1_data/saved_models/v2/model_" + dt_string + ".hdf5")
        agent1.save_model('../ludo_rl_bot_v1_data/saved_models/v2/current_model.hdf5')

        requests.post("http://ludo-rl-bot-v1.rgu.network:3000/load_model", json={'path': '../ludo_rl_bot_v1_data/saved_models/v1/current_model.hdf5'})

if __name__ == '__main__':
    train()