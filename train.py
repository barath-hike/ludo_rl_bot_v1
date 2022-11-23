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

    files = glob.glob("../train_data/*.pkl")

    if len(files) > 1:

        for file in files:

            agent0 = Agent(17, 4)
            agent0.load_model('../saved_models/current_model.hdf5')

            with open(file, 'rb') as f:
                data = pickle.load(f)

            states = []
            actions = []
            rewards = []

            for i, s_ in enumerate(data['states']):
                states.append(np.reshape(agent0.preprocess(s_), [1, 17]))
                actions.append(data['actions'][i])
                rewards.append(np.sum(s_[9:13]) - np.sum(s_[13:17]))
                if i > 0:
                    rewards[i] -= rewards[i-1]

            states = states[:-1]
            actions = actions[:-1]
            rewards = rewards[1:]

            batch[0].extend(states)
            batch[1].extend(actions)
            rew_path.append(rewards)

            os.system("mv " + file + " ../trained_data/")

        agent0.learn(batch, agent0.get_returns(rew_path))

        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")

        agent0.save_model("../saved_models/model_" + dt_string + ".hdf5")
        agent0.save_model('../saved_models/current_model.hdf5')

        requests.post('api', json={'path': '../saved_models/current_model.hdf5'})

if __name__ == '__main__':
    train()