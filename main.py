import numpy as np
from ReinforceAgent import Agent
from flask import Flask, request, jsonify, make_response
import json
import pickle
import logging
import time
import os

app = Flask(__name__)

logging.basicConfig(format='%(asctime)s %(message)s')

#Creating a logger object for MAIN API
logger = logging.getLogger('MM-API-v4')
logger.setLevel(logging.INFO)

pieces = 4
players = 2

start = [1, 27]
inflect = [51, 25]
new_start = [51, 63]
end = [57, 69]

agent0 = Agent(17, 4)
if os.path.exists('../saved_models/current_model.hdf5'):
    agent0.load('../saved_models/current_model.hdf5')
else:
    agent0.load_model('./data_model_84.hdf5')

data = {}


def find_possible_moves(state, p):

    action_list = []

    for j in range(0, pieces):

        old_pos = state[1 + p * pieces + j]

        if old_pos != end[p]:
            new_pos = (old_pos + state[0])

            if new_pos > inflect[p] and old_pos <= inflect[p]:
                new_pos = (new_pos - inflect[p]) + new_start[p]
            elif old_pos < 52 and new_pos >= 52 and p != 0:
                new_pos = new_pos % 52

            if new_pos <= end[p]:
                action_list.append(j)

    return action_list


def cycle_viewpoint(s):
    return [*[s[0]], *s[5:9], *s[1:5], *s[13:17], *s[9:13], s[18], s[17]]


def convert_state(p, s):
    if p != 0:
        s = cycle_viewpoint(s)
    return s


@app.route('/rl_bot_act', methods = ['GET', 'POST'])
def api():

    start_time = time.time()

    inp = request.data
    inp = json.loads(inp)

    state = [inp["diceRoll"]] + inp["pawnStates"][0]["pos"] + inp["pawnStates"][1]["pos"] + inp["pawnScores"][0]["score"] + inp["pawnScores"][1]["score"]
    p = inp["botPlayerIdx"]
    action_list = find_possible_moves(state, p)
    state = convert_state(p, state)
    action = agent0.act_test(state, action_list)

    response = make_response(jsonify({'action': int(action)}))

    end_time = time.time()

    logger.info("State: %s", str(np.array(state)))
    logger.info("Action %d", action)
    logger.info("Time taken: %f", (end_time-start_time)*1000)

    if inp["game_id"] not in data:
        globals()["data"][inp["game_id"]] = {}
        globals()["data"][inp["game_id"]]["states"] = []
        globals()["data"][inp["game_id"]]["actions"] = []

    globals()["data"][inp["game_id"]]["states"].append(state)
    globals()["data"][inp["game_id"]]["actions"].append(action)

    if inp["game_over"]:
        with open('../train_data/' + inp["game_id"] + '.pkl', 'wb') as f:
            pickle.dump(globals()["data"][inp["game_id"]], f)
        del globals()["data"][inp["game_id"]]
        logger.info("Game data saved for game_id: %s", inp["game_id"])

    return response


@app.route('/load_model', methods = ['GET', 'POST'])
def load_model():
    inp = request.data
    globals()['agent0'].load_model(inp['path'])
    logger.info("Model loaded")
    return make_response(jsonify({"msg":"got data","isSuccess":True,"status":200}),200)


@app.route('/ping', methods=['GET'])
def ping():
    if request.method == 'GET':
        return make_response(jsonify({"msg":"got data","isSuccess":True,"status":200}),200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, threaded=True, debug=False)