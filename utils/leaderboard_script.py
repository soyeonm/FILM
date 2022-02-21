import pickle
import argparse
import json
from glob import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dn_startswith', type=str, required=True)
parser.add_argument('--seen', action='store_true')
parser.add_argument('--json_name', type=str)

args = parser.parse_args()

if args.json_name is None:
	args.json_name = args.dn

seen_str = 'seen'
if not(args.seen):
	seen_str= 'unseen'

pickle_globs = glob("results/leaderboard/actseqs_test_" + seen_str + "_" + args.dn_startswith + "*")

pickles = []
for g in pickle_globs:
	pickles += pickle.load(open(g, 'rb'))

total_logs =[]
for i, t in enumerate(pickles):
	key = list(t.keys())[0]
	actions = t[key]
	trial = key[1]
	total_logs.append({trial:actions})

for i, t in enumerate(total_logs):
	key = list(t.keys())[0]
	actions = t[key]
	new_actions = []
	for action in actions:
		if action['action'] == 'LookDown_0' or action['action'] == 'LookUp_0':
			pass
		else:
			new_actions.append(action)
	total_logs[i] = {key: new_actions}

if args.seen:
	assert len(total_logs) == 1533
	results = {'tests_unseen':[{'trial_T20190908_010002_441405': [{'action': 'LookDown_15', 'forceAction': True}]}], 'tests_seen': total_logs}
else:
	assert len(total_logs) == 1529
	results = {'tests_seen':[{'trial_T20190909_042500_949430': [{'action': 'LookDown_15', 'forceAction': True}]}], 'tests_unseen': total_logs}

if not os.path.exists('leaderboard_jsons'):
	os.makedirs('leaderboard_jsons')

save_path = 'leaderboard_jsons/tests_actseqs_' + args.json_name + '.json'
with open(save_path, 'w') as r:
	json.dump(results, r, indent=4, sort_keys=True)


