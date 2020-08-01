import pandas as pd
import numpy as np
import glob

import data.normalise as nm
from data.duplicates import group_duplicates
from data.features import compute_features

nw_features_disc = {
	'Time': {
		'func': nm.change_time,
		'input': 'time'
	},
	'Date': {
		'func': nm.has_dates,
		'input': 'message'
	},
	'Number': {
		'func': nm.has_numbers,
		'input': 'message'
	},
	'Decimal': {
		'func': nm.has_decimals,
		'input': 'message'
	},
	'URL': {
		'func': nm.has_urls,
		'input': 'message'
	}
}


if __name__ == '__main__':
	all_files = glob.glob('data/raw_db' + '/*.csv')

	li = []
	for filename in all_files:
		df = pd.read_csv(filename)
		df.columns = ['sender', 'time', 3, 'message']
		li.append(df)

	data = pd.concat(li, axis=0)
	sms = data.iloc[:, [0, 1, 3]]
	sms = sms.drop(sms.columns[0], axis=1)  # doubt

	sms['message'] = sms['message'].apply(nm.remove_newlines)

	human = pd.read_csv('data/pruned_db/old_human.csv')
	human.columns = ['index', 'sender', 'message', 'label']
	human = pd.concat(human, [sms.iloc[:, [0, 1, 2]]]).reset_index()

	nw_features = pd.DataFrame(index=range(len(human)), columns=[feat for feat in nw_features_disc])

	for feature in nw_features_disc:
		disc = nw_features_disc[feature]
		nw_features[feature] = human[disc['input']].astype(str).apply(disc['func'])

	nw_features = nw_features.to_numpy()

	words, w_features = compute_features(human['Message'].to_numpy().astype(str), compute=True)
	number_words = nm.number_words(w_features)
	nw_features = np.append(nw_features, number_words, axis=1)

	data = np.append(nw_features, w_features, axis=1)

	human.to_csv("data/pruned_db/human.csv")

	np.savetxt("data/pruned_db/words.csv", words, fmt='%s', encoding='utf8')
	np.savetxt("data/pruned_db/unlabeled.csv", data, delimiter=",")

	group_duplicates(w_features)
