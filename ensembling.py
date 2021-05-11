import numpy as np 
np.random.seed(1001)
import pandas as pd 
from keras.utils import to_categorical
from metrics import mapk
from sklearn.metrics import log_loss
import os
import time
import pickle
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, 'data')
train_path = os.path.join(data_dir, 'audio_train')
test_path = os.path.join(data_dir, 'audio_test')
PREDICTION_FOLDER = os.path.join(dir_path, 'predictions', 'ensembles')

train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
train["label_idx"] = train.label.apply(lambda x: label_idx[x])
y = list(train['label_idx'])

train_split = pickle.load(open(os.path.join(dir_path, 'splits/train_inds.pickle'), 'rb'))
val_split = pickle.load(open(os.path.join(dir_path, 'splits/val_inds.pickle'), 'rb'))

def stack_preds(y, prediction_name=None, to_drop=[], save_test=False):
	'''
	function responsible for stacking predictions
	y - labels
	prediction_name - name of the generated predictions file (only relevenat if save_test=True)
	to_drop - file_names that won't be used in the ensemble
	save_test - whether to save predictions (when True) or just display results on validation data (when False)
	'''
	pred_path = os.path.join(dir_path, 'predictions', 'solution')
	train_preds, val_preds, test_preds = [], [], []
	
	model_names, test_names = [], []
	for file in  os.listdir(pred_path):
		if file not in to_drop:
			print(file)
			pred = pickle.load(open(os.path.join(pred_path, file), 'rb'))
			if 'train' in file:
				train_preds.append(np.array(pred))
			elif 'val' in file:
				val_preds.append(np.array(pred))
				model_names.append(file)
			else:
				test_preds.append(np.array(pred)), test_names.append(file)
	
	features, features_test_main = np.column_stack(val_preds), np.column_stack(test_preds)
	target = [y[i] for i in range(len(y)) if i in val_split]
	target_one_hot = to_categorical(target)

	skf = StratifiedKFold(n_splits=5, random_state=5)
	preds_one_hot, preds_test = np.empty(target_one_hot.shape), []
	for train_split_tmp, val_split_tmp in skf.split(features, target):
		features_train, features_val = features[train_split_tmp], features[val_split_tmp]
		target_train = [target[i] for i in range(len(target)) if i in train_split_tmp]
		target_val = [target[i] for i in range(len(target)) if i in val_split_tmp]
		target_train_one_hot, target_val_one_hot = target_one_hot[train_split_tmp], target_one_hot[val_split_tmp]
		features_test = features_test_main
		
		pca = PCA(n_components=120)
		pca.fit(features_train)
		features_train, features_val = pca.transform(features_train), pca.transform(features_val) 
		features_test = pca.transform(features_test)
		clf = LogisticRegression(C=4) 
		clf.fit(features_train, target_train)
		
		preds_train = clf.predict_proba(features_train)
		preds_val = clf.predict_proba(features_val)
		preds_one_hot[val_split_tmp] = preds_val
		preds_test_tmp = clf.predict_proba(features_test)
		preds_test.append(preds_test_tmp)
		print(log_loss(target_train_one_hot, preds_train))
		print(log_loss(target_val_one_hot, preds_val))
		preds_top3_train = [list(np.argsort(preds_train[i])[::-1][:3]) for i in range(len(preds_train))]
		preds_top3_val = [list(np.argsort(preds_val[i])[::-1][:3]) for i in range(len(preds_val))]
		actual_train = [[label] for label in target_train]
		actual_val = [[label] for label in target_val]
		train_score = mapk(actual_train, preds_top3_train, k=3)
		val_score = mapk(actual_val, preds_top3_val, k=3)
		print(train_score, val_score)
		print('---------------')

	preds_top3_val = [list(np.argsort(preds_one_hot[i])[::-1][:3]) for i in range(len(preds_one_hot))]
	actual_target = [[label] for label in target]
	val_score = mapk(actual_target, preds_top3_val, k=3)
	print(val_score)

	if save_test:
		pickle.dump(preds_one_hot, open(os.path.join(PREDICTION_FOLDER, prediction_name + ".pickle"), 'wb'))
		avrg_test = np.mean(np.array(preds_test), axis=0)
		pickle.dump(avrg_test, open(os.path.join(PREDICTION_FOLDER, prediction_name + ".pickle"), 'wb'))
		top_3 = np.array(LABELS)[np.argsort(-avrg_test, axis=1)[:, :3]]
		predicted_labels = [' '.join(list(x)) for x in top_3]
		test_new = pd.DataFrame()
		test_new['label'] = predicted_labels
		test_new['fname'] = test['fname']
		test_new[['fname', 'label']].to_csv(os.path.join(PREDICTION_FOLDER, prediction_name + ".csv"), index=False)


if __name__ == '__main__':
	stack_preds(y, save_test=True, prediction_name='stacking_final')