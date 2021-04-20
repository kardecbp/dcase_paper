import numpy as np
from keras.callbacks import *
np.random.seed(5)

def apk(actual, predicted, k=3):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=3):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


class Mapk_callback(Callback):

	def __init__(self,training_data,validation_data):
		self.x_trn = training_data[0]
		self.y_trn = training_data[1]
		self.x_val = validation_data[0]
		self.y_val = validation_data[1]        

	def on_train_begin(self, logs={}):
		return
	def on_train_end(self, logs={}):
		return
	def on_epoch_begin(self, epoch, logs={}):
		return
	def on_epoch_end(self, epoch, logs={}): 
		if not epoch % 5:       
			y_pred_trn = self.model.predict(self.x_trn)
			y_pred_top3_classes_trn = [np.argsort(x).tolist()[::-1][0:3] for x in y_pred_trn]
			obs_y_trn = [[x] for x in self.y_trn.tolist()]
			mapk_score_trn = mapk(obs_y_trn, y_pred_top3_classes_trn,k=3)

			y_pred_val = self.model.predict(self.x_val)
			y_pred_top3_classes_val = [np.argsort(x).tolist()[::-1][0:3] for x in y_pred_val]
			obs_y_val = [[x] for x in self.y_val.tolist()]
			mapk_score_val = mapk(obs_y_val, y_pred_top3_classes_val, k=3)

			print('\rmapk: %s - mapk_val: %s' % (str(round(mapk_score_trn, 4)), str(round(mapk_score_val, 4))),end=100*' '+'\n')
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return