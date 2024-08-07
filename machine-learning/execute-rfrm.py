import numpy as np
import seaborn as sns
from joblib import load


sns.set_theme()
sns.set_style('ticks')

X = np.load("new-x-matrix.npy")


model = load('rf_model_for_constants.pkl')

# Load the scaler from the file
Xscaler = load("rf_Xscaler.joblib")
yscaler = load("rf_yscaler.joblib")
'''
model = load_model('model_for_constants')

# Load the scaler from the file
Xscaler = load("Xscaler.joblib")
yscaler = load("yscaler.joblib")
'''

X = Xscaler.transform(X)
print(X.shape)
predict = yscaler.inverse_transform(model.predict(X))
print(predict.shape)
np.save("new-y-matrix.npy", predict)
