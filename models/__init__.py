from .cnn import CnnModel1D, CnnModel2D, CnnModelRawData
from .crnn import CrnnModel
from .svm import SvmModel
from .bilstm_attention import BilstmAttention
from .lstm_attention import LstmAttention
from .logistic_regression import LogisticRegression
from .cnn_features import CnnFeatures

# MODEL NAME
CNN_MODEL_1D = 'cnn_1d'
CNN_MODEL_2D = 'cnn_2d'
CNN_MODEL_RAW_DATA = 'cnn_raw'
CNN_FEATURES = 'cnn_features'
CRNN_MODEL = 'crnn'
SVM_MODEL = 'svm'
BILSTM_MODEL = 'bilstm'
LSTM_MODEL = 'lstm'
LR_MODEL = 'lr'

