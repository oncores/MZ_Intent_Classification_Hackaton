import pandas as pd
import numpy as np
from collections import Counter


# In[44]:


import keras
from keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, Dropout, Concatenate, Flatten, Conv1D, GlobalMaxPooling1D, TimeDistributed, SpatialDropout1D, GRU, multiply, Lambda, Reshape, MaxPooling3D, Permute, RepeatVector, Multiply
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization


def main(all_feature, second_feature, third_feature):
    all_labels = pd.read_csv('./input/all_labels.csv', header=None)
    num_labels = len(all_labels)
    label_dict = {i[0]: idx for idx, i in enumerate(all_labels.values)}
    idx_to_label = {v: k for k, v in label_dict.items()}

    def mlp():
        K.clear_session()
        np.random.seed(9311)

        all_feature = Input(shape=(785), name='all_input')
        second_feature = Input(shape=(139), name='second_input')
        third_feature = Input(shape=(685), name='third_input')

        fc1 = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(all_feature)
        fc2 = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(second_feature)
        fc3 = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(third_feature)

        fc4 = Concatenate()([fc1,fc2,fc3])

        do = Dropout(0.1)(fc4)
        bn = BatchNormalization()(do)

        output = Dense(num_labels, activation='softmax')(bn)

        model = Model(inputs=[all_feature, second_feature, third_feature], outputs=output)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(0.00001),
                      metrics=['accuracy'])
        return model

    model_path = './ckpt/mlp_model.hdf5'

    model = mlp()
    model.load_weights(model_path)

    preds = model.predict([all_feature, second_feature, third_feature])
    preds = [np.argmax(pred) for pred in preds]

    label_preds = [idx_to_label[i] for i in preds]

    return label_preds
