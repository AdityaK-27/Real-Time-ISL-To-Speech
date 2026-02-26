import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Layer
import tensorflow.keras.backend as K

import os

base_dir = os.path.dirname(os.path.abspath(__file__))
keypoint_path = os.path.join(base_dir, 'keypoint_data')
actions = os.listdir(keypoint_path)



def lstm_v1(device_name):

    with tf.device(device_name):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 150)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(256, return_sequences=False, activation='relu'))

        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(actions), activation='softmax'))

    return model


def lstm_v2(device_name):

    with tf.device(device_name):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 150)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(256, return_sequences=True, activation='relu'))
        model.add(LSTM(256, return_sequences=False, activation='relu'))

        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(actions), activation='softmax'))

    return model

def lstm_v3(device_name):

    with tf.device(device_name):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(30, 150)))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(128, return_sequences=False))

        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(actions), activation='softmax'))

    return model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def transformer(
        input_shape=(30, 150),
        output_shape=len(actions),
        head_size=512,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        dropout=0.1,
        mlp_dropout=0.1,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(output_shape, activation="softmax")(x)
    return keras.Model(inputs, outputs)


# ---------------------------
# Attention Layer (Bi-LSTM + Attention Model)
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform"
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)


# ---------------------------
# BiLSTM + Attention Model
# (Optimized for small dataset)
# ---------------------------
def bilstm_attention(device_name):

    with tf.device(device_name):

        inputs = tf.keras.Input(shape=(30, 150))

        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        x = Dropout(0.3)(x)

        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.3)(x)

        x = AttentionLayer()(x)

        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)

        x = Dense(128, activation='relu')(x)

        outputs = Dense(len(actions), activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs)

    return model

#Compile Model 
def compile_model(model):
    adam = tf.keras.optimizers.Adam(3e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def load_model(name='lstm_v3', pretrained=False, training=True, device=None):

    if not pretrained:
        if name == 'lstm_v1':
            model = lstm_v1(device)
            if training:
                return compile_model(model)
            else:
                return model

        if name == 'lstm_v2':
            model = lstm_v2(device)
            if training:
                return compile_model(model)
            else:
                return model

        if name == 'lstm_v3':
            model = lstm_v3(device)
            if training:
                return compile_model(model)
            else:
                return model

        if name == 'bilstm_attention':
            model = bilstm_attention(device)
            if training:
                return compile_model(model)
            else:
                return model
            
        if name == 'transformer':
            model = transformer()
            if training:
                return compile_model(model)
            else:
                return model

    model_dir = os.path.join('models', name)
    model_path = [os.path.join(model_dir, _) for _ in os.listdir(model_dir) if _.endswith(r".h5")][0]
    print(f"Loading Model from : {model_path}")

    model = tf.keras.models.load_model(model_path)

    if training:
        return compile_model(model)
    else:
        return model