import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Dense, Flatten, GlobalAveragePooling2D, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

from Evaluation import evaluation


def residual_dense_block(x, filters, growth_rate=32):
    inputs = x
    for _ in range(4):  # 4 dense layers
        conv = Conv2D(growth_rate, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.Concatenate()([x, conv])  # Dense Connection

    # 1x1 Convolution to compress channels
    x = Conv2D(filters, (1, 1), padding='same')(x)

    # Residual Connection
    x = Add()([inputs, x])
    return x


def channel_attention(x):
    """Squeeze-and-Excitation Channel Attention"""
    channels = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Dense(channels // 16, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    return Multiply()([x, se])

def spatial_attention(x):
    """Spatial Attention with Conv"""
    sa = Conv2D(1, (7, 7), padding='same', activation='sigmoid')(x)
    return Multiply()([x, sa])


def build_model(sol, input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(int(sol[0]), (3, 3), padding='same', activation='relu')(inputs)
    x = residual_dense_block(x, 64)

    x = channel_attention(x)
    x = spatial_attention(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, x)
    return model


def Model_ResDAM(X_train, y_train, X_test, y_test, steps_per_epoch, sol=None):
    if sol is None:
        sol = [5, 5, 50]

    num_classes = y_train.shape[1]
    input_shape = (32, 32, 3)

    # Create and Compile Model
    model = build_model(sol, input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train Model
    model.fit(X_train, y_train, batch_size=32, epochs=int(sol[1]), validation_data=(X_test, y_test))


    # Predict
    predicted = model.predict(X_test)
    Eval = evaluation(predicted, y_test)

    return Eval, predicted


