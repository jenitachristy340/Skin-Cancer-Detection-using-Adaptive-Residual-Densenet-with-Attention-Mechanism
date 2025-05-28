import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, BatchNormalization, Activation, Add
from tensorflow.keras.models import Model

from Evaluation import net_evaluation


# Residual Dense Block (RDB)
def residual_dense_block(x, filters, kernel_size=(3, 3), strides=(1, 1), padding="same"):
    inputs = x
    for _ in range(3):  # Three dense connections
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        inputs = Concatenate()([inputs, x])
    x = Conv2D(filters, (1, 1), padding=padding)(inputs)
    x = Add()([x, inputs])  # Residual connection
    return x


# Transformer Encoder Block
def transformer_block(x, filters, num_heads=4):
    x_norm = BatchNormalization()(x)
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=filters)(x_norm, x_norm)
    x = Add()([x, attn_output])  # Residual connection
    x = BatchNormalization()(x)
    ffn = Conv2D(filters, (1, 1), activation='relu')(x)
    ffn = Conv2D(filters, (1, 1))(ffn)
    x = Add()([x, ffn])  # Residual connection
    return x


# UNet++ with Residual Dense Blocks and Transformer Blocks
def build_unet_plus_plus(input_shape=(256, 256, 3), num_classes=1):
    inputs = Input(input_shape)

    # Encoder
    e1 = residual_dense_block(inputs, 64)
    e2 = residual_dense_block(e1, 128)
    e3 = residual_dense_block(e2, 256)
    e4 = residual_dense_block(e3, 512)

    # Bottleneck with Transformer
    bottleneck = transformer_block(e4, 512)

    # Decoder
    d4 = UpSampling2D()(bottleneck)
    d4 = Concatenate()([d4, e3])
    d4 = residual_dense_block(d4, 256)

    d3 = UpSampling2D()(d4)
    d3 = Concatenate()([d3, e2])
    d3 = residual_dense_block(d3, 128)

    d2 = UpSampling2D()(d3)
    d2 = Concatenate()([d2, e1])
    d2 = residual_dense_block(d2, 64)

    d1 = UpSampling2D()(d2)
    output = Conv2D(num_classes, (1, 1), activation='sigmoid')(d1)

    model = Model(inputs, output)
    return model

def Model_TRDUnetPlusPlus(image, mask):
    # Compile the model
    model = build_unet_plus_plus()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    model.fit(image, mask, validation_data=(image, mask), batch_size=16, epochs=10, steps_per_epochs=100)


    # Predict
    Segmented_Image = model.predict(image)[0]
    Eval = net_evaluation(Segmented_Image, image)
    return Eval, Segmented_Image
