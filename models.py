from keras.models import Model, Sequential
from keras import layers
import keras.backend as K


def get_baseline_convolutional_encoder(filters, embedding_dimension):
    encoder = Sequential()

    # Initial conv
    encoder.add(layers.Conv1D(filters, 32, padding='same', activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(0.1))
    encoder.add(layers.MaxPool1D())

    # Further convs
    encoder.add(layers.Conv1D(2*filters, 3, padding='same', activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(0.1))
    encoder.add(layers.MaxPool1D())

    encoder.add(layers.Conv1D(3 * filters, 3, padding='same', activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(0.1))
    encoder.add(layers.MaxPool1D())

    encoder.add(layers.Conv1D(4 * filters, 3, padding='same', activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(0.1))
    encoder.add(layers.MaxPool1D())

    encoder.add(layers.GlobalMaxPool1D())

    encoder.add(layers.Dense(embedding_dimension))

    return encoder


def euclidean_distance(inputs):
    assert len(inputs) == 2
    distance = K.mean(K.square(inputs[0] - inputs[1]), axis=-1)
    distance = K.expand_dims(distance, 1)
    return distance


def build_siamese_net(encoder, input_shape):
    input_1 = layers.Input(input_shape)
    input_2 = layers.Input(input_shape)

    encoded_1 = encoder(input_1)
    encoded_2 = encoder(input_2)

    embedded_distance = layers.Subtract()([encoded_1, encoded_2])
    embedded_distance = layers.Lambda(lambda x: K.abs(x))(embedded_distance)

    output = layers.Dense(1, activation='sigmoid')(embedded_distance)

    siamese = Model(inputs=[input_1, input_2], outputs=output)

    return siamese
