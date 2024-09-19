import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing import image

image_height = 227
image_width = 227
batch_size = 20
train_path = '/content/train'
test_path = '/content/test'

training_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    batch_size=batch_size,
    image_size=(image_height, image_width),
    shuffle=True,
    seed=123
)

validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    batch_size=batch_size,
    image_size=(image_height, image_width),
    shuffle=True,
    seed=123
)

class QuantumLikeLayer(tf.keras.layers.Layer):
    def __init__(self, num_qubits, trainable=True):
        super(QuantumLikeLayer, self).__init__(trainable=trainable)

        self.num_qubits = num_qubits
        self.hadamard_gate = self.create_hadamard_gate()
        self.cz_gate = self.create_cz_gate()
    def get_config(self):
        config = super(QuantumLikeLayer, self).get_config()
        config.update({
            'num_qubits': self.num_qubits,
        })
        return config
    def create_hadamard_gate(self):
        H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.float64)
        hadamard_full = H
        for _ in range(1, self.num_qubits):
            hadamard_full = np.kron(hadamard_full, H)
        return tf.linalg.LinearOperatorFullMatrix(hadamard_full)

    def create_cz_gate(self):
        CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.float64)
        cz_full = CZ
        for _ in range(2, self.num_qubits):
            cz_full = np.kron(cz_full, np.eye(2, dtype=np.float64))
        return tf.linalg.LinearOperatorFullMatrix(cz_full)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float64)
        state = self.hadamard_gate.matvec(inputs)
        state = self.cz_gate.matvec(state)
        return state

num_qubits = 11
quantum_like_layer = QuantumLikeLayer(num_qubits, trainable=False)
input_state = tf.constant(np.ones(2**num_qubits, dtype=np.float64))
output_state = quantum_like_layer(input_state)

resnet_model = Sequential()

pretrained_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(227, 227, 3),
    pooling='avg',
    classes=23,
    weights='imagenet'
)

for layer in pretrained_model.layers:
    layer.trainable = True

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())

# num_qubits = 11
resnet_model.add(Dense(2**num_qubits))

resnet_model.add(quantum_like_layer)
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(23, activation='softmax'))
# Compile the model
resnet_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.optimizers.SGD(learning_rate=0.001),
    metrics=['accuracy']
)

# Train the model
epochs = 5
history = resnet_model.fit(
    training_data,
    validation_data=validation_data,
    epochs=epochs
)


