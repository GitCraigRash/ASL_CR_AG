import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Define the ResNet-50 architecture
def resnet50_model(input_shape, num_classes):
    # Input Layer
    input_tensor = Input(shape=input_shape)

    # Initial Conv Layer
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Max Pooling Layer
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Residual Blocks
    x = _residual_block(x, 64, 3, 1)
    x = _residual_block(x, 128, 4, 2)
    x = _residual_block(x, 256, 6, 2)
    x = _residual_block(x, 512, 3, 2)

    # Global Average Pooling Layer
    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layer (Dense Layer)
    output_tensor = Dense(num_classes, activation='softmax')(x)

    # Create and compile the model
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Helper function to create residual blocks
def _residual_block(x, filters, blocks, stride):
    shortcut = x

    # First convolution layer of the block
    x = Conv2D(filters, (1, 1), strides=(stride, stride))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second convolution layer of the block
    x = Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Third convolution layer of the block
    x = Conv2D(filters * 4, (1, 1))(x)
    x = BatchNormalization()(x)

    # Adjust the shortcut connection dimension if necessary
    if stride != 1 or shortcut.shape[-1] != filters * 4:
        shortcut = Conv2D(filters * 4, (1, 1), strides=(stride, stride))(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add the shortcut to the output
    x = Add()([x, shortcut])
    x = ReLU()(x)

    return x

# Define input shape and number of classes
input_shape = (200, 200, 3)  # Adjust based on your image size
num_classes = 28  # Number of classes (ASL alphabet)

# Create the ResNet-50 model
model = resnet50_model(input_shape, num_classes)

# Print the model summary
model.summary()

