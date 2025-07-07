import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from PIL import Image

# Set the dimensions of the generated images
latent_dim = 100
output_dim = (28, 28, 1)  # Adjust according to your image size and channels

# Define the generator model
def make_generator_model():
    model = Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(latent_dim,)))
    model.add(Reshape((7, 7, 256)))
    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

# Load the GAN model
gan_model = make_generator_model()
gan_model.load_weights('gan_model_weights.h5')  # Load the pre-trained weights

# Specify the directory path where your image data is stored
output_directory = '/path/to/image/directory'
# Specify the directory path where your image data is loaded
image_directory = '/path/to/image/directory'


# Specify the number of images you want to generate
num_images = 10

# Generate synthetic images
for i in range(num_images):
    # Load an image from the image directory
    image_path = os.path.join(image_directory, f'image_{i}.png')  # Adjust the filename pattern if needed
    image = Image.open(image_path)

    # Preprocess the loaded image (resize, normalize, etc.) to match the GAN input requirements
    preprocessed_image = image.resize(output_dim[:2])  # Resize the image to match the GAN output size
    preprocessed_image = tf.keras.preprocessing.image.img_to_array(preprocessed_image)
    preprocessed_image = (preprocessed_image - 127.5) / 127.5  # Normalize the pixel values to the range [-1, 1]
    preprocessed_image = tf.expand_dims(preprocessed_image, axis=0)  # Add a batch dimension

    # Generate an image using the GAN model
    generated_image = gan_model.predict(preprocessed_image)

    # Rescale the pixel values from [-1, 1] to [0, 255]
    generated_image = (generated_image + 1) * 127.5

    # Convert the generated image to PIL format
    generated_image = Image.fromarray(generated_image[0].astype('uint8'))

    # Set the save path for the generated image
    save_path = os.path.join(output_directory, f'generated_image_{i}.png')

    # Save the generated image
    generated_image.save(save_path)

    print(f'Saved generated image {i+1}/{num_images} as {save_path}')