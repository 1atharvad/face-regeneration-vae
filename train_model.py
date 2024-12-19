import os
import numpy as np
import tensorflow as tf
from keras import utils, models, layers, Model, optimizers, metrics

class ImageDataset:
    def __init__(self, file_name, batch_size=32, split_size=0.2):
        """ ImageDataset object for extracting the images from a specific
        folder.

        Args:
        file_name: Name of the folder from where the images are to be
            extracted.
        batch_size: The size of each batch, default is 32
        split_size: The value by which the dataset is to be divided into
            training and validation sets.
        """
        self.file_name = file_name
        self.batch_size = batch_size
        self.img_height = 218
        self.img_width = 178
        self.seed = 1234
        self.split_size = split_size
        self.train_count, self.test_count = self.get_image_count()

    def get_image_count(self):
        """ Gets the count of total number of images in the specific folder.
        """
        fileList = os.listdir(f'{self.file_name}/images')
        image_count = len(fileList)
        return np.ceil(image_count * (1 - self.split_size)), np.floor(image_count * self.split_size)

    def get_image_dataset(self):
        """ Get the dataset of training and validation images from the image
        directory for training and testing respectively.
        """
        train_df = utils.image_dataset_from_directory(
            os.path.join(self.file_name),
            label_mode=None,
            validation_split=self.split_size,
            subset="training",
            seed=self.seed,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)
        
        val_df = utils.image_dataset_from_directory(
            os.path.join(self.file_name),
            label_mode=None,
            validation_split=self.split_size,
            subset="validation",
            seed=self.seed,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)
        
        return train_df, val_df
    
class VariationalAutoencoder(Model):
    def __init__(self, latent_dim=6):
        """ VariationalAutoencoder model for initializing and creating the
        structure of encoder and decoder, and other required functions required
        for training the variational autoencoder (VAE) model.
        """
        super(VariationalAutoencoder, self).__init__()
        self.epochs = 10
        self.latent_dim = latent_dim
        self.encoder_model = self.encoder_()
        self.decoder_model = self.decoder_()
        # self.encoder_model.summary()
        # self.decoder_model.summary()

    def encoder_(self):
        """ Keras sequential model for encoder of VAE.
        """
        model = models.Sequential()
        model.add(layers.Conv2D(16, 3, strides=(2, 2), padding='same', input_shape=(218, 178, 3)))
        model.add(layers.Conv2D(32, 3, strides=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(2 * self.latent_dim))

        return model
    
    def decoder_(self):
        """ Keras sequential model for decoder of VAE.
        """
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(self.latent_dim,)))
        model.add(layers.Dense(units=54*44*64, activation=tf.nn.relu)),
        model.add(layers.Reshape(target_shape=(54, 44, 64)))
        model.add(layers.Conv2DTranspose(32, 3, strides=(2, 2), activation='relu'))
        model.add(layers.Conv2DTranspose(16, 3, strides=(2, 2), padding='same', activation='relu'))
        model.add(layers.Conv2DTranspose(3, 3, strides=(1, 1), padding='same'))

        return model
    
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, input_values):
        """ Sends the input top the encoder and splits the list into 2 lists
        for mean and variance for the latent space.

        Args:
        input_values: Input values for the encoder model.
        """
        return tf.split(self.encoder_model(input_values), num_or_size_splits=2, axis=1)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=(mean.shape[1],))
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, latent_variables, apply_sigmoid=False):
        """ Decodes the latent space variables to image.

        Args:
        latent_variables: Latent variables which are obtained from encoder.
        apply_sigmoid: Checks if the sigmoid activation function is to be
            applied to the output or not.
        """
        logits = self.decoder_model(latent_variables)
        return tf.sigmoid(logits) if apply_sigmoid else logits
    
class TrainEncoderDecoder:
    def __init__(self, epochs, batch_size=32):
        """ Training object for encoder and decoder of the variational
        autoencoder.

        Args:
        epochs: Number of times the training of encoder and decoder is to be
            executed.
        batch_size: The size of each batch, default is 32
        """
        self.epochs = epochs
        self.batch_size = batch_size
        df = ImageDataset('img_align_celeba', batch_size=self.batch_size)
        self.train_df, self.test_df = df.get_image_dataset()
        self.train_sample_count = df.train_count / self.batch_size
        self.model = VariationalAutoencoder()
        self.optimizer = optimizers.Adam(1e-4)
        self.train_model()

    @staticmethod
    def preprocess_images(images):
        """ Normalizes the values of images from 0 - 255 to values 0 - 1.

        Args:
        images: Images in tensor format.
        """
        return tf.Variable(images) / 255.

    @staticmethod
    def log_normal_pdf(sample, mean, logvar):
        return tf.reduce_sum(-0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + tf.math.log(2. * np.pi)), axis=1)

    def compute_loss(self, input_values):
        """ Computes the loss for variational autoencoder.

        Args:
        input_values: Input values for the encoder model.
        """
        mean, logvar = self.model.encode(input_values)
        z = self.model.reparameterize(mean, logvar)
        x_logit = self.model.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=input_values)

        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    @tf.function
    def train_step(self, input_values):
        """ Training the encoder and decoder of the VAE and updating the
        weights of encoder and decoder using Adam optimizer.

        Args:
        input_values: Input values for the encoder model. 
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(input_values)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def save_model(self):
        """ Save the weights of the encoder and decoder model in weights.h5
        file.
        """
        model_json = self.model.encoder_model.to_json()
        self.model.encoder_model.save_weights('encoder_model.weights.h5')
        with open('encoder_model.json', 'w') as json_file:
            json_file.write(model_json)

        model_json = self.model.decoder_model.to_json()
        self.model.decoder_model.save_weights('decoder_model.weights.h5')
        with open('decoder_model.json', 'w') as json_file:
            json_file.write(model_json)

    def train_model(self):
        """ Trains the models for certain number of epochs and calculates ELBO
        loss for validation.
        """
        initial_epoch = 0
        metrics_names = ['test_elbo']

        for epoch in range(initial_epoch, self.epochs):
            print(f'\nEpoch {epoch + 1}/{self.epochs}')
            progress_bar = utils.Progbar(self.train_sample_count + 1, stateful_metrics=metrics_names)
            for train_batch in self.train_df:
                train_batch_input = self.preprocess_images(list(train_batch))
                if not train_batch_input.shape[0] == self.batch_size:
                    continue
                self.train_step(self.preprocess_images(list(train_batch)))
                values=[]
                progress_bar.add(1)
        
            loss = metrics.Mean()
            for test_batch in self.test_df:
                test_batch_input = self.preprocess_images(list(test_batch))
                if not test_batch_input.shape[0] == self.batch_size:
                    continue
                loss(self.compute_loss(test_batch_input))
            values=[('elbo', -loss.result())]
            progress_bar.add(1, values=values)
            self.save_model()
    
if __name__ == '__main__':
    TrainEncoderDecoder(epochs=10, batch_size=100)