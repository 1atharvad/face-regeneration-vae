import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button
from train_model import VariationalAutoencoder, ImageDataset

class VaeModel:
    def __init__(self, images_df, latent_dim=6):
        self.images_df = images_df
        self.latent_dim = latent_dim
        self.model_count = 2
        # self.latent_space = [[-0.2401372, -0.4247415, 5.128045, 2.3157817, 5.5133376, 4.0321558]]
        self.latent_space = [[-4.1201372, 1.4247415, -1.428045, -2.3957817, 0.3633376, 2.0321558]]
        self.latent_space_min = [-8.52058 , -6.1391387, -9.171443, -6.952, -7.37408, -7.703192]
        self.latent_space_max = [6.4052167, 7.942433, 6.4576216, 5.3063803, 7.0634346, 7.1675863]
        # print(self.get_latent_space_min_max())

    def load_vae_model(self, model_id):
        vae = VariationalAutoencoder(self.latent_dim)
        vae.encoder_model.load_weights(f'models/model{model_id}/encoder_model.weights.h5')
        vae.decoder_model.load_weights(f'models/model{model_id}/decoder_model.weights.h5')
        return vae

    def load_models(self):
        """ Loads pre-trained models of encoder and decoder of variational
        autoencoder from weights.h5 files.
        """
        return [self.load_vae_model(i + 1) for i in range(self.model_count)]

    def get_latent_space_min_max(self, vae):
        """ Gets the minimum and maximum values of the latent variables.
        """
        max = 0
        min = 0
        for index, train_batch in enumerate(self.images_df):
            mean, logvar = vae.encode(tf.Variable(list(train_batch)) / 255.)
            latent_space = vae.reparameterize(mean, logvar)
            min = tf.minimum(min, tf.reduce_min(latent_space, axis=0)) if index != 0 else tf.reduce_min(latent_space, axis=0)
            max = tf.maximum(max, tf.reduce_max(latent_space, axis=0)) if index != 0 else tf.reduce_max(latent_space, axis=0)

        return min, max
    
    def mask_image(self, image, mask_details):
        """ Masks certain section of the images using the mask details by
        setting certain pixel values to 0.

        Args:
        image: Image on which certain section is to be masked.
        mask_details: Details for masking certain section of the image.
        """
        image_shape = image.shape
        image = image[0].numpy()
        mask_width = int(image_shape[1] * mask_details['width'])
        mask_left = int((image_shape[2] - mask_width) * mask_details['left'])
        mask_top = int((image_shape[1] - mask_width) * (1 - mask_details['top']))

        for i in range(mask_width):
            for j in range(mask_width):
                image[i + mask_top][j + mask_left] = [0, 0, 0]

        return tf.Variable([image])
    
    def get_original_image(self):
        """ Returns random image from the image dataset.
        """
        return tf.Variable([list(self.images_df.take(1))[0][4]])
    
    def get_latent_space(self, vae, mask_details=None):
        """ Add mask to the image, and returns the masked image and latent
        space retrieved after encoding the masked image.

        Args:
        vae: Object of variational autoencoder.
        mask_details: Details for masking certain section of the image.

        Returns:
        masked_image: Masked image created using mask details.
        latent_space: Latent space retrieved after encoding the masked image.
        """
        masked_image = self.mask_image(self.original_image, mask_details) if mask_details != None else self.original_image
        mean, logvar = vae.encode(masked_image / 255.)
        latent_space = vae.reparameterize(mean, logvar)

        return masked_image, latent_space
    
    def calculate_MSE(self, actual_values, target_values):
        """ Calculate mean square error between actual and expected values.

        Args:
        actual_values: Values retrieved from decoder of VAE.
        target_values: Input values to the encoder of VAE.
        """
        return np.round(tf.reduce_mean(tf.square((target_values / 255.) - (actual_values / 255.))).numpy() / 1., 4)

    def generate_images(self):
        """ Generates a plot for displaying the reconstructed image for set of
        latent variables captured from the slider on the plot.
        """
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(9, 2, figure=fig)
        ax_title = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1:-1, 1])
        ax2 = fig.add_subplot(gs[2, 0])
        ax3 = fig.add_subplot(gs[3, 0])
        ax4 = fig.add_subplot(gs[4, 0])
        ax5 = fig.add_subplot(gs[5, 0])
        ax6 = fig.add_subplot(gs[6, 0])
        ax7 = fig.add_subplot(gs[7, 0])
        ax8 = fig.add_subplot(gs[1, 0])
        ax9 = fig.add_subplot(gs[-1, 1])

        ax_title.text(0.5, 0.5, 'Task 1', ha='center', va='center', fontsize=14, fontweight='bold')
        ax8.text(0.5, 0.5, 'Latent Variables', ha='center', va='center')
        ax9.text(0.5, 0.5, 'Reconstructed Image', ha='center', va='center')

        vae = self.vae_models[0]
        predictions = vae.sample(tf.Variable(self.latent_space)) * 255.
        image = ax1.imshow(predictions[0].numpy().astype("uint8"))

        def update(val):
            """ Updates the decoder image after changes in latent variables
            done using sliders.
            """
            for i, slider in enumerate(freq_slider):
                self.latent_space[0][i] = slider.val
            predictions = vae.sample(tf.Variable(self.latent_space)) * 255.
            image.set_data(predictions[0].numpy().astype("uint8"))
            fig.canvas.draw_idle()

        freq_slider = []
        for index, ax in enumerate([ax2, ax3, ax4, ax5, ax6, ax7]):
            freq_slider.append(Slider(
                ax=fig.add_axes(ax).inset_axes([0, 0.35, 1, 0.3]),
                label=f'Dim {index + 1}',
                valmin=self.latent_space_min[index],
                valmax=self.latent_space_max[index],
                valinit=self.latent_space[0][index],
            ))

            freq_slider[index].on_changed(update)

        ax_title.axis('off')
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')
        ax5.axis('off')
        ax6.axis('off')
        ax7.axis('off')
        ax8.axis('off')
        ax9.axis('off')

        fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.1)

        plt.show()

    def generate_images_from_masked_frame(self):
        """ Generates a plot for displaying the image with the mask and 2 VAE
        model's decoder outputs for the masked image.
        """
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(9, 4, figure=fig, width_ratios=[1, 3, 3, 3])
        ax_title = fig.add_subplot(gs[0, :])
        ax0 = fig.add_subplot(gs[3:-1, 0])
        ax1 = fig.add_subplot(gs[3:-1, 1])
        ax2 = fig.add_subplot(gs[3:-1, 2])
        ax3 = fig.add_subplot(gs[3:-1, 3])
        ax4 = fig.add_subplot(gs[-1, 1])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[2, 1])
        ax7 = fig.add_subplot(gs[2, 2])
        ax8 = fig.add_subplot(gs[2, 3])
        ax9 = fig.add_subplot(gs[-1, 2])
        ax10 = fig.add_subplot(gs[-1, 3])

        mask_details = {
            'width': 0.25,
            'left': 0.5,
            'top': 0.5 
        }

        ax_title.text(0.5, 0.5, 'Task 2', ha='center', va='center', fontsize=14, fontweight='bold')
        ax9.text(0.5, 0.5, f'Reconstructed image\nusing VAE 1', ha='center', va='center')
        ax10.text(0.5, 0.5, f'Reconstructed image\nusing VAE 2', ha='center', va='center')

        decoder_images = []
        decoder_MSE_axis = [ax7, ax8]
        decoder_MSE_text = []
        decoder_images_axis = [ax2, ax3]

        self.original_image = self.get_original_image()
        for index, ax in enumerate(decoder_images_axis):
            masked_images, latent_space = self.get_latent_space(self.vae_models[index], mask_details)
            predictions = self.vae_models[index].sample(tf.Variable(latent_space)) * 255.
            mse = self.calculate_MSE(masked_images, predictions)
            decoder_images.append(ax.imshow(predictions[0].numpy().astype("uint8")))
            decoder_MSE_text.append(decoder_MSE_axis[index].text(0.5, 0.5, f'MSE = {mse}', ha='center', va='center'))
        main_image = ax1.imshow(masked_images[0].numpy().astype("uint8"))

        def next(event):
            """ To load new image when the "Load new image" button is clicked.
            """
            self.original_image = self.get_original_image()
            for index, image in enumerate(decoder_images):
                masked_images, latent_space = self.get_latent_space(self.vae_models[index], mask_details)
                predictions = self.vae_models[index].sample(tf.Variable(latent_space)) * 255.
                mse = self.calculate_MSE(masked_images, predictions)
                image.set_data(predictions[0].numpy().astype("uint8"))
                decoder_MSE_text[index].set_text(f'MSE = {mse}')
            main_image.set_data(masked_images[0].numpy().astype("uint8"))
            fig.canvas.draw_idle()

        def update_mask(val):
            """ Makes update to the original image by updating the mask on the
            image based on the slider values.
            """
            for slider, details in zip(sliders, slider_details):
                mask_details[details['type']] = slider.val

            for index, image in enumerate(decoder_images):
                masked_images, latent_space = self.get_latent_space(self.vae_models[index], mask_details)
                predictions = self.vae_models[index].sample(tf.Variable(latent_space)) * 255.
                mse = self.calculate_MSE(masked_images, predictions)
                image.set_data(predictions[0].numpy().astype("uint8"))
                decoder_MSE_text[index].set_text(f'MSE = {mse}')
            main_image.set_data(masked_images[0].numpy().astype("uint8"))
            fig.canvas.draw_idle()
            
        slider_details = [
            {
                'axis': ax5,
                'inner_axis': [0, 0.35, 0.8, 0.3],
                'label': 'Mask Width',
                'type': 'width',
                'min_val': 0,
                'max_val': 0.5,
                'default': mask_details['width'],
                'orientation': 'horizontal'
            },
            {
                'axis': ax6,
                'inner_axis': [0, 0.35, 0.8, 0.3],
                'label': 'Top left x',
                'type': 'left',
                'min_val': 0,
                'max_val': 1,
                'default': mask_details['left'],
                'orientation': 'horizontal'
            },
            {
                'axis': ax0,
                'inner_axis': [0.35, 0.07, 0.3, 0.83],
                'label': 'Top left y',
                'type': 'top',
                'min_val': 0,
                'max_val': 1,
                'default': mask_details['top'],
                'orientation': 'vertical'
            }
        ]

        sliders = []

        for index, slider in enumerate(slider_details):
            sliders.append(Slider(
                ax=fig.add_axes(slider['axis'].inset_axes(slider['inner_axis'])),
                label=slider['label'],
                valmin=slider['min_val'],
                valmax=slider['max_val'],
                valinit=slider['default'],
                orientation=slider['orientation']
            ))

            sliders[index].on_changed(update_mask)

        new_image_btn = Button(ax4, 'Load New Image')
        new_image_btn.on_clicked(next)
        ax_title.axis('off')
        ax0.axis('off')
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        # ax4.axis('off')
        ax5.axis('off')
        ax6.axis('off')
        ax7.axis('off')
        ax8.axis('off')
        ax9.axis('off')
        ax10.axis('off')

        fig.subplots_adjust(top=0.95, bottom=0.1, right=0.95, left=0.05)

        plt.show()

def transfer_images():
    import shutil
    import os

    src_dir = 'img_align_celeba/images'
    dst_dir = 'images'
    n = 202599
    for imageNumber in range(1, n, int(n/3000)):
        shutil.copy(os.path.join(src_dir, f'{str(imageNumber).zfill(len(str(n)))}.jpg'), dst_dir)
    
if __name__ == '__main__':
    df = ImageDataset('img_align_celeba', split_size=0.01, batch_size=100)
    train_df, _ = df.get_image_dataset()
    model = VaeModel(train_df)
    model.vae_models = model.load_models()
    model.generate_images()
    model.generate_images_from_masked_frame()
