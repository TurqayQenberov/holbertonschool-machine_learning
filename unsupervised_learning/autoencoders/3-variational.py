#!/usr/bin/env python3
"""
Variational Autoencoder Module
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Parameters:
    - input_dims: integer containing dimensions of model input
    - hidden_layers: list containing number of nodes for each hidden layer
    - latent_dims: integer containing dimensions of latent space

    Returns:
    - encoder: the encoder model
    - decoder: the decoder model
    - auto: the full autoencoder model
    """
    # Sampling layer
    def sampling(args):
        """Samples latent vector z using mu and log_sig"""
        mu, log_sig = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(mu)[0], latent_dims)
        )
        return mu + keras.backend.exp(log_sig / 2) * epsilon

    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(encoded)

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    encoder = keras.Model(
        inputs, [z, z_mean, z_log_sigma], name='encoder'
    )

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    decoded = latent_inputs
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    decoded_output = keras.layers.Dense(
        input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(latent_inputs, decoded_output, name='decoder')

    # Full VAE Autoencoder
    outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs, outputs, name='autoencoder')

    # Calculate VAE loss components
    reconstruction_loss = keras.losses.binary_crossentropy(
        inputs, outputs
    )
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) - \
        keras.backend.exp(z_log_sigma)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
