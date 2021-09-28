import tensorflow as tf
import os
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

SIZE = 256
IMAGE_SIZE = [SIZE, SIZE]
NUM_EPOCHS = 2
INITIAL_LR = 0.001
BETA_1 = 0.5


def load_data(monet, photo):
    monet_ds = decode_image(monet)
    photo_ds = decode_image(photo)
    return tf.data.Dataset.zip((monet_ds, photo_ds))


def decode_image(filenames):
    images = []
    for file in filenames:
        image = (tf.cast(tf.image.decode_jpeg(tf.io.read_file(file), channels=3), tf.float32) / SIZE / 2.0) - 1.0
        images.append(tf.reshape(image, [*IMAGE_SIZE, 3]))
    return tf.data.Dataset.from_tensor_slices(images).batch(1, drop_remainder=True)


def decode_a_image(filename):
    image = (tf.cast(tf.image.decode_jpeg(tf.io.read_file(filename), channels=3), tf.float32) / SIZE / 2.0) - 1.0
    return tf.data.Dataset.from_tensors(tf.reshape(image, [*IMAGE_SIZE, 3])).batch(1, drop_remainder=True)


def create_generator():
    first_stack = tf.keras.layers.Input(shape=[*IMAGE_SIZE, 3])
    down_stack = [
        downsample(64, 4, apply_instancenorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    last_stack = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding="same",
                                                 kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                                 activation=tf.keras.activations.tanh)

    x = first_stack
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last_stack(x)
    return tf.keras.Model(inputs=first_stack, outputs=x)


def create_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    first_stack = tf.keras.layers.Input(shape=[*IMAGE_SIZE, 3], name="input_image")

    x = first_stack

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(pad1)
    norm = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)
    leaky = tf.keras.layers.LeakyReLU()(norm)
    pad2 = tf.keras.layers.ZeroPadding2D()(leaky)
    last_stack = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(pad2)
    return tf.keras.Model(inputs=first_stack, outputs=last_stack)


def downsample(filters, size, apply_instancenorm=True, strides=2):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding="same",
                                      kernel_initializer=tf.random_normal_initializer(0., 0.02)))
    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False, strides=2):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding="same",
                                               kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


with tf.distribute.get_strategy().scope():
    def discriminator_loss(real, fake):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)
        fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(fake), fake)
        return (real_loss + fake_loss) * 0.5

    def generator_loss(fake):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(fake), fake)

    with tf.distribute.get_strategy().scope():
        def cycle_loss(real, cycled, LAMBDA):
            loss1 = tf.reduce_mean(tf.abs(real - cycled))
            return LAMBDA * loss1

    with tf.distribute.get_strategy().scope():
        def identity_loss(real, same, LAMBDA):
            loss = tf.reduce_mean(tf.abs(real - same))
            return LAMBDA * 0.5 * loss

with tf.distribute.get_strategy().scope():
    def train_model(monet, photo):
        if os.path.isfile('./monet_model'):
            model = tf.keras.models.load_model('./monet_model.')
        else:
            m_gen_optimizer = tf.keras.optimizers.Adam(INITIAL_LR, beta_1=BETA_1)
            p_gen_optimizer = tf.keras.optimizers.Adam(INITIAL_LR, beta_1=BETA_1)
            m_disc_optimizer = tf.keras.optimizers.Adam(INITIAL_LR, beta_1=BETA_1)
            p_disc_optimizer = tf.keras.optimizers.Adam(INITIAL_LR, beta_1=BETA_1)

            model = CycleGan(monet_generator, photo_generator, monet_discriminator, photo_discriminator)
            model.compile(m_gen_optimizer=m_gen_optimizer, p_gen_optimizer=p_gen_optimizer, m_disc_optimizer=m_disc_optimizer, p_disc_optimizer=p_disc_optimizer,
                          gen_loss_fn=generator_loss, disc_loss_fn=discriminator_loss, cycle_loss_fn=cycle_loss, identity_loss_fn=identity_loss)
        history = model.fit(load_data(monet, photo), epochs=NUM_EPOCHS, batch_size=1).history
        return model, history


def main():
    monet_path = '.\\data\\art_creator\\monet_jpg'
    photo_path = '.\\data\\art_creator\\photo_jpg'
    monet = [os.path.join(monet_path, file) for file in os.listdir(monet_path)]
    photo = [os.path.join(photo_path, file) for file in os.listdir(photo_path)]
    model, history = train_model(monet, photo)
    monet_generator.save('./monet_generator.h5')
    photo_generator.save('./photo_generator.h5')
    monet_discriminator.save('./monet_discriminator.h5')
    photo_discriminator.save('./photo_discriminator.h5')
    display_generated_samples(decode_a_image(photo[0]), monet_generator)


def display_generated_samples(image, model):
    generated_sample = model.predict(image)

    plt.subplot(121)
    plt.title("input image")
    plt.imshow(tf.data.experimental.get_single_element(image)[0] * 0.5 + 0.5)
    plt.axis('off')

    plt.subplot(122)
    plt.title("generated image")
    plt.imshow(generated_sample[0] * 0.5 + 0.5)
    plt.axis('off')
    plt.show()


with tf.distribute.get_strategy().scope():
    if os.path.isfile('./monet_generator.h5'):
        monet_generator = tf.keras.models.load_model('./monet_generator.h5')
        photo_generator = tf.keras.models.load_model('./photo_generator.h5')

        monet_discriminator = tf.keras.models.load_model('./monet_discriminator.h5')
        photo_discriminator = tf.keras.models.load_model('./photo_discriminator.h5')
    else:
        monet_generator = create_generator()
        photo_generator = create_generator()

        monet_discriminator = create_discriminator()
        photo_discriminator = create_discriminator()


class CycleGan(tf.keras.Model):
    def __init__(self, monet_generator, photo_generator, monet_discriminator, photo_discriminator, lambda_cycle=10):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle

    def compile(self, m_gen_optimizer, p_gen_optimizer, m_disc_optimizer, p_disc_optimizer, gen_loss_fn, disc_loss_fn, cycle_loss_fn, identity_loss_fn):
        super(CycleGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def train_step(self, batch_data):
        real_monet, real_photo = batch_data

        with tf.GradientTape(persistent=True) as tape:
            fake_monet = self.m_gen(real_photo, training=True)
            cycled_photo = self.p_gen(fake_monet, training=True)

            fake_photo = self.p_gen(real_monet, training=True)
            cycled_monet = self.m_gen(fake_photo, training=True)

            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            disc_real_monet = self.m_disc(real_monet, training=True)
            disc_real_photo = self.p_disc(real_photo, training=True)

            disc_fake_monet = self.m_disc(fake_monet, training=True)
            disc_fake_photo = self.p_disc(fake_photo, training=True)

            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet, self.lambda_cycle) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)

            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)

            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

            monet_gen_grads = tape.gradient(total_monet_gen_loss, self.m_gen.trainable_variables)
            photo_gen_grads = tape.gradient(total_photo_gen_loss, self.p_gen.trainable_variables)

            monet_disc_grads = tape.gradient(monet_disc_loss, self.m_disc.trainable_variables)
            photo_disc_grads = tape.gradient(photo_disc_loss, self.p_disc.trainable_variables)

            self.m_gen_optimizer.apply_gradients(zip(monet_gen_grads, self.m_gen.trainable_variables))
            self.p_gen_optimizer.apply_gradients(zip(photo_gen_grads, self.p_gen.trainable_variables))
            self.m_disc_optimizer.apply_gradients(zip(monet_disc_grads, self.m_disc.trainable_variables))
            self.p_disc_optimizer.apply_gradients(zip(photo_disc_grads, self.p_disc.trainable_variables))

            return {
                "monet_gen_loss": total_monet_gen_loss,
                "photo_gen_loss": total_photo_gen_loss,
                "monet_disc_loss": monet_disc_loss,
                "photo_disc_loss": photo_disc_loss
            }


if __name__ == '__main__':
    main()
