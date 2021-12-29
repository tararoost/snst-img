content = None
style = None
VERBOSE = True
MAX = 256

def nst(content_path, style_path, epochs, steps_per_epoch):
    content = get_image(content_path, MAX)
    style = get_image(style_path, MAX)
    opt_performer(epochs, steps_per_epoch, image)

import numpy as np
import matplotlib.pyplot as pplot
import tensorflow as tf
import PIL.Image

def load_tf_im(im):
    if len(im.shape) > 3:
        im = tf.squeeze(im, axis=0)

def tensor_2_image(t):
    t = t*255
    t = np.array(t, dtype=np.uint8)
    if np.ndim(t) > 3:
        assert t.shape[0] == 1
        t = t[0]
    return PIL.Image.fromarray(t)

def get_image(path, MAX):
    im = tf.io.read_file(path)
    im = tf.image.decode_image(im, channels=3)
    im = tf.image.convert_image_dtype(im, tf.float32)

    shape = tf.cast(tf.shape(im)[:-1], tf.float32)
    liel = max(shape) / MAX
    _shape = tf.cast(shape * liel, tf.int32)
    im = tf.image.resize(im, _shape)
    im = im[tf.newaxis, :]
    return im

def imshow(im, title=None):
    if len(im.shape) > 3:
        im = tf.squeeze(im, axis=0)
    pplot.imshow(im)
    if title:
        pplot.title(title)

content_layers = ["block5_conv2"]

style_layers = ["block1_conv1",
                "block2_conv1",
                "block3_conv1",
                "block4_conv1",
                "block5_conv1"]
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style*255)
if (VERBOSE):
    for name,output in zip(style_layers, style_outputs):
        print(name)
        print("  shape: ", output.numpy().shape)
        print("  min: ", output.numpy().min())
        print("  max: ", output.numpy().max())
        print("  mean: ", output.numpy().mean())
        print()

def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                     for style_name, value
                     in zip(self.style_layers, style_outputs)}
        return {"content": content_dict, "style": style_dict}

extractor = StyleContentModel(style_layers, content_layers)
results = extractor(tf.constant(content))

if (VERBOSE):
    print('Styles:')
    for name, output in sorted(results['style'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())
        print()

    print("Contents:")
    for name, output in sorted(results['content'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())

# Stila un satura mērķi
style_targets = extractor(style)['style']
content_targets = extractor(content)['content']

# Optimizējamais attēls
image = tf.Variable(content)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# Optimizētājs
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# Svari
style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 30

def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    return x_var, y_var

def total_variation_loss(image):
    x_d, y_d = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_d)) + tf.reduce_sum(tf.abs(y_d))

# Loss funkcija
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                 for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                 for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

@tf.function()
@tf.autograph.experimental.do_not_convert
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

import time

def opt_performer(epochs, steps_per_epoch, image):
    begin = time.time()
    step = 0
    print("NST Optimizācija [{} / {} ({})]" .format(epochs, steps_per_epoch, epochs*steps_per_epoch))
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
            print("[{}, {}]".format(n, m),end='', flush=True)
            print("", end='\r')
        print("NST Optimizācija [{} / {} ({})]" .format(epochs, steps_per_epoch, epochs*steps_per_epoch))
        print("NST Solis: {} / {}" .format(step, epochs*steps_per_epoch))

    end = time.time()
    print("NST Kopā: {:.1f} s / {:.5f} min" .format(end - begin, (end - begin) / 60))
