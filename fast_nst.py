import numpy as np
import os
import PIL.Image
import shutil
import tensorflow as tf
import tensorflow_hub as hub

def fast_nst(image_path, style_path, model_path, max):

    model_tf_hub_url = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"

    if (not max):
        max = 256

    def tensor_2_image(t):
        t = t * 255
        t = np.array(t, dtype=np.uint8)
        if np.ndim(t) > 3:
            assert t.shape[0] == 1
            t = t[0]
        return PIL.Image.fromarray(t)

    content = tf.io.decode_image(tf.io.read_file(image_path),
                                 channels = 3, dtype = tf.float32)[tf.newaxis, ...]

    style = tf.io.decode_image(tf.io.read_file(style_path),
                                 channels = 3, dtype = tf.float32)[tf.newaxis, ...]

    style = tf.image.resize(style, (max, max), preserve_aspect_ratio = True)

    if (not model_path):
        nst = hub.load(model_tf_hub_url)
        out = nst(tf.constant(content), tf.constant(style))
        return tensor_2_image(out[0])
    else:
        ld = os.path.dirname(__file__)
        m_fnst = os.path.join(ld, "tmp/m_fnst")
        os.makedirs(m_fnst, exist_ok = True)
        shutil.unpack_archive(model_path, m_fnst)

        nst = hub.load(m_fnst)
        out = nst(tf.constant(content), tf.constant(style))
        return tensor_2_image(out[0])

