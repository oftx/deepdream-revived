import argparse
import os
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # https://stackoverflow.com/questions/38073432/how-to-suppress-verbose-tensorflow-logging
os.environ["AUTOGRAPH_VERBOSITY"] = "0" # https://www.tensorflow.org/api_docs/python/tf/autograph/set_verbosity
import tensorflow
import numpy as np
import PIL.Image
from tensorflow.keras import layers, models

# Download an image and read it into a NumPy array.
def download(url, max_dim=None):
    name = url.split('/')[-1]
    image_path = tensorflow.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)

def load_img(image_path : str, max_dim=None):
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)

# Normalize an image
def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tensorflow.cast(img, tensorflow.uint8)

def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tensorflow.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tensorflow.math.reduce_mean(act)
        losses.append(loss)

    return  tensorflow.reduce_sum(losses)

class DeepDream(tensorflow.Module):
    def __init__(self, model):
        self.model = model

    @tensorflow.function(
        input_signature=(
            tensorflow.TensorSpec(shape=[None,None,3], dtype=tensorflow.float32),
            tensorflow.TensorSpec(shape=[], dtype=tensorflow.int32),
            tensorflow.TensorSpec(shape=[], dtype=tensorflow.float32),)
    )
    def __call__(self, img, steps, step_size):
        loss = tensorflow.constant(0.0)
        for n in tensorflow.range(steps):
            with tensorflow.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tensorflow.Variable`s by default
                tape.watch(img)
                loss = calc_loss(img, self.model)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tensorflow.math.reduce_std(gradients) + 1e-8 

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients*step_size
            img = tensorflow.clip_by_value(img, -1, 1)

        return loss, img

def random_roll(img, maxroll):
    # Randomly shift the image to avoid tiled boundaries.
    shift = tensorflow.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tensorflow.int32)
    img_rolled = tensorflow.roll(img, shift=shift, axis=[0,1])
    return shift, img_rolled

class TiledGradients(tensorflow.Module):
    def __init__(self, model):
        self.model = model

    @tensorflow.function(
        input_signature=(
            tensorflow.TensorSpec(shape=[None,None,3], dtype=tensorflow.float32),
            tensorflow.TensorSpec(shape=[2], dtype=tensorflow.int32),
            tensorflow.TensorSpec(shape=[], dtype=tensorflow.int32),)
    )
    def __call__(self, img, img_size, tile_size=512):
        shift, img_rolled = random_roll(img, tile_size)

        # Initialize the image gradients to zero.
        gradients = tensorflow.zeros_like(img_rolled)

        # Skip the last tile, unless there's only one tile.
        xs = tensorflow.range(0, img_size[1], tile_size)[:-1]
        if not tensorflow.cast(len(xs), bool):
            xs = tensorflow.constant([0])
        ys = tensorflow.range(0, img_size[0], tile_size)[:-1]
        if not tensorflow.cast(len(ys), bool):
            ys = tensorflow.constant([0])

        for x in xs:
            for y in ys:
                # Calculate the gradients for this tile.
                with tensorflow.GradientTape() as tape:
                    # This needs gradients relative to `img_rolled`.
                    # `GradientTape` only watches `tensorflow.Variable`s by default.
                    tape.watch(img_rolled)

                    # Extract a tile out of the image.
                    img_tile = img_rolled[y:y+tile_size, x:x+tile_size]
                    loss = calc_loss(img_tile, self.model)

                # Update the image gradients for this tile.
                gradients = gradients + tape.gradient(loss, img_rolled)

        # Undo the random shift applied to the image and its gradients.
        gradients = tensorflow.roll(gradients, shift=-shift, axis=[0,1])

        # Normalize the gradients.
        gradients /= tensorflow.math.reduce_std(gradients) + 1e-8 

        return gradients


def run_deep_dream_simple(img, dream_model, steps=100, step_size=0.01):
    deepdream = DeepDream(dream_model)
    # Convert from uint8 to the range expected by the model.
    img = tensorflow.keras.applications.inception_v3.preprocess_input(img)
    img = tensorflow.convert_to_tensor(img)
    step_size = tensorflow.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining>100:
            run_steps = tensorflow.constant(100)
        else:
            run_steps = tensorflow.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, tensorflow.constant(step_size))

        print ("  Step {}, loss {}".format(step, loss))

    result = deprocess(img)
    return result

def run_deep_dream_with_octaves(img, dream_model, steps_per_octave=100, step_size=0.01, octaves=range(-2,3), octave_scale=1.3):
    get_tiled_gradients = TiledGradients(dream_model)
    base_shape = tensorflow.shape(img)
    img = tensorflow.keras.utils.img_to_array(img)
    img = tensorflow.keras.applications.inception_v3.preprocess_input(img)

    initial_shape = img.shape[:-1]
    img = tensorflow.image.resize(img, initial_shape)
    for octave in octaves:
        print(f'  Octave {octave}')
        # Scale the image based on the octave
        new_size = tensorflow.cast(tensorflow.convert_to_tensor(base_shape[:-1]), tensorflow.float32)*(octave_scale**octave)
        new_size = tensorflow.cast(new_size, tensorflow.int32)
        img = tensorflow.image.resize(img, new_size)

        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(img, new_size)
            img = img + gradients*step_size
            img = tensorflow.clip_by_value(img, -1, 1)

    result = deprocess(img)
    return result


def save_img(img, img_name : str):
    PIL.Image.fromarray(np.array(img)).save(img_name, 'PNG')

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(prog='DeepDream runner')
        parser.add_argument('-i', '--input', type=str, default='example.png', help='Input file to process')
        parser.add_argument('-m', '--mode', type=str, choices=['simple', 'octaves'], default='simple', help='DeepDream processing method')
        parser.add_argument('-s', '--steps', type=int, default=100, help='Total number of steps, or steps per octave if using "octaves" mode')
        parser.add_argument('--step_size', type=float, default=0.1, help='Step size')
        parser.add_argument('--cpu', action='store_true', help="Use tensorflow in CPU only mode")

        args, unknown_args = parser.parse_known_args()
        if help in args:
            parser.print_help()

        if args.cpu:
            tensorflow.config.set_visible_devices([], 'GPU')
            print('Using CPU only mode.')
        else:
            print('Num GPUs Available: ', len(tensorflow.config.list_physical_devices('GPU')))
        
        original_img = load_img(args.input, max_dim=768)

        base_model = tensorflow.keras.applications.InceptionV3(include_top=False, weights='imagenet')

        # Maximize the activations of these layers
        names = ['mixed3', 'mixed5']
        layers = [base_model.get_layer(name).output for name in names]

        # Create the feature extraction model
        dream_model = tensorflow.keras.Model(inputs=base_model.input, outputs=layers)

        print(f'Using DeepDream mode "{args.mode}", steps: {args.steps}, step size: {args.step_size}')
        if args.mode == 'simple':
            dream_img = run_deep_dream_simple(original_img, dream_model, steps=args.steps, step_size=args.step_size)
        elif args.mode == 'octaves':
            dream_img = run_deep_dream_with_octaves(original_img, dream_model, steps_per_octave=args.steps, step_size=args.step_size)
        else:
            raise RuntimeError(f'Unrecognized mode {args.mode}')

        os.makedirs('output', exist_ok=True)
        save_img(dream_img, f'output/01.png')
        
    except Exception:
        print(traceback.format_exc())