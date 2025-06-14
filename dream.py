import argparse
import random
import mimetypes
import os
import subprocess
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # https://stackoverflow.com/questions/38073432/how-to-suppress-verbose-tensorflow-logging
os.environ["AUTOGRAPH_VERBOSITY"] = "0" # https://www.tensorflow.org/api_docs/python/tf/autograph/set_verbosity
import tensorflow
import numpy as np
import PIL.Image
import PIL.ImageChops
from tensorflow.keras import layers, models

ffmpeg_path = None # Edit this if you want to specify a custom path to ffmpeg
ffprobe_path = ffmpeg_path # Usually you should not have to edit this
ffmpeg_exe = os.path.join(ffmpeg_path, 'ffmpeg') if ffmpeg_path else 'ffmpeg'
ffprobe_exe = os.path.join(ffprobe_path, 'ffprobe') if ffprobe_path else 'ffprobe'

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
    return np.array(img), img.size

def blend_img(next_image : str, prev_image : str, prev_dream_image : str, blend_amount=0.0, diff=False, max_dim=None):
    img = PIL.Image.open(next_image).convert('RGB')
    dream = PIL.Image.open(prev_dream_image).convert('RGB')
    if diff:
        prev = PIL.Image.open(prev_image).convert('RGB')
        if prev.size != dream.size:
            prev = prev.resize(dream.size)
        diff = PIL.ImageChops.difference(dream,prev)
        dream = diff
    # Images need to be the same size (this can happen when the dream has been scaled up or down
    if img.size != dream.size:
        dream = dream.resize(img.size)

    # Note: If blend is 0.0, a copy of image 1 is returned. If blend is 1.0, a copy of image 2 is returned.
    # Therefore the previous dream image in the sequence should be passed in as image 2
    blended = PIL.Image.blend(img, dream, blend_amount)
    if max_dim:
        blended.thumbnail((max_dim, max_dim))
    return np.array(blended), blended.size

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
        # Can't process images with an alpha channel, have to reshape to simple RGB
        if img.shape[2] == 4:
            img = img[..., :3]  # Keep only the first three channels (R, G, B)
        loss, img = deepdream(img, run_steps, tensorflow.constant(step_size))
        print ("  Step {}, loss {}".format(step, loss))

    result = deprocess(img)
    return result

def run_deep_dream_with_octaves(img, dream_model, steps_per_octave=100, step_size=0.01, octaves=range(-2,3), octave_scale=1.3, tile_size=512):
    get_tiled_gradients = TiledGradients(dream_model)
    # Can't process images with an alpha channel, have to reshape to simple RGB
    if img.shape[2] == 4:
        img = img[..., :3]  # Keep only the first three channels (R, G, B)
    base_shape = tensorflow.shape(img)
    img = tensorflow.keras.utils.img_to_array(img)
    img = tensorflow.keras.applications.inception_v3.preprocess_input(img)

    initial_shape = img.shape[:-1]
    img = tensorflow.image.resize(img, initial_shape)
    for octave in octaves:
        #print(f'  Octave {octave}')
        # Scale the image based on the octave
        new_size = tensorflow.cast(tensorflow.convert_to_tensor(base_shape[:-1]), tensorflow.float32)*(octave_scale**octave)
        new_size = tensorflow.cast(new_size, tensorflow.int32)
        img = tensorflow.image.resize(img, new_size)

        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(img, new_size, tile_size=tile_size)
            img = img + gradients*step_size
            img = tensorflow.clip_by_value(img, -1, 1)

    result = deprocess(img)
    return result

def save_img(img, img_name : str):
    PIL.Image.fromarray(np.array(img)).save(img_name, 'PNG')

def get_input_framerate(input_filename : str):
    result = subprocess.run([ffprobe_exe,"-v", "error", "-select_streams", "v", "-of", "default=noprint_wrappers=1:nokey=1", "-show_entries", "stream=r_frame_rate", input_filename], stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print('ffprobe returned error code {}'.format(result.returncode))
        print('Error getting input fps.')
        return None
    # Outputs the frame rate as a precise fraction. Have to convert to decimal.
    source_fps_fractional = result.stdout.split('/')
    source_fps = round(float(source_fps_fractional[0]) / float(source_fps_fractional[1]), 2)
    return source_fps

def output_to_png_sequence(input_filename : str, output_dir : str):
    ffmpeg_args = ["ffmpeg", '-hide_banner', '-y', '-i', input_filename]
    framerate = get_input_framerate(input_filename)
    if framerate is not None:
        ffmpeg_args.extend(['-vf', f'fps={framerate}'])
    ffmpeg_args.append(os.path.join(output_dir,'%03d.png'))
    # Use popen so we can pend on completion
    result = subprocess.run(ffmpeg_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, encoding='utf-8', errors='ignore')
    if result.returncode != 0:
        print(' '.join(ffmpeg_args))
        print(result.stderr)
        raise RuntimeError('ffmpeg returned error code {}'.format(result.returncode))
    png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    return sorted(png_files)

def concat_png_sequence(input_filename : str, png_dir : str, output_dir : str):
    mime, subtype = mimetypes.guess_type(input_filename)[0].split('/')
    framerate = get_input_framerate(input_filename)
    file_basename = os.path.splitext(os.path.basename(args.input))[0]
    output_filename = os.path.join(output_dir,file_basename + ('.gif' if subtype == 'gif' else '.mp4'))
    ffmpeg_args = ["ffmpeg", '-hide_banner', '-y', '-framerate', f'{framerate}', '-pattern_type', 'glob', '-i', os.path.join(png_dir,'*.png')]
    if subtype != 'gif':
        ffmpeg_args.extend(['-c:v', 'libx264', '-crf', '20'])
    ffmpeg_args.append(output_filename)
    result = subprocess.run(ffmpeg_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, encoding='utf-8', errors='ignore')
    if result.returncode != 0:
        print(' '.join(ffmpeg_args))
        print(result.stderr)
        raise RuntimeError('ffmpeg returned error code {}'.format(result.returncode))
    return output_filename

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(prog='DeepDream runner')
        parser.add_argument('--blend', type=float, default=0.0, help='Amount to blend images when processing video')
        parser.add_argument('--cpu', action='store_true', help="Use tensorflow in CPU only mode")
        parser.add_argument('--diff', action='store_true', help="Enable calculating the difference between AI noise and original image when blending images for video")
        parser.add_argument('-i', '--input', type=str, default='example.png', help='Input file to process')
        parser.add_argument('--max_size', type=int, help='Maximum allowed image size. Default is no max size. Limit this if you run out of RAM/VRAM')
        parser.add_argument('--mode', type=str, choices=['simple', 'octaves'], default='simple', help='DeepDream processing method')
        parser.add_argument('--octaves', type=str, default='-2, 1, 0, 1, 2', help='List of octaves to run. May be a comma separated list of integers, or "random x" or "range x y"')
        parser.add_argument('--output', type=str, default='output', help='Output directory')
        parser.add_argument('--rand_min', type=int, default=-65535, help='Lower bound for an octave when using --octaves "random x"')
        parser.add_argument('--rand_max', type=int, default=65535, help='Upper bound for an octave when using --octaves "random x"')
        parser.add_argument('--scale', type=float, default=1.0, help='Scale factor to use in octaves mode')
        parser.add_argument('--steps', type=int, default=20, help='Total number of steps, or steps per octave if using "octaves" mode')
        parser.add_argument('--step_size', type=float, default=0.1, help='Step size')
        parser.add_argument('--tile_size', type=int, help='Tile size used for "octaves" mode. If not specified, the tile size is the max dimension of the input image.')

        args, unknown_args = parser.parse_known_args()
        if help in args:
            parser.print_help()

        if args.cpu:
            tensorflow.config.set_visible_devices([], 'GPU')
            print('Using CPU only mode.')
        else:
            print('Num GPUs Available: ', len(tensorflow.config.list_physical_devices('GPU')))

        base_model = tensorflow.keras.applications.InceptionV3(include_top=False, weights='imagenet')

        # Maximize the activations of these layers
        names = ['mixed3', 'mixed5']
        layers = [base_model.get_layer(name).output for name in names]

        # Create the feature extraction model
        dream_model = tensorflow.keras.Model(inputs=base_model.input, outputs=layers)

        print(f'Using DeepDream mode "{args.mode}", steps: {args.steps}, step size: {args.step_size}')
        if 'random' in args.octaves:
            num_octaves = int(args.octaves.split(' ')[-1])
            octaves = [random.randint(args.rand_min, args.rand_max) for _ in range(num_octaves)]
        elif 'range' in args.octaves:
            _, lower, upper = args.octaves.split(' ')
            octaves = range(int(lower), int(upper))
        else:
            octaves = [int(num) for num in args.octaves.split(',')]
        if args.mode == 'octaves':
            print(f'Octaves: {[octave for octave in octaves]}')
        os.makedirs(args.output, exist_ok=True)
        # Determine what to do with the input
        mime, subtype = mimetypes.guess_type(args.input)[0].split('/')
        if mime == 'image' and subtype != 'gif':
            original_img, img_size = load_img(args.input, max_dim=args.max_size)
            if args.mode == 'simple':
                dream_img = run_deep_dream_simple(original_img, dream_model, steps=args.steps, step_size=args.step_size)
            elif args.mode == 'octaves':
                dream_img = run_deep_dream_with_octaves(original_img, dream_model,
                    steps_per_octave=args.steps,
                    step_size=args.step_size,
                    octave_scale=args.scale,
                    tile_size=args.tile_size if args.tile_size is not None else max(img_size))
            else:
                raise RuntimeError(f'Unrecognized mode {args.mode}')
            output_basename = os.path.splitext(os.path.basename(args.input))[0]
            output_filename = os.path.join(args.output,output_basename + '.png')
            save_img(dream_img, output_filename)
            print(f'Output rendered to {output_filename}')
        elif mime == 'video' or (mime == 'image' and subtype == 'gif'):
            file_basename = os.path.splitext(os.path.basename(args.input))[0]
            output_dirname = os.path.join(args.output, file_basename) 
            os.makedirs(output_dirname, exist_ok=True)
            print(f'Processing frames into {output_dirname} ...')
            output_files = output_to_png_sequence(args.input, output_dirname)
            print(f'{len(output_files)} images to process.')
            dream_dirname = os.path.join(args.output, file_basename + '-dream') 
            os.makedirs(dream_dirname, exist_ok=True)
            prev_filename = None # Used for blend feedback
            prev_dream_filename = None
            if args.blend > 0.0: # Need to pass back over first image if blending
                output_files.append(output_files[0])
            for o in output_files:
                infile = os.path.join(output_dirname,o)
                if args.blend > 0.0 and prev_dream_filename is not None:
                    original_img, img_size = blend_img(infile, prev_dream_filename, prev_filename,
                        blend_amount=args.blend,
                        diff = args.diff,
                        max_dim=args.max_size)
                else:
                    original_img, img_size = load_img(infile, max_dim=args.max_size)
                dream_filename = os.path.join(dream_dirname, o)
                print(f'  {infile} -> {dream_filename}')
                if args.mode == 'simple':
                    dream_img = run_deep_dream_simple(original_img, dream_model, steps=args.steps, step_size=args.step_size)
                elif args.mode == 'octaves':
                    dream_img = run_deep_dream_with_octaves(original_img, dream_model,
                        steps_per_octave=args.steps,
                        step_size=args.step_size,
                        octave_scale = args.scale,
                        tile_size = args.tile_size if args.tile_size is not None else max(img_size))
                else:
                    raise RuntimeError(f'Unrecognized mode {args.mode}')
                save_img(dream_img, dream_filename)
                prev_dream_filename = dream_filename
                prev_filename = infile
            print('Rendering output file ...')
            assembled_output = concat_png_sequence(args.input, dream_dirname, args.output)
            print(f'Output rendered to {assembled_output}')
        else:
            print(f'Unsupported mimetype {mime}/{subtype}')
    except Exception:
        print(traceback.format_exc())