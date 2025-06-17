import argparse
import random
import mimetypes
import os
import subprocess
import traceback
import sys # For flushing stdout

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Suppress verbose TensorFlow logging
os.environ["AUTOGRAPH_VERBOSITY"] = "0" # Suppress autograph verbosity
import tensorflow
import numpy as np
import PIL.Image
import PIL.ImageChops
from tensorflow.keras import layers, models
import imageio # For efficient video processing

# --- Global Configuration ---
ffmpeg_path = None # Edit this if you want to specify a custom path to ffmpeg
ffprobe_path = ffmpeg_path # Usually you should not have to edit this
ffmpeg_exe = os.path.join(ffmpeg_path, 'ffmpeg') if ffmpeg_path else 'ffmpeg'
ffprobe_exe = os.path.join(ffprobe_path, 'ffprobe') if ffprobe_path else 'ffprobe'

# --- Image Loading and Preprocessing ---
def download(url, max_dim=None):
    """Downloads an image and reads it into a NumPy array."""
    name = url.split('/')[-1]
    image_path = tensorflow.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path).convert('RGB') # Ensure RGB
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)

def load_img(image_path: str, max_dim=None):
    """Loads an image from a path into a NumPy array and returns its original size."""
    img = PIL.Image.open(image_path).convert('RGB') # Ensure RGB
    original_size = img.size
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img), original_size # Return np.array and original PIL size

def blend_pil_images(next_image_pil: PIL.Image.Image,
                     prev_image_pil: PIL.Image.Image,
                     prev_dream_image_pil: PIL.Image.Image,
                     blend_amount=0.0, diff=False, max_dim=None):
    """
    Blends three PIL Images.
    next_image_pil: The current original frame.
    prev_image_pil: The previous original frame (used for diff).
    prev_dream_image_pil: The previously dreamed image.
    """
    img_pil = next_image_pil.convert('RGB')
    dream_pil = prev_dream_image_pil.convert('RGB')

    if diff:
        # prev_original_pil = PIL.Image.open(prev_image_path).convert('RGB')
        prev_original_pil = prev_image_pil.convert('RGB')
        if prev_original_pil.size != dream_pil.size:
            prev_original_pil = prev_original_pil.resize(dream_pil.size, PIL.Image.LANCZOS)
        diff_img = PIL.ImageChops.difference(dream_pil, prev_original_pil)
        dream_pil_to_blend = diff_img
    else:
        dream_pil_to_blend = dream_pil

    if img_pil.size != dream_pil_to_blend.size:
        dream_pil_to_blend = dream_pil_to_blend.resize(img_pil.size, PIL.Image.LANCZOS)

    blended_pil = PIL.Image.blend(img_pil, dream_pil_to_blend, blend_amount)
    if max_dim:
        blended_pil.thumbnail((max_dim, max_dim))
    return np.array(blended_pil), blended_pil.size


# --- TensorFlow and DeepDream Core ---
def deprocess(img_tensor):
    """Normalize an image tensor from [-1,1] to [0,255] uint8."""
    img_tensor = 255 * (img_tensor + 1.0) / 2.0
    return tensorflow.cast(img_tensor, tensorflow.uint8)

def calc_loss(img_tensor, model):
    """Pass forward the image through the model to retrieve activations and calculate loss."""
    img_batch = tensorflow.expand_dims(img_tensor, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1: # Handle models with single output layer
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tensorflow.math.reduce_mean(act)
        losses.append(loss)
    return tensorflow.reduce_sum(losses)

class DeepDream(tensorflow.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tensorflow.function(
        input_signature=(
            tensorflow.TensorSpec(shape=[None, None, 3], dtype=tensorflow.float32),
            tensorflow.TensorSpec(shape=[], dtype=tensorflow.int32),
            tensorflow.TensorSpec(shape=[], dtype=tensorflow.float32),)
    )
    def __call__(self, img, steps, step_size):
        loss = tensorflow.constant(0.0)
        for _ in tensorflow.range(steps): # Use _ if n is not used
            with tensorflow.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.model)
            gradients = tape.gradient(loss, img)
            gradients /= tensorflow.math.reduce_std(gradients) + 1e-8
            img = img + gradients * step_size
            img = tensorflow.clip_by_value(img, -1, 1)
        return loss, img

def random_roll(img, maxroll):
    """Randomly shift the image to avoid tiled boundaries."""
    shift = tensorflow.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tensorflow.int32)
    img_rolled = tensorflow.roll(img, shift=shift, axis=[0, 1])
    return shift, img_rolled

class TiledGradients(tensorflow.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tensorflow.function(
        input_signature=(
            tensorflow.TensorSpec(shape=[None, None, 3], dtype=tensorflow.float32),
            tensorflow.TensorSpec(shape=[2], dtype=tensorflow.int32), # img_size [height, width]
            tensorflow.TensorSpec(shape=[], dtype=tensorflow.int32),)  # tile_size
    )
    def __call__(self, img, img_size_tf, tile_size=512): # img_size_tf is [height, width]
        shift, img_rolled = random_roll(img, tile_size)
        gradients = tensorflow.zeros_like(img_rolled)
        
        # img_size_tf[1] is width, img_size_tf[0] is height
        xs = tensorflow.range(0, img_size_tf[1], tile_size)[:-1] # Iterate over width (columns)
        if not tensorflow.cast(len(xs), bool):
            xs = tensorflow.constant([0])
        ys = tensorflow.range(0, img_size_tf[0], tile_size)[:-1] # Iterate over height (rows)
        if not tensorflow.cast(len(ys), bool):
            ys = tensorflow.constant([0])

        for y in ys: # Outer loop for rows (height)
            for x in xs: # Inner loop for columns (width)
                with tensorflow.GradientTape() as tape:
                    tape.watch(img_rolled)
                    img_tile = img_rolled[y:y + tile_size, x:x + tile_size]
                    loss = calc_loss(img_tile, self.model)
                gradients += tape.gradient(loss, img_rolled) # Accumulate gradients

        gradients = tensorflow.roll(gradients, shift=-shift, axis=[0, 1])
        gradients /= tensorflow.math.reduce_std(gradients) + 1e-8
        return gradients

def run_deep_dream_simple(img_np, dream_model, steps=100, step_size=0.01):
    """Runs DeepDream in a simple, non-octaved manner."""
    if img_np.shape[-1] == 4: # Handle RGBA
        img_np = img_np[..., :3]
    
    img_tf = tensorflow.keras.applications.inception_v3.preprocess_input(img_np.astype(np.float32))
    img_tf = tensorflow.convert_to_tensor(img_tf)
    
    deepdream_obj = DeepDream(dream_model) # Renamed to avoid conflict
    
    # Process in chunks to avoid OOM for very large step counts, though less critical here
    steps_remaining = steps
    current_step = 0
    while steps_remaining > 0:
        run_steps = tensorflow.constant(min(100, steps_remaining))
        loss, img_tf = deepdream_obj(img_tf, run_steps, tensorflow.constant(step_size))
        steps_remaining -= run_steps.numpy()
        current_step += run_steps.numpy()
        print(f"  Step {current_step}/{steps}, loss {loss.numpy():.4f}", end='\r')
        sys.stdout.flush()
    print() # Newline after progress
    result_tensor = deprocess(img_tf)
    return result_tensor.numpy()

def run_deep_dream_with_octaves(img_np, dream_model, steps_per_octave=100, step_size=0.01,
                                octaves=range(-2, 3), octave_scale=1.3, tile_size=512):
    """Runs DeepDream with octaves and tiled gradients."""
    if img_np.shape[-1] == 4: # Handle RGBA
        img_np = img_np[..., :3]

    get_tiled_gradients = TiledGradients(dream_model)
    
    original_shape_hw = tensorflow.shape(img_np)[:-1] # Height, Width
    img_tf = tensorflow.keras.applications.inception_v3.preprocess_input(img_np.astype(np.float32))
    img_tf = tensorflow.convert_to_tensor(img_tf)

    for i, octave_val in enumerate(octaves): # Use octave_val as it might not be a simple index
        print(f"  Processing octave {i+1}/{len(octaves)} (value: {octave_val})...")
        new_size_hw_float = tensorflow.cast(original_shape_hw, tensorflow.float32) * (octave_scale ** octave_val)
        new_size_hw_int = tensorflow.cast(new_size_hw_float, tensorflow.int32)
        
        # Ensure new_size is at least 1x1
        new_size_hw_int = tensorflow.maximum(new_size_hw_int, tensorflow.constant([1,1], dtype=tensorflow.int32))

        img_tf = tensorflow.image.resize(img_tf, new_size_hw_int, method=PIL.Image.LANCZOS) # Use LANCZOS for quality

        for step in range(steps_per_octave):
            # Pass new_size_hw_int (height, width) to get_tiled_gradients
            gradients = get_tiled_gradients(img_tf, new_size_hw_int, tile_size=tile_size)
            img_tf = img_tf + gradients * step_size
            img_tf = tensorflow.clip_by_value(img_tf, -1, 1)
            print(f"    Octave {i+1}, step {step + 1}/{steps_per_octave}", end='\r')
            sys.stdout.flush()
        print() # Newline after octave steps
    
    # Resize back to original size before deprocessing if needed, though often kept at last octave size
    # img_tf = tensorflow.image.resize(img_tf, original_shape_hw)
    result_tensor = deprocess(img_tf)
    return result_tensor.numpy()

def save_img(img_np, img_name: str):
    """Saves a NumPy array as an image."""
    PIL.Image.fromarray(img_np).save(img_name, 'PNG')
    print(f"Saved image to {img_name}")

# --- FFmpeg/FFprobe Utilities ---
def get_media_info(input_filename: str):
    """Gets framerate and total frames using ffprobe."""
    fps = None
    total_frames = None

    # Get r_frame_rate and nb_frames
    cmd = [
        ffprobe_exe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,nb_frames,duration",
        "-of", "default=noprint_wrappers=1:nokey=1", input_filename
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        info = {}
        for line in lines:
            if '=' in line:
                key, value = line.split('=', 1)
                info[key] = value

        # Parse r_frame_rate
        if 'r_frame_rate' in info and info['r_frame_rate'] != 'N/A':
            num, den = map(float, info['r_frame_rate'].split('/'))
            if den != 0:
                fps = round(num / den, 2)

        # Parse nb_frames
        if 'nb_frames' in info and info['nb_frames'] != 'N/A':
            total_frames = int(info['nb_frames'])
        
        # Fallback for total_frames if nb_frames is N/A but duration and fps are available
        if total_frames is None and fps is not None and 'duration' in info and info['duration'] != 'N/A':
            duration = float(info['duration'])
            total_frames = int(duration * fps)

    except subprocess.CalledProcessError as e:
        print(f"ffprobe error getting media info for {input_filename}: {e.stderr}")
    except Exception as e:
        print(f"Error parsing ffprobe output for {input_filename}: {e}")

    if fps is None: print(f"Warning: Could not determine framerate for {input_filename}. Will try imageio default or user must specify.")
    if total_frames is None: print(f"Warning: Could not determine total frames for {input_filename}.")
    return fps, total_frames


def output_to_png_sequence(input_filename: str, output_dir: str, desired_fps=None):
    """Extracts frames from video to a PNG sequence with proper zero-padding and sorting."""
    os.makedirs(output_dir, exist_ok=True)
    
    source_fps, total_frames = get_media_info(input_filename)
    
    output_pattern = '%03d.png' # Default
    if total_frames and total_frames > 0:
        num_digits = max(3, len(str(total_frames)))
        output_pattern = f'%0{num_digits}d.png'
        print(f"Source has ~{total_frames} frames. Using pattern: {output_pattern} for ffmpeg.")
    else:
        print(f"Could not determine total frames. Using fallback pattern: {output_pattern}. Numerical sort will be critical.")

    ffmpeg_args = [ffmpeg_exe, '-hide_banner', '-y', '-i', input_filename]
    
    # Use desired_fps if provided, otherwise source_fps if available for -vf fps filter
    # Note: This changes the number of output frames if desired_fps != source_fps
    effective_fps_for_vf = desired_fps if desired_fps is not None else source_fps
    if effective_fps_for_vf is not None:
         ffmpeg_args.extend(['-vf', f'fps={effective_fps_for_vf}'])

    ffmpeg_args.append(os.path.join(output_dir, output_pattern))
    
    print(f"Running ffmpeg to extract frames: {' '.join(ffmpeg_args)}")
    try:
        result = subprocess.run(ffmpeg_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, encoding='utf-8', errors='ignore')
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error during frame extraction:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        raise RuntimeError(f'ffmpeg returned error code {e.returncode}') from e
    
    png_files_on_disk = [f for f in os.listdir(output_dir) if f.endswith('.png') and os.path.splitext(f)[0].isdigit()]
    
    # Crucial: Sort numerically
    sorted_png_files = sorted(png_files_on_disk, key=lambda f: int(os.path.splitext(f)[0]))
    print(f"Extracted {len(sorted_png_files)} frames to {output_dir}.")
    return sorted_png_files, (desired_fps or source_fps or 25) # Return fps used for extraction or a default

def concat_png_sequence_to_video(png_dir: str, output_video_path: str, framerate: float, input_video_is_gif: bool):
    """Concatenates a PNG sequence into a video or GIF."""
    # Ensure output directory exists for the video
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Use the number of digits from the first file found (assume consistency)
    # This helps ffmpeg pick up the sequence correctly, especially if leading zeros change
    sample_png_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png') and os.path.splitext(f)[0].isdigit()],
                               key=lambda f: int(os.path.splitext(f)[0]))
    if not sample_png_files:
        print(f"No PNG files found in {png_dir} to concatenate.")
        return None
    
    first_file_basename = os.path.splitext(sample_png_files[0])[0]
    num_digits = len(first_file_basename)
    input_pattern = os.path.join(png_dir, f'%0{num_digits}d.png')
    
    # FFMPEG uses start_number 1 by default for %d if not specified.
    # If your files are 0-indexed, you might need -start_number 0
    # For simplicity, we assume 1-indexed or that ffmpeg handles it from the pattern.

    ffmpeg_args = [ffmpeg_exe, '-hide_banner', '-y', '-framerate', str(framerate),
                   '-i', input_pattern] # Using specific pattern

    if not input_video_is_gif: # Output to MP4
        ffmpeg_args.extend(['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20'])
    # Else, output will be GIF if output_video_path ends with .gif (ffmpeg infers from extension)
    
    ffmpeg_args.append(output_video_path)
    
    print(f"Running ffmpeg to concatenate frames: {' '.join(ffmpeg_args)}")
    try:
        result = subprocess.run(ffmpeg_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, encoding='utf-8', errors='ignore')
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error during concatenation:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        raise RuntimeError(f'ffmpeg returned error code {e.returncode}') from e

    print(f"Output video rendered to {output_video_path}")
    return output_video_path

# --- Main Execution ---
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(prog='DeepDream Runner', description="Applies DeepDream effect to images or videos.")
        parser.add_argument('--blend', type=float, default=0.0, help='Amount to blend images when processing video (0.0 to 1.0). Default: 0.0')
        parser.add_argument('--cpu', action='store_true', help="Use TensorFlow in CPU only mode.")
        parser.add_argument('--diff', action='store_true', help="Enable calculating the difference between AI noise and original image when blending images for video.")
        parser.add_argument('-i', '--input', type=str, default='example.png', help='Input file to process (image or video). Default: example.png')
        parser.add_argument('--max_img_dim', type=int, help='Maximum dimension (width or height) for processing images. Images will be resized if larger. Default: no limit.')
        parser.add_argument('--mode', type=str, choices=['simple', 'octaves'], default='simple', help='DeepDream processing method. Default: simple')
        parser.add_argument('--octaves', type=str, default='-2,0,1,2', help='List of octaves for "octaves" mode. Comma-separated integers (e.g., "-2,0,1"), "random N", or "range start end". Default: "-2,0,1,2"')
        parser.add_argument('--output_dir', type=str, default='output', help='Output directory. Default: output')
        parser.add_argument('--rand_min', type=int, default=-2, help='Lower bound for random octaves. Default: -2')
        parser.add_argument('--rand_max', type=int, default=2, help='Upper bound for random octaves. Default: 2')
        parser.add_argument('--scale', type=float, default=1.3, help='Scale factor between octaves in "octaves" mode. Default: 1.3')
        parser.add_argument('--steps', type=int, default=20, help='Total steps for "simple" mode, or steps per octave for "octaves" mode. Default: 20')
        parser.add_argument('--step_size', type=float, default=0.01, help='Step size for gradient ascent. Default: 0.01')
        parser.add_argument('--tile_size', type=int, help='Tile size for "octaves" mode. If not specified, uses the max dimension of the input image.')

        args = parser.parse_args()

        if args.cpu:
            tensorflow.config.set_visible_devices([], 'GPU')
            print('Using CPU only mode.')
        else:
            gpus = tensorflow.config.list_physical_devices('GPU')
            if gpus:
                print(f'Num GPUs Available: {len(gpus)}')
                try: # Set memory growth to True for all GPUs
                    for gpu in gpus:
                        tensorflow.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"Could not set memory growth: {e}") # May happen if already set or in use
            else:
                print('No GPUs available, using CPU.')

        # --- Model Setup ---
        base_model = tensorflow.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        layer_names = ['mixed3', 'mixed5'] # Layers to maximize activations
        selected_layers = [base_model.get_layer(name).output for name in layer_names]
        dream_model = tensorflow.keras.Model(inputs=base_model.input, outputs=selected_layers)
        print(f"Dream model created using layers: {layer_names}")

        print(f'Using DeepDream mode "{args.mode}", steps: {args.steps}, step size: {args.step_size}')
        
        parsed_octaves = []
        if 'random' in args.octaves:
            try:
                num_octaves_str = args.octaves.split(' ')[-1]
                num_octaves = int(num_octaves_str)
                parsed_octaves = [random.randint(args.rand_min, args.rand_max) for _ in range(num_octaves)]
            except (ValueError, IndexError):
                print(f"Error: Invalid format for 'random' octaves: '{args.octaves}'. Expected 'random N'. Using default.")
                parsed_octaves = [-2, 0, 1, 2] # Fallback
        elif 'range' in args.octaves:
            try:
                _, lower_str, upper_str = args.octaves.split(' ')
                parsed_octaves = list(range(int(lower_str), int(upper_str) + 1)) # Inclusive range
            except (ValueError, IndexError):
                print(f"Error: Invalid format for 'range' octaves: '{args.octaves}'. Expected 'range start end'. Using default.")
                parsed_octaves = [-2, 0, 1, 2] # Fallback
        else:
            try:
                parsed_octaves = [int(num_str.strip()) for num_str in args.octaves.split(',')]
            except ValueError:
                print(f"Error: Invalid format for octaves list: '{args.octaves}'. Expected comma-separated integers. Using default.")
                parsed_octaves = [-2, 0, 1, 2] # Fallback
        
        if args.mode == 'octaves':
            print(f'Octaves to process: {parsed_octaves}')

        os.makedirs(args.output_dir, exist_ok=True)
        
        input_type = mimetypes.guess_type(args.input)[0]
        is_gif_input = False
        if input_type:
            mime_main, mime_subtype = input_type.split('/')
            is_gif_input = (mime_main == 'image' and mime_subtype == 'gif')
        else:
            print(f"Warning: Could not determine MIME type for input {args.input}. Assuming image if not video extension.")
            if args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                mime_main, mime_subtype = 'video', 'unknown'
            else:
                mime_main, mime_subtype = 'image', 'unknown'


        # --- Single Image Processing ---
        if mime_main == 'image' and not is_gif_input:
            print(f"Processing single image: {args.input}")
            original_img_np, img_pil_size = load_img(args.input, max_dim=args.max_img_dim)
            
            # Determine tile_size for octaves mode
            current_tile_size = args.tile_size
            if args.mode == 'octaves' and current_tile_size is None:
                 # img_pil_size is (width, height)
                current_tile_size = max(img_pil_size) if img_pil_size else 512
                print(f"Tile size not specified for octaves mode, using max image dimension: {current_tile_size}")


            if args.mode == 'simple':
                dream_img_np = run_deep_dream_simple(original_img_np, dream_model, steps=args.steps, step_size=args.step_size)
            elif args.mode == 'octaves':
                dream_img_np = run_deep_dream_with_octaves(original_img_np, dream_model,
                    steps_per_octave=args.steps, step_size=args.step_size,
                    octaves=parsed_octaves, octave_scale=args.scale,
                    tile_size=current_tile_size)
            else: # Should not happen due to choices in argparser
                raise RuntimeError(f'Unrecognized mode {args.mode}')
            
            output_basename = os.path.splitext(os.path.basename(args.input))[0]
            output_filename = os.path.join(args.output_dir, output_basename + '_dream.png')
            save_img(dream_img_np, output_filename)

        # --- Video or GIF Processing (Optimized with imageio) ---
        elif mime_main == 'video' or is_gif_input:
            print(f"Processing video/GIF: {args.input} using imageio stream.")
            file_basename = os.path.splitext(os.path.basename(args.input))[0]
            
            # Determine output path and type
            output_extension = '.gif' if is_gif_input and args.blend == 0 else '.mp4' # Prefer mp4 if blending, or if input is video
            output_video_path = os.path.join(args.output_dir, file_basename + '_dream' + output_extension)

            prev_dream_pil = None
            prev_original_pil = None
            first_original_pil_for_blend_loop = None # For looping blend

            try:
                reader = imageio.get_reader(args.input)
                meta_data = reader.get_meta_data()
                fps = meta_data.get('fps', 25) # Default to 25 fps if not found
                
                # For GIF, imageio writer might need `duration` (1/fps) per frame instead of global fps
                # For MP4, fps is fine.
                writer_kwargs = {'fps': fps}
                if output_extension == '.gif':
                    writer_kwargs = {'duration': 1.0 / fps, 'loop': 0} # loop 0 for infinite

                with imageio.get_writer(output_video_path, **writer_kwargs) as writer:
                    num_frames = meta_data.get('nframes', float('inf')) # Try to get nframes for progress
                    if num_frames == float('inf') or num_frames == 0 : # ffprobe sometimes returns 0 for nframes of gifs
                        try:
                            _, fallback_frames = get_media_info(args.input)
                            if fallback_frames: num_frames = fallback_frames
                        except: pass


                    for i, frame_np in enumerate(reader):
                        print(f"  Processing frame {i + 1}" + (f"/{num_frames}" if num_frames != float('inf') else ""), end='\r')
                        sys.stdout.flush()

                        current_original_pil = PIL.Image.fromarray(frame_np).convert('RGB')
                        
                        if i == 0 and args.blend > 0.0:
                            first_original_pil_for_blend_loop = current_original_pil.copy()

                        img_to_process_np = np.array(current_original_pil)
                        img_pil_size = current_original_pil.size # (width, height)

                        if args.blend > 0.0 and prev_dream_pil is not None and prev_original_pil is not None:
                            # Use the PIL-based blending function
                            img_to_process_np, img_pil_size = blend_pil_images(
                                current_original_pil,
                                prev_original_pil,
                                prev_dream_pil,
                                blend_amount=args.blend,
                                diff=args.diff,
                                max_dim=args.max_img_dim
                            )
                        elif args.max_img_dim: # No blending, but max_dim applies
                            temp_pil = PIL.Image.fromarray(img_to_process_np)
                            temp_pil.thumbnail((args.max_img_dim, args.max_img_dim))
                            img_to_process_np = np.array(temp_pil)
                            img_pil_size = temp_pil.size

                        # Determine tile_size for octaves mode
                        current_tile_size = args.tile_size
                        if args.mode == 'octaves' and current_tile_size is None:
                            current_tile_size = max(img_pil_size) if img_pil_size else 512
                        
                        if args.mode == 'simple':
                            dream_frame_np = run_deep_dream_simple(img_to_process_np, dream_model, steps=args.steps, step_size=args.step_size)
                        elif args.mode == 'octaves':
                            dream_frame_np = run_deep_dream_with_octaves(img_to_process_np, dream_model,
                                steps_per_octave=args.steps, step_size=args.step_size,
                                octaves=parsed_octaves, octave_scale=args.scale,
                                tile_size=current_tile_size)
                        
                        writer.append_data(dream_frame_np.astype(np.uint8))
                        
                        prev_dream_pil = PIL.Image.fromarray(dream_frame_np.astype(np.uint8))
                        prev_original_pil = current_original_pil
                    print() # Newline after frame processing loop

                    # Handle seamless blending loop: blend last dream frame with first original frame
                    if args.blend > 0.0 and prev_dream_pil is not None and first_original_pil_for_blend_loop is not None and prev_original_pil is not None:
                        print("  Processing final blend loop frame...")
                        img_to_process_np, img_pil_size = blend_pil_images(
                            first_original_pil_for_blend_loop, # 'next' image is the first original
                            prev_original_pil,                 # 'prev' original is the last original
                            prev_dream_pil,                    # 'prev' dream is the last dream
                            blend_amount=args.blend,
                            diff=args.diff,
                            max_dim=args.max_img_dim
                        )
                        current_tile_size = args.tile_size
                        if args.mode == 'octaves' and current_tile_size is None:
                            current_tile_size = max(img_pil_size) if img_pil_size else 512

                        if args.mode == 'simple':
                            final_blend_dream_np = run_deep_dream_simple(img_to_process_np, dream_model, steps=args.steps, step_size=args.step_size)
                        elif args.mode == 'octaves':
                            final_blend_dream_np = run_deep_dream_with_octaves(img_to_process_np, dream_model,
                                steps_per_octave=args.steps, step_size=args.step_size,
                                octaves=parsed_octaves, octave_scale=args.scale,
                                tile_size=current_tile_size)
                        writer.append_data(final_blend_dream_np.astype(np.uint8))
                        print("  Final blend loop frame processed and added.")

                print(f"Output video/GIF rendered to {output_video_path}")

            except Exception as e:
                print(f"Error during video/GIF processing with imageio: {e}")
                print(traceback.format_exc())
                print("Consider falling back to ffmpeg frame extraction if imageio fails (not implemented as automatic fallback).")
        
        # --- Fallback or alternative PNG sequence method (can be invoked if needed, or by different script parts)
        # This part is mostly for demonstrating the fixed output_to_png_sequence and concat_png_sequence
        # if you were *not* using the imageio streaming method above.
        # Example:
        # if some_condition_for_png_sequence_method:
        #     file_basename = os.path.splitext(os.path.basename(args.input))[0]
        #     temp_png_output_dir = os.path.join(args.output_dir, file_basename + "_frames_temp")
        #     dream_png_output_dir = os.path.join(args.output_dir, file_basename + "_dream_frames_temp")
        #     os.makedirs(dream_png_output_dir, exist_ok=True)

        #     extracted_frame_files, processing_fps = output_to_png_sequence(args.input, temp_png_output_dir)
            
        #     # ... (loop through extracted_frame_files, load, process, save to dream_png_output_dir) ...
        #     # This loop would be similar to the imageio loop but with file I/O

        #     output_video_path = os.path.join(args.output_dir, file_basename + "_dream_from_seq" + ('.gif' if is_gif_input else '.mp4'))
        #     concat_png_sequence_to_video(dream_png_output_dir, output_video_path, processing_fps, is_gif_input)
        #     # Cleanup temp_png_output_dir and dream_png_output_dir afterwards if desired

        else:
            print(f"Unsupported input type: {input_type if input_type else 'Unknown'}. Please provide an image or video file.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(traceback.format_exc())