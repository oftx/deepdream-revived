import argparse
import random
import mimetypes
import os
import subprocess
import traceback
import sys
import concurrent.futures # Added for threading

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
import tensorflow
import numpy as np
import PIL.Image
import PIL.ImageChops
from tensorflow.keras import layers, models
import imageio

# --- Global Configuration ---
ffmpeg_path = None
ffprobe_path = ffmpeg_path
ffmpeg_exe = os.path.join(ffmpeg_path, 'ffmpeg') if ffmpeg_path else 'ffmpeg'
ffprobe_exe = os.path.join(ffprobe_path, 'ffprobe') if ffprobe_path else 'ffprobe'

# --- Image Loading and Preprocessing ---
def load_img(image_path: str, max_dim=None):
    img = PIL.Image.open(image_path).convert('RGB')
    original_size = img.size
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img), original_size

def blend_pil_images(next_image_pil: PIL.Image.Image,
                     prev_image_pil: PIL.Image.Image,
                     prev_dream_image_pil: PIL.Image.Image,
                     blend_amount=0.0, diff=False, max_dim=None):
    img_pil = next_image_pil.convert('RGB')
    dream_pil = prev_dream_image_pil.convert('RGB')

    if diff:
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
    img_tensor = 255 * (img_tensor + 1.0) / 2.0
    return tensorflow.cast(img_tensor, tensorflow.uint8)

def calc_loss(img_tensor, model):
    img_batch = tensorflow.expand_dims(img_tensor, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]
    losses = [tensorflow.math.reduce_mean(act) for act in layer_activations]
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
        for _ in tensorflow.range(steps):
            with tensorflow.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.model)
            gradients = tape.gradient(loss, img)
            gradients /= tensorflow.math.reduce_std(gradients) + 1e-8
            img = img + gradients * step_size
            img = tensorflow.clip_by_value(img, -1, 1)
        return loss, img

def random_roll(img, maxroll):
    shift = tensorflow.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tensorflow.int32)
    return shift, tensorflow.roll(img, shift=shift, axis=[0, 1])

class TiledGradients(tensorflow.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tensorflow.function(
        input_signature=(
            tensorflow.TensorSpec(shape=[None, None, 3], dtype=tensorflow.float32),
            tensorflow.TensorSpec(shape=[2], dtype=tensorflow.int32),
            tensorflow.TensorSpec(shape=[], dtype=tensorflow.int32),)
    )
    def __call__(self, img, img_size_tf, tile_size=512):
        shift, img_rolled = random_roll(img, tile_size)
        gradients = tensorflow.zeros_like(img_rolled)
        
        xs = tensorflow.range(0, img_size_tf[1], tile_size)[:-1]
        if not tensorflow.cast(len(xs), bool): xs = tensorflow.constant([0])
        ys = tensorflow.range(0, img_size_tf[0], tile_size)[:-1]
        if not tensorflow.cast(len(ys), bool): ys = tensorflow.constant([0])

        for y_offset in ys:
            for x_offset in xs:
                with tensorflow.GradientTape() as tape:
                    tape.watch(img_rolled)
                    img_tile = img_rolled[y_offset:y_offset + tile_size, x_offset:x_offset + tile_size]
                    loss = calc_loss(img_tile, self.model)
                gradients += tape.gradient(loss, img_rolled)

        gradients = tensorflow.roll(gradients, shift=-shift, axis=[0, 1])
        gradients /= tensorflow.math.reduce_std(gradients) + 1e-8
        return gradients

def run_deep_dream_simple(img_np, dream_model, steps=100, step_size=0.01, frame_info=""):
    if img_np.shape[-1] == 4: img_np = img_np[..., :3]
    
    img_tf = tensorflow.keras.applications.inception_v3.preprocess_input(img_np.astype(np.float32))
    img_tf = tensorflow.convert_to_tensor(img_tf)
    deepdream_obj = DeepDream(dream_model)
    
    steps_remaining, current_step = steps, 0
    while steps_remaining > 0:
        run_steps = tensorflow.constant(min(100, steps_remaining))
        loss, img_tf = deepdream_obj(img_tf, run_steps, tensorflow.constant(step_size))
        steps_remaining -= run_steps.numpy()
        current_step += run_steps.numpy()
        print(f"  {frame_info}Step {current_step}/{steps}, loss {loss.numpy():.4f}", end='\r')
        sys.stdout.flush()
    print()
    return deprocess(img_tf).numpy()

def run_deep_dream_with_octaves(img_np, dream_model, steps_per_octave=100, step_size=0.01,
                                octaves=range(-2, 3), octave_scale=1.3, tile_size=512, frame_info=""):
    if img_np.shape[-1] == 4: img_np = img_np[..., :3]
    get_tiled_gradients = TiledGradients(dream_model)
    original_shape_hw = tensorflow.shape(img_np)[:-1]
    img_tf = tensorflow.keras.applications.inception_v3.preprocess_input(img_np.astype(np.float32))
    img_tf = tensorflow.convert_to_tensor(img_tf)

    for i, octave_val in enumerate(octaves):
        print(f"  {frame_info}Octave {i+1}/{len(octaves)} (val: {octave_val})...")
        new_size_hw_float = tensorflow.cast(original_shape_hw, tensorflow.float32) * (octave_scale ** octave_val)
        new_size_hw_int = tensorflow.maximum(tensorflow.cast(new_size_hw_float, tensorflow.int32), [1,1])
        img_tf = tensorflow.image.resize(img_tf, new_size_hw_int, method='lanczos3') # or 'lanczos5'

        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(img_tf, new_size_hw_int, tile_size=tile_size)
            img_tf = img_tf + gradients * step_size
            img_tf = tensorflow.clip_by_value(img_tf, -1, 1)
            print(f"    {frame_info}Octave {i+1}, step {step + 1}/{steps_per_octave}", end='\r')
            sys.stdout.flush()
        print()
    return deprocess(img_tf).numpy()

def save_img_threaded(img_np_uint8, img_name: str):
    """Saves a NumPy array (already uint8) as an image. For threading."""
    try:
        PIL.Image.fromarray(img_np_uint8).save(img_name, 'PNG')
        # print(f"Saved image to {img_name}") # Optionally reduce print frequency
    except Exception as e:
        print(f"Error saving {img_name}: {e}")

# --- FFmpeg/FFprobe Utilities ---
def get_media_info(input_filename: str):
    fps, total_frames = None, None
    cmd = [ffprobe_exe, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate,nb_frames,duration",
           "-of", "default=noprint_wrappers=1:nokey=1", input_filename]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        info = dict(line.split('=', 1) for line in result.stdout.strip().split('\n') if '=' in line)
        if 'r_frame_rate' in info and info['r_frame_rate'] != 'N/A':
            num, den = map(float, info['r_frame_rate'].split('/'))
            if den != 0: fps = round(num / den, 2)
        if 'nb_frames' in info and info['nb_frames'] != 'N/A':
            total_frames = int(info['nb_frames'])
        if total_frames is None and fps and 'duration' in info and info['duration'] != 'N/A':
            total_frames = int(float(info['duration']) * fps)
    except Exception as e:
        print(f"ffprobe error for {input_filename}: {e}")
    if fps is None: print(f"Warning: Could not get framerate for {input_filename}.")
    if total_frames is None: print(f"Warning: Could not get total frames for {input_filename}.")
    return fps, total_frames

def concat_png_sequence_to_video(png_dir: str, output_video_path: str, framerate: float, input_video_is_gif: bool):
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    sample_png_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png') and os.path.splitext(f)[0].isdigit()],
                               key=lambda f: int(os.path.splitext(f)[0]))
    if not sample_png_files:
        print(f"No PNG files found in {png_dir} to concatenate.")
        return None
    
    num_digits = len(os.path.splitext(sample_png_files[0])[0])
    input_pattern = os.path.join(png_dir, f'%0{num_digits}d.png')
    
    ffmpeg_args = [ffmpeg_exe, '-hide_banner', '-y', '-framerate', str(framerate), '-i', input_pattern]
    if not input_video_is_gif:
        ffmpeg_args.extend(['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20'])
    ffmpeg_args.append(output_video_path)
    
    print(f"Running ffmpeg to concatenate frames: {' '.join(ffmpeg_args)}")
    try:
        subprocess.run(ffmpeg_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, encoding='utf-8', errors='ignore')
        print(f"Output video rendered to {output_video_path}")
        return output_video_path
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error during concatenation: {e.stderr}")
        raise RuntimeError(f'ffmpeg returned error code {e.returncode}') from e

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Applies DeepDream effect to images or videos.")
    parser.add_argument('--blend', type=float, default=0.0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--diff', action='store_true')
    parser.add_argument('-i', '--input', type=str, default='example.png')
    parser.add_argument('--max_img_dim', type=int)
    parser.add_argument('--mode', type=str, choices=['simple', 'octaves'], default='simple')
    parser.add_argument('--octaves', type=str, default='-2,0,1,2')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--rand_min', type=int, default=-2)
    parser.add_argument('--rand_max', type=int, default=2)
    parser.add_argument('--scale', type=float, default=1.3)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--tile_size', type=int)
    args = parser.parse_args()

    # --- GPU/CPU Setup ---
    if args.cpu:
        tensorflow.config.set_visible_devices([], 'GPU')
        print('Using CPU only mode.')
    else:
        gpus = tensorflow.config.list_physical_devices('GPU')
        if gpus:
            print(f'Num GPUs Available: {len(gpus)}')
            for gpu in gpus: tensorflow.config.experimental.set_memory_growth(gpu, True)
        else:
            print('No GPUs available, using CPU.')

    # --- Model Setup ---
    base_model = tensorflow.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    selected_layers = [base_model.get_layer(name).output for name in ['mixed3', 'mixed5']]
    dream_model = tensorflow.keras.Model(inputs=base_model.input, outputs=selected_layers)

    # --- Octave Parsing ---
    parsed_octaves = [-2, 0, 1, 2] # Default
    if 'random' in args.octaves:
        try: parsed_octaves = [random.randint(args.rand_min, args.rand_max) for _ in range(int(args.octaves.split(' ')[-1]))]
        except: print(f"Invalid 'random' octaves: {args.octaves}. Using default.")
    elif 'range' in args.octaves:
        try: _, lower, upper = args.octaves.split(' '); parsed_octaves = list(range(int(lower), int(upper) + 1))
        except: print(f"Invalid 'range' octaves: {args.octaves}. Using default.")
    else:
        try: parsed_octaves = [int(s.strip()) for s in args.octaves.split(',')]
        except: print(f"Invalid octaves list: {args.octaves}. Using default.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Input Type Detection ---
    input_type = mimetypes.guess_type(args.input)[0]
    is_gif_input, mime_main = False, 'unknown'
    if input_type:
        mime_main, mime_subtype = input_type.split('/')
        is_gif_input = (mime_main == 'image' and mime_subtype == 'gif')
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        mime_main = 'video'
    else: # Default to image if not video extension and no mime type
        mime_main = 'image'


    # --- Single Image Processing ---
    if mime_main == 'image' and not is_gif_input:
        print(f"Processing single image: {args.input}")
        original_img_np, img_pil_size = load_img(args.input, max_dim=args.max_img_dim)
        
        current_tile_size = args.tile_size
        if args.mode == 'octaves' and current_tile_size is None:
            current_tile_size = max(img_pil_size) if img_pil_size else 512

        if args.mode == 'simple':
            dream_img_np = run_deep_dream_simple(original_img_np, dream_model, args.steps, args.step_size)
        else: # octaves
            dream_img_np = run_deep_dream_with_octaves(original_img_np, dream_model, args.steps, args.step_size,
                                                       parsed_octaves, args.scale, current_tile_size)
        
        output_basename = os.path.splitext(os.path.basename(args.input))[0]
        output_filename = os.path.join(args.output_dir, output_basename + '_dream.png')
        PIL.Image.fromarray(dream_img_np).save(output_filename, 'PNG') # Direct save for single image
        print(f"Saved image to {output_filename}")

    # --- Video or GIF Processing with Temp Frames and Threaded Save ---
    elif mime_main == 'video' or is_gif_input:
        print(f"Processing video/GIF: {args.input}")
        file_basename = os.path.splitext(os.path.basename(args.input))[0]
        
        dream_frames_temp_dir = os.path.join(args.output_dir, file_basename + "_dream_frames_temp")
        os.makedirs(dream_frames_temp_dir, exist_ok=True)
        
        output_extension = '.gif' if is_gif_input and args.blend == 0 else '.mp4'
        output_video_path = os.path.join(args.output_dir, file_basename + '_dream' + output_extension)

        prev_dream_pil, prev_original_pil, first_original_pil_for_blend_loop = None, None, None
        
        try:
            reader = imageio.get_reader(args.input)
            meta_data = reader.get_meta_data()
            fps = meta_data.get('fps', 25)
            num_frames_total = meta_data.get('nframes')
            if not isinstance(num_frames_total, int) or num_frames_total <= 0 : # float('inf') or 0
                _, fallback_frames = get_media_info(args.input)
                if fallback_frames: num_frames_total = fallback_frames
                else: num_frames_total = -1 # Indicate unknown for progress display

            start_frame_index = 0
            existing_dream_files = sorted([f for f in os.listdir(dream_frames_temp_dir) if f.endswith('.png') and f.replace('.png','').isdigit()])
            if existing_dream_files:
                last_processed_frame_name = existing_dream_files[-1]
                try:
                    start_frame_index = int(os.path.splitext(last_processed_frame_name)[0]) # Assumes 1-based names like "00001.png"
                    print(f"Resuming from frame {start_frame_index + 1}")
                    if args.blend > 0.0:
                        prev_dream_pil = PIL.Image.open(os.path.join(dream_frames_temp_dir, last_processed_frame_name)).convert('RGB')
                        # For prev_original_pil on resume, it's complex. Simplification: it will be set in the loop.
                        # This means the first blended frame on resume might not use the *actual* previous original.
                except ValueError:
                    print(f"Could not parse frame number from {last_processed_frame_name}. Restarting.")
                    start_frame_index = 0
            
            num_digits_for_filename = len(str(num_frames_total)) if num_frames_total > 0 else 5

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as saving_executor: # 1-2 workers for saving
                future_saves = []

                for i, frame_np in enumerate(reader):
                    if i < start_frame_index:
                        continue # Skip already processed frames based on resume logic
                    
                    frame_info_str = f"Frame {i+1}{f'/{num_frames_total}' if num_frames_total > 0 else ''}: "
                    print(f"  {frame_info_str}Processing...", end='\r')
                    sys.stdout.flush()

                    current_original_pil = PIL.Image.fromarray(frame_np).convert('RGB')
                    if i == start_frame_index and args.blend > 0.0: # Capture first original for loop (even on resume)
                        first_original_pil_for_blend_loop = current_original_pil.copy()

                    img_to_process_np = np.array(current_original_pil)
                    img_pil_size = current_original_pil.size

                    if args.blend > 0.0 and prev_dream_pil and prev_original_pil:
                        img_to_process_np, img_pil_size = blend_pil_images(
                            current_original_pil, prev_original_pil, prev_dream_pil,
                            args.blend, args.diff, args.max_img_dim)
                    elif args.max_img_dim:
                        temp_pil = PIL.Image.fromarray(img_to_process_np)
                        temp_pil.thumbnail((args.max_img_dim, args.max_img_dim))
                        img_to_process_np = np.array(temp_pil); img_pil_size = temp_pil.size
                    
                    current_tile_size = args.tile_size
                    if args.mode == 'octaves' and current_tile_size is None:
                        current_tile_size = max(img_pil_size) if img_pil_size else 512
                    
                    dream_frame_np = None
                    try:
                        if args.mode == 'simple':
                            dream_frame_np = run_deep_dream_simple(img_to_process_np, dream_model, args.steps, args.step_size, frame_info_str)
                        else: # octaves
                            dream_frame_np = run_deep_dream_with_octaves(img_to_process_np, dream_model, args.steps,
                                                                         args.step_size, parsed_octaves, args.scale,
                                                                         current_tile_size, frame_info_str)
                    except Exception as e:
                        print(f"\nError processing {frame_info_str}: {e}. Skipping.")
                        prev_dream_pil = None # Reset blending for next frame
                        prev_original_pil = current_original_pil
                        continue
                    
                    temp_frame_filename = f"{str(i+1).zfill(num_digits_for_filename)}.png"
                    temp_frame_path = os.path.join(dream_frames_temp_dir, temp_frame_filename)
                    
                    # Submit save task to thread pool
                    frame_to_save = dream_frame_np.astype(np.uint8).copy() # Ensure data is copied for thread
                    future = saving_executor.submit(save_img_threaded, frame_to_save, temp_frame_path)
                    future_saves.append(future)
                    
                    prev_dream_pil = PIL.Image.fromarray(dream_frame_np.astype(np.uint8))
                    prev_original_pil = current_original_pil
                
                # Handle seamless blending loop frame (if applicable)
                if args.blend > 0.0 and prev_dream_pil and first_original_pil_for_blend_loop and prev_original_pil:
                    print("  Processing final blend loop frame...")
                    img_to_process_np, img_pil_size = blend_pil_images(
                        first_original_pil_for_blend_loop, prev_original_pil, prev_dream_pil,
                        args.blend, args.diff, args.max_img_dim)
                    
                    current_tile_size = args.tile_size
                    if args.mode == 'octaves' and current_tile_size is None:
                        current_tile_size = max(img_pil_size) if img_pil_size else 512

                    frame_info_str = "Final Blend Frame: "
                    if args.mode == 'simple':
                        final_blend_dream_np = run_deep_dream_simple(img_to_process_np, dream_model, args.steps, args.step_size, frame_info_str)
                    else:
                        final_blend_dream_np = run_deep_dream_with_octaves(img_to_process_np, dream_model, args.steps,
                                                                            args.step_size, parsed_octaves, args.scale,
                                                                            current_tile_size, frame_info_str)
                    
                    # This frame will be part of the sequence for concatenation
                    # Its name should follow sequence or be handled by concat_png_sequence_to_video if it's special
                    # For simplicity, let's assume it's the "next" frame number if it were a loop
                    final_blend_filename = f"{str(i+2).zfill(num_digits_for_filename)}_blendloop.png" # Needs careful naming for concat
                    final_blend_path = os.path.join(dream_frames_temp_dir, final_blend_filename)
                    frame_to_save = final_blend_dream_np.astype(np.uint8).copy()
                    future = saving_executor.submit(save_img_threaded, frame_to_save, final_blend_path)
                    future_saves.append(future)
                    print("  Final blend loop frame submitted for saving.")

                print("\nWaiting for all frame save operations to complete...")
                for future in concurrent.futures.as_completed(future_saves):
                    try: future.result() # Check for exceptions from save_img_threaded
                    except Exception as exc: print(f'A save task failed: {exc}')
            
            reader.close()
            print("All frames processed and saved to temporary directory.")
            concat_png_sequence_to_video(dream_frames_temp_dir, output_video_path, fps, is_gif_input)
            # print(f"Consider deleting temporary directory: {dream_frames_temp_dir}")

        except Exception as e:
            print(f"Error during video/GIF processing: {e}")
            print(traceback.format_exc())
            print(f"Temporary frames (if any) are in: {dream_frames_temp_dir}")
    
    else:
        print(f"Unsupported input type: {mime_main}. Please provide an image or video file.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        print(traceback.format_exc())