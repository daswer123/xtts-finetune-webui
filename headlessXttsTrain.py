import argparse
import os
import sys
import tempfile
from pathlib import Path
import shutil
import glob
import subprocess # For running ffmpeg
import math

# Keep relevant imports from the original script
import librosa # For duration check
import numpy as np
import torch
import torchaudio
import traceback
from utils.formatter import format_audio_list, list_audios # Assuming these utils exist and work standalone
from utils.gpt_train import train_gpt # Assuming this util exists and works standalone

from faster_whisper import WhisperModel # Keep for data processing

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Keep utility functions (potentially modified)
def download_file(url, destination):
    # (Keep the original download_file function - it might be used by train_gpt)
    import requests # Keep requests import local to this function if only used here
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file to {destination}")
        return destination
    except Exception as e:
        print(f"Failed to download the file: {e}")
        return None

def clear_gpu_cache():
    if torch.cuda.is_available():
        print("Clearing GPU cache...")
        torch.cuda.empty_cache()
        print("GPU cache cleared.")

# Global variable for the loaded model (needed for inference step)
XTTS_MODEL = None

# --- Modified/New Headless Functions ---

def run_ffmpeg(cmd):
    """Runs an ffmpeg command."""
    try:
        print(f"Running FFmpeg command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors='ignore')
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"FFmpeg Error Output:\n{stderr}")
            raise RuntimeError(f"FFmpeg command failed with exit code {process.returncode}")
        print("FFmpeg command executed successfully.")
        # print(f"FFmpeg Output:\n{stdout}") # Optional: print stdout
        return True
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Please ensure FFmpeg is installed and in your PATH.")
        return False
    except Exception as e:
        print(f"An error occurred while running FFmpeg: {e}")
        traceback.print_exc()
        return False

def get_audio_duration(file_path):
    """Gets the duration of an audio file in seconds."""
    try:
        # Use torchaudio.info for potentially faster metadata reading
        info = torchaudio.info(str(file_path))
        return info.num_frames / info.sample_rate
        # Alternatively, use librosa:
        # y, sr = librosa.load(file_path, sr=None)
        # return librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        # Try ffmpeg as a fallback
        try:
            cmd = ["ffmpeg", "-i", str(file_path), "-f", "null", "-"]
            print(f"Running FFmpeg command for duration: {' '.join(cmd)}")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors='ignore')
            stdout, stderr = process.communicate()
            # Search for duration in stderr
            for line in stderr.splitlines():
                if "Duration:" in line:
                    time_str = line.split("Duration:")[1].split(",")[0].strip()
                    h, m, s = map(float, time_str.split(':'))
                    duration = h * 3600 + m * 60 + s
                    print(f"Found duration via FFmpeg: {duration}")
                    return duration
            print("Could not extract duration using FFmpeg.")
            return None
        except Exception as ff_e:
            print(f"Error getting duration using FFmpeg for {file_path}: {ff_e}")
            return None


def prepare_audio(input_path, temp_dir, max_duration_minutes=40):
    """
    Prepares the input audio:
    1. Converts to MP3 if it isn't already.
    2. Trims to max_duration_minutes from the middle if it's too long.
    Returns the path to the prepared audio file (in the temp_dir).
    """
    input_path = Path(input_path)
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem
    current_file = input_path
    intermediate_files = [] # Keep track of files to delete later

    # 1. Convert to MP3 if necessary
    if input_path.suffix.lower() != ".mp3":
        print(f"Input file is not MP3 ({input_path.suffix}). Converting to MP3...")
        mp3_path = temp_dir / f"{base_name}_converted.mp3"
        # Basic conversion command, adjust parameters if needed
        cmd_convert = [
            "ffmpeg", "-i", str(current_file),
            "-vn", "-acodec", "libmp3lame", "-q:a", "2", # VBR quality setting
            "-ac", "1", # Force mono? XTTS might prefer mono
            "-ar", "44100", # Standard sample rate before dataset prep resampling
            str(mp3_path)
        ]
        if not run_ffmpeg(cmd_convert):
            print(f"Failed to convert {current_file} to MP3.")
            return None
        current_file = mp3_path
        intermediate_files.append(current_file)
        print(f"Converted to MP3: {current_file}")
    else:
        # Copy the original MP3 to temp dir to avoid modifying original
        temp_mp3_path = temp_dir / f"{base_name}_original.mp3"
        shutil.copy(str(current_file), str(temp_mp3_path))
        current_file = temp_mp3_path
        intermediate_files.append(current_file) # Add even original copy to potential cleanup
        print("Input file is already MP3.")


    # 2. Check duration and trim if necessary
    duration = get_audio_duration(current_file)
    if duration is None:
        print("Could not determine audio duration.")
        # Clean up intermediate files before returning
        for f in intermediate_files:
             if f.exists(): f.unlink()
        return None

    max_duration_seconds = max_duration_minutes * 60
    print(f"Audio duration: {duration:.2f} seconds.")

    if duration > max_duration_seconds:
        print(f"Audio duration ({duration:.2f}s) exceeds maximum ({max_duration_seconds}s). Trimming...")
        trimmed_path = temp_dir / f"{base_name}_trimmed.mp3"

        # Calculate start time for trimming from the middle
        trim_start_seconds = math.floor((duration - max_duration_seconds) / 2)
        # ffmpeg duration is relative to start time
        trim_duration = max_duration_seconds

        cmd_trim = [
            "ffmpeg", "-i", str(current_file),
            "-ss", str(trim_start_seconds),
            "-t", str(trim_duration), # Use -t for duration
            "-c", "copy", # Use stream copy if possible (faster, avoids re-encoding)
            str(trimmed_path)
        ]
        if not run_ffmpeg(cmd_trim):
             # Fallback if copy codec fails (e.g., format issues)
            print("Stream copy failed, attempting re-encode trim...")
            cmd_trim_reencode = [
                "ffmpeg", "-i", str(current_file),
                "-ss", str(trim_start_seconds),
                "-t", str(trim_duration),
                "-vn", "-acodec", "libmp3lame", "-q:a", "2", # Re-encode if needed
                "-ac", "1", "-ar", "44100", # Ensure consistency
                str(trimmed_path)
            ]
            if not run_ffmpeg(cmd_trim_reencode):
                print(f"Failed to trim {current_file}.")
                 # Clean up intermediate files before returning
                for f in intermediate_files:
                     if f.exists(): f.unlink()
                return None

        print(f"Trimmed audio saved to: {trimmed_path}")
        # Clean up intermediate converted file if it exists and is different from trimmed
        for f in intermediate_files:
             if f != trimmed_path and f.exists():
                  print(f"Cleaning up intermediate file: {f}")
                  f.unlink()
        return trimmed_path
    else:
        print("Audio duration is within the limit. No trimming needed.")
        # Return the path to the MP3 in the temp dir
        # No need to clean up intermediate files here as current_file is the one we want
        return current_file

def preprocess_dataset_headless(audio_file_path, language, whisper_model_name, dataset_out_path):
    """Headless version of preprocess_dataset."""
    clear_gpu_cache()
    print(f"\n--- Starting Step 1: Data Processing ---")
    print(f"Using audio file: {audio_file_path}")
    print(f"Language: {language}")
    print(f"Whisper model: {whisper_model_name}")
    print(f"Dataset output path: {dataset_out_path}")

    os.makedirs(dataset_out_path, exist_ok=True)

    train_meta = ""
    eval_meta = ""

    try:
        # Loading Whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32" # Check bf16 support for potential speedup
        # compute_type = "float16" if torch.cuda.is_available() else "float32" # Original
        print(f"Loading Whisper model '{whisper_model_name}' on device '{device}' with compute type '{compute_type}'...")

        # Check if model path exists, download if necessary (logic adapted from FasterWhisper)
        # This part might need adjustment depending on how FasterWhisper handles model downloads/paths
        # Assuming WhisperModel handles download/cache internally based on name
        asr_model = WhisperModel(whisper_model_name, device=device, compute_type=compute_type)
        print("Whisper model loaded.")

        print("Formatting audio list...")
        # Pass the single audio file path in a list
        train_meta, eval_meta, audio_total_size = format_audio_list(
            [str(audio_file_path)],
            asr_model=asr_model,
            target_language=language,
            out_path=dataset_out_path,
            gradio_progress=None # No progress bar
        )
        print("Audio list formatted.")

    except Exception as e:
        print(f"\n---!!! Data processing failed! !!!---")
        traceback.print_exc()
        return f"Data processing failed: {e}", "", ""

    # Clear Whisper model from memory
    del asr_model
    clear_gpu_cache()

    # Check audio duration
    # It seems format_audio_list already handles slicing, so the total size might be > target size.
    # The crucial check is whether any valid segments were created. Let's check if meta files exist.
    if not Path(train_meta).exists() or not Path(eval_meta).exists():
         message = "Data processing failed to create metadata files. The input audio might be silent or too noisy after processing."
         print(f"\n---!!! Data processing error: {message} !!!---")
         # Let's check the reported audio_total_size anyway
         if audio_total_size < 1: # If whisper reported almost no audio
              print("Reported audio size from Whisper was less than 1 second.")
         return message, "", ""
    elif audio_total_size < 120: # Check if total *detected* speech is > 2 mins
        message = f"Warning: The total detected speech duration ({audio_total_size:.2f} seconds) is less than the recommended 120 seconds. Training quality might be affected."
        print(f"\n---!!! Data processing warning: {message} !!!---")
        # Continue anyway, but warn the user


    print(f"Total detected speech size: {audio_total_size:.2f} seconds.")
    print(f"Training metadata file: {train_meta}")
    print(f"Evaluation metadata file: {eval_meta}")
    print(f"--- Step 1: Data Processing Completed ---")
    return "Dataset Processed Successfully!", str(train_meta), str(eval_meta)


def train_model_headless(language, train_csv_path, eval_csv_path, num_epochs, batch_size, grad_acumm, output_path_base, max_audio_length_sec, version="v2.0.2", custom_model=""):
    """Headless version of train_model."""
    clear_gpu_cache()
    print(f"\n--- Starting Step 2: Fine-tuning XTTS ---")
    print(f"Language: {language}")
    print(f"Training CSV Path: {train_csv_path}") # Log original path for reference
    print(f"Evaluation CSV Path: {eval_csv_path}") # Log original path for reference
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Gradient Accumulation: {grad_acumm}")
    print(f"Max Audio Length: {max_audio_length_sec} seconds")
    print(f"Output Path Base: {output_path_base}")
    print(f"XTTS Base Version: {version}")
    print(f"Custom Base Model Path: {'Default' if not custom_model else custom_model}")


    output_path_base = Path(output_path_base)
    run_dir = output_path_base / "run"
    ready_dir = output_path_base / "ready"
    os.makedirs(ready_dir, exist_ok=True) # Ensure ready dir exists

    # Remove previous run dir if exists
    if run_dir.exists():
        print(f"Removing existing training run directory: {run_dir}")
        shutil.rmtree(run_dir)

    # --- Ensure CSV paths are absolute Path objects ---
    train_csv_path_obj = Path(train_csv_path).resolve()
    eval_csv_path_obj = Path(eval_csv_path).resolve()

    # Check for essential input files using resolved paths
    if not train_csv_path_obj.is_file() or not eval_csv_path_obj.is_file():
        print(f"Error: Training CSV ({train_csv_path_obj}) or Evaluation CSV ({eval_csv_path_obj}) file not found.")
        return "Training or Evaluation CSV file not found. Ensure Step 1 completed successfully.", "", "", "", "", ""

    # --- Pass the full, resolved paths for the CSV files ---
    train_csv_full_path = str(train_csv_path_obj)
    eval_csv_full_path = str(eval_csv_path_obj)

    try:
        # Convert seconds to waveform frames (assuming 22050 Hz sample rate used internally by train_gpt)
        max_audio_length_frames = int(max_audio_length_sec * 22050)
        print(f"Max audio length in frames: {max_audio_length_frames}")

        # Call the core training function
        # Pass the FULL paths for train_csv and eval_csv.
        # Pass the BASE output path (e.g., xtts_finetuned_models/Death) resolved to absolute.
        speaker_xtts_path, config_path, _, vocab_file, exp_path, speaker_wav = train_gpt(
            custom_model=custom_model, # Path to custom model or ""
            version=version,           # Base model version
            language=language,
            num_epochs=num_epochs,
            batch_size=batch_size,
            grad_acumm=grad_acumm,
            train_csv=train_csv_full_path,  # Pass FULL absolute path
            eval_csv=eval_csv_full_path,    # Pass FULL absolute path
            output_path=str(output_path_base.resolve()), # Base directory for 'run' and 'ready' folders
            max_audio_length=max_audio_length_frames
        )

        # --- Find the best model checkpoint from the experiment path ---
        exp_path_obj = Path(exp_path) # exp_path is usually output_path_base/run/training_run_XXX
        best_model_path = exp_path_obj / "best_model.pth"
        if not best_model_path.exists():
             # Fallback: Check for models like epoch_X.pth sorted by modification time
             pth_files = sorted(list(exp_path_obj.glob("*.pth")), key=os.path.getmtime, reverse=True) # Get latest first
             if pth_files:
                 # Exclude optimizer checkpoints if they exist
                 model_files = [p for p in pth_files if "optimizer" not in p.name.lower() and "dvae" not in p.name.lower()]
                 if model_files:
                     best_model_path = model_files[0] # Latest model file
                     print(f"Warning: 'best_model.pth' not found. Using latest model checkpoint: {best_model_path}")
                 elif pth_files: # If only optimizer files were found somehow, use the latest of those (less ideal)
                     best_model_path = pth_files[0]
                     print(f"Warning: 'best_model.pth' not found and no other model checkpoints. Using latest .pth file: {best_model_path}")
                 else: # Should not happen if pth_files was not empty, but defensively check
                     raise FileNotFoundError(f"No model checkpoints (best_model.pth or epoch_*.pth) found in {exp_path_obj}")
             else:
                 raise FileNotFoundError(f"No '.pth' model checkpoint found in {exp_path_obj}")


        # Copy the best model to the 'ready' directory as 'unoptimize_model.pth'
        unoptimized_model_target_path = ready_dir / "unoptimize_model.pth"
        print(f"Copying best model checkpoint {best_model_path} to {unoptimized_model_target_path}")
        shutil.copy(str(best_model_path), str(unoptimized_model_target_path))

        # --- Copy other essential files generated by train_gpt to 'ready' directory ---
        # Check if train_gpt already placed files in the final 'ready' dir.

        # Config file
        source_config_path = Path(config_path).resolve()
        intended_config_path = (ready_dir / source_config_path.name).resolve()
        if source_config_path != intended_config_path:
            print(f"Copying config {source_config_path} to {intended_config_path}")
            shutil.copy(str(source_config_path), str(intended_config_path))
        else:
            print(f"Config file already in place: {intended_config_path}")
        final_config_path = intended_config_path # Use the intended path going forward

        # Vocab file
        source_vocab_path = Path(vocab_file).resolve()
        intended_vocab_path = (ready_dir / source_vocab_path.name).resolve()
        if source_vocab_path != intended_vocab_path:
            print(f"Copying vocab {source_vocab_path} to {intended_vocab_path}")
            shutil.copy(str(source_vocab_path), str(intended_vocab_path))
        else:
            print(f"Vocab file already in place: {intended_vocab_path}")
        final_vocab_path = intended_vocab_path # Use the intended path going forward

        # Speaker file (handle potential None)
        final_speaker_xtts_path = None
        if speaker_xtts_path: # Check if a path was returned
             source_speaker_xtts_path = Path(speaker_xtts_path).resolve()
             intended_speaker_xtts_path = (ready_dir / source_speaker_xtts_path.name).resolve()
             if source_speaker_xtts_path.exists():
                 if source_speaker_xtts_path != intended_speaker_xtts_path:
                     print(f"Copying speaker profile {source_speaker_xtts_path} to {intended_speaker_xtts_path}")
                     shutil.copy(str(source_speaker_xtts_path), str(intended_speaker_xtts_path))
                 else:
                     print(f"Speaker file already in place: {intended_speaker_xtts_path}")
                 final_speaker_xtts_path = intended_speaker_xtts_path # Use the intended path
             else:
                 # Try finding it in the experiment path as a fallback
                 exp_path_obj = Path(exp_path) # Define exp_path_obj if not already defined earlier in function
                 alt_speaker_path = exp_path_obj / source_speaker_xtts_path.name
                 if alt_speaker_path.exists():
                     print(f"Copying speaker profile from experiment dir {alt_speaker_path} to {intended_speaker_xtts_path}")
                     shutil.copy(str(alt_speaker_path), str(intended_speaker_xtts_path))
                     final_speaker_xtts_path = intended_speaker_xtts_path
                 else:
                     print(f"Warning: Speaker profile {source_speaker_xtts_path.name} not found in expected location or experiment dir.")
        else:
             print("Warning: No speaker_xtts_path returned from train_gpt.")


        # speaker_wav is the path to the reference audio used during training (often in dataset/wavs)
        # This path should be correct as returned, resolve it for safety.
        final_speaker_wav_path = str(Path(speaker_wav).resolve())

        print("--- Step 2: Fine-tuning Completed ---")
        # Ensure returned paths are strings and absolute for consistency downstream
        return ("Model training done!",
                str(final_config_path), # Already resolved
                str(final_vocab_path), # Already resolved
                str(unoptimized_model_target_path.resolve()),
                str(final_speaker_xtts_path) if final_speaker_xtts_path else None, # Already resolved or None
                final_speaker_wav_path) # Already resolved string

    except Exception as e:
        print(f"\n---!!! Model training failed! !!!---")
        traceback.print_exc()
        return f"Model training failed: {e}", "", "", "", "", ""

def optimize_model_headless(output_path_base):
    """Headless version of optimize_model."""
    print(f"\n--- Starting Step 2.5: Optimizing Model ---")
    print(f"Looking for model in: {output_path_base}")

    output_path = Path(output_path_base)
    ready_dir = output_path / "ready"
    unoptimized_model_path = ready_dir / "unoptimize_model.pth"
    optimized_model_path = ready_dir / "model.pth"

    if not unoptimized_model_path.is_file():
        # Check if optimization already happened
        if optimized_model_path.is_file():
             print("Optimized model already exists. Skipping optimization.")
             return "Model already optimized.", str(optimized_model_path)
        else:
            print(f"Error: Unoptimized model not found at {unoptimized_model_path}")
            return "Unoptimized model not found in ready folder", ""

    try:
        print(f"Loading unoptimized model from {unoptimized_model_path}...")
        # Load to CPU first to avoid potential GPU memory issues if model is large
        checkpoint = torch.load(unoptimized_model_path, map_location=torch.device("cpu"))

        print("Removing optimizer state...")
        if "optimizer" in checkpoint:
             del checkpoint["optimizer"]
        else:
             print("Optimizer state not found in checkpoint.")


        print("Removing DVAE weights...")
        # Check if 'model' key exists before trying to access its keys
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            keys_to_delete = [key for key in checkpoint["model"].keys() if "dvae" in key]
            if keys_to_delete:
                for key in keys_to_delete:
                    del checkpoint["model"][key]
                print(f"Removed {len(keys_to_delete)} DVAE keys.")
            else:
                print("No DVAE keys found to remove.")
        else:
            print("Warning: 'model' key not found or is not a dictionary in the checkpoint. Skipping DVAE removal.")


        print(f"Saving optimized model to {optimized_model_path}...")
        torch.save(checkpoint, optimized_model_path)

        print(f"Removing unoptimized model file: {unoptimized_model_path}")
        os.remove(unoptimized_model_path)

        clear_gpu_cache()
        print(f"--- Step 2.5: Optimization Completed ---")
        return f"Model optimized successfully!", str(optimized_model_path)

    except Exception as e:
        print(f"\n---!!! Model optimization failed! !!!---")
        traceback.print_exc()
        # Don't delete the unoptimized model if saving failed
        return f"Model optimization failed: {e}", ""

def create_reference_wavs(original_ref_wav_path, output_dir, output_basename):
    """Copies the original reference wav and creates 16kHz and 24kHz versions."""
    print("\n--- Creating Reference WAV Files ---")
    original_ref_wav_path = Path(original_ref_wav_path)
    output_dir = Path(output_dir) # Should be the 'ready' directory
    output_dir.mkdir(parents=True, exist_ok=True)

    if not original_ref_wav_path.exists():
        print(f"Error: Original reference WAV not found at {original_ref_wav_path}")
        # Attempt to find *any* wav in the original directory as a fallback?
        wav_dir = original_ref_wav_path.parent
        found_wavs = list(wav_dir.glob("*.wav"))
        if found_wavs:
            original_ref_wav_path = found_wavs[0]
            print(f"Warning: Original path invalid. Using first found WAV as reference: {original_ref_wav_path}")
        else:
            print(f"Error: Cannot find any WAV file in {wav_dir} to use as reference.")
            return False


    # 1. Copy and rename the original reference wav
    final_ref_path = output_dir / f"{output_basename}.wav"
    print(f"Copying original reference {original_ref_wav_path} to {final_ref_path}")
    shutil.copy(str(original_ref_wav_path), str(final_ref_path))

    # 2. Create 16kHz version
    ref_16k_path = output_dir / f"{output_basename}_16000.wav"
    print(f"Creating 16kHz reference WAV: {ref_16k_path}")
    cmd_16k = ["ffmpeg", "-i", str(final_ref_path), "-ar", "16000", "-ac", "1", str(ref_16k_path), "-y"] # Force mono, overwrite
    if not run_ffmpeg(cmd_16k):
        print("Failed to create 16kHz reference WAV.")
        # Continue anyway, maybe user doesn't need it

    # 3. Create 24kHz version (XTTS native rate)
    ref_24k_path = output_dir / f"{output_basename}_24000.wav"
    print(f"Creating 24kHz reference WAV: {ref_24k_path}")
    cmd_24k = ["ffmpeg", "-i", str(final_ref_path), "-ar", "24000", "-ac", "1", str(ref_24k_path), "-y"] # Force mono, overwrite
    if not run_ffmpeg(cmd_24k):
        print("Failed to create 24kHz reference WAV.")
        # Continue anyway

    print("--- Reference WAV Creation Completed ---")
    return True

def load_model_headless(xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker):
    """Headless version of load_model."""
    global XTTS_MODEL
    clear_gpu_cache()
    print(f"\n--- Starting Step 3: Loading Fine-tuned Model ---")
    print(f"Checkpoint: {xtts_checkpoint}")
    print(f"Config: {xtts_config}")
    print(f"Vocab: {xtts_vocab}")
    print(f"Speaker: {xtts_speaker}")


    if not Path(xtts_checkpoint).exists() or not Path(xtts_config).exists() or not Path(xtts_vocab).exists():
         missing = [p for p in [xtts_checkpoint, xtts_config, xtts_vocab] if not Path(p).exists()]
         print(f"Error: Model loading failed. Missing essential files: {missing}")
         return "Model loading failed: Essential files not found."
    if xtts_speaker and not Path(xtts_speaker).exists():
        print(f"Warning: Speaker file {xtts_speaker} not found. Model might load but speaker info will be missing.")
        # Allow loading without speaker file if user proceeds. Inference might need speaker_wav.

    try:
        print("Initializing XTTS model configuration...")
        config = XttsConfig()
        config.load_json(xtts_config)

        print("Initializing XTTS model from configuration...")
        XTTS_MODEL = Xtts.init_from_config(config)

        print("Loading checkpoint and speaker data...")
        XTTS_MODEL.load_checkpoint(
             config,
             checkpoint_path=xtts_checkpoint,
             vocab_path=xtts_vocab,
             speaker_file_path=xtts_speaker if xtts_speaker and Path(xtts_speaker).exists() else None, # Pass None if missing
             use_deepspeed=False
             )

        if torch.cuda.is_available():
            print("Moving model to GPU...")
            XTTS_MODEL.cuda()
        else:
            print("CUDA not available, using CPU.")

        print("--- Step 3: Model Loading Completed ---")
        return "Model Loaded Successfully!"
    except Exception as e:
        print(f"\n---!!! Model loading failed! !!!---")
        traceback.print_exc()
        XTTS_MODEL = None # Ensure model is not partially loaded
        return f"Model loading failed: {e}"


def run_tts_headless(lang, tts_text, speaker_audio_file, output_wav_path, temperature=0.75, length_penalty=1.0, repetition_penalty=5.0, top_k=50, top_p=0.85, sentence_split=True):
    """Headless version of run_tts."""
    print(f"\n--- Starting Step 4: Generating Example TTS ---")
    print(f"Language: {lang}")
    print(f"Text: '{tts_text}'")
    print(f"Reference Speaker WAV: {speaker_audio_file}")
    print(f"Output Path: {output_wav_path}")
    print(f"Settings: Temp={temperature}, LenPenalty={length_penalty}, RepPenalty={repetition_penalty}, TopK={top_k}, TopP={top_p}, Split={sentence_split}")

    if XTTS_MODEL is None:
        print("Error: Model is not loaded. Cannot run TTS.")
        return "TTS failed: Model not loaded.", None
    if not Path(speaker_audio_file).exists():
        print(f"Error: Speaker reference audio not found at {speaker_audio_file}")
        return "TTS failed: Speaker reference audio not found.", None

    try:
        print("Getting conditioning latents...")
        # Use the 24kHz version if available, otherwise the primary one. XTTS expects 24kHz internally.
        speaker_audio_path = Path(speaker_audio_file)
        ref_24k_path = speaker_audio_path.parent / f"{speaker_audio_path.stem}_24000.wav"
        if ref_24k_path.exists():
             print(f"Using 24kHz reference: {ref_24k_path}")
             speaker_ref_to_use = str(ref_24k_path)
        else:
             print(f"Warning: 24kHz reference not found, using original: {speaker_audio_file}. Model will resample.")
             speaker_ref_to_use = str(speaker_audio_file)

        # Ensure conditioning latents are calculated correctly, matching model's expectations
        # Handle potential API changes in XTTS versions
        if hasattr(XTTS_MODEL, "get_conditioning_latents"):
            gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
                audio_path=speaker_ref_to_use,
                gpt_cond_len=getattr(XTTS_MODEL.config, 'gpt_cond_len', 30), # Provide default if missing
                max_ref_length=getattr(XTTS_MODEL.config, 'max_ref_len', 60), # Provide default
                sound_norm_refs=getattr(XTTS_MODEL.config, 'sound_norm_refs', False) # Provide default
            )
        elif hasattr(XTTS_MODEL, "extract_tts_latents"): # Check for alternative method names
             # This might require different parameters depending on the XTTS version
             # Example structure, needs verification:
             latents = XTTS_MODEL.extract_tts_latents(
                 speaker_wav=speaker_ref_to_use,
                 language=lang, # Might need language here
                 # Potentially other args like gpt_cond_len etc.
             )
             gpt_cond_latent = latents.get("gpt_cond_latents") # Adjust key based on actual return
             speaker_embedding = latents.get("speaker_embedding") # Adjust key
             if gpt_cond_latent is None or speaker_embedding is None:
                 raise RuntimeError("Failed to extract latents using 'extract_tts_latents'. Check XTTS version compatibility.")
        else:
             raise NotImplementedError("Could not find a method to get conditioning latents (get_conditioning_latents or extract_tts_latents) in the loaded XTTS model.")

        print("Conditioning latents obtained.")

        print("Running TTS inference...")
        # Ensure inference parameters match the model's expected signature
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting=sentence_split
            # Add other potential args like `speed` if supported/needed by your XTTS version
        )
        print("Inference completed.")

        print(f"Saving generated audio to {output_wav_path}...")
        # Ensure output is tensor and correct shape before saving
        if isinstance(out["wav"], (list, np.ndarray)):
            wav_tensor = torch.tensor(out["wav"]).unsqueeze(0)
        elif isinstance(out["wav"], torch.Tensor):
            # Ensure it has batch and channel dims if needed, though torchaudio usually handles [samples] or [batch, samples]
            wav_tensor = out["wav"] if out["wav"].dim() > 1 else out["wav"].unsqueeze(0)
        else:
            raise TypeError(f"Unexpected type for output waveform: {type(out['wav'])}")

        torchaudio.save(str(output_wav_path), wav_tensor.cpu(), 24000) # XTTS output is 24kHz, ensure tensor is on CPU

        print(f"--- Step 4: Example TTS Generation Completed ---")
        return "Speech generated successfully!", str(output_wav_path)

    except Exception as e:
        print(f"\n---!!! TTS inference failed! !!!---")
        traceback.print_exc()
        return f"TTS inference failed: {e}", None

# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Headless XTTS Fine-tuning and Inference Script")

    # Input/Output Arguments
    parser.add_argument("--input_audio", type=str, required=True, help="Path to the single input audio file (MP3, WAV, FLAC, etc.)")
    parser.add_argument("--output_dir_base", type=str, default="./xtts_finetuned_models", help="Base directory where the output folder for this model will be created.")
    parser.add_argument("--model_name", type=str, default=None, help="Name for the output folder and reference files (defaults to input audio filename without extension).")

    # Data Processing Arguments
    parser.add_argument("--lang", type=str, default="en", help="Language of the dataset (ISO 639-1 code, e.g., en, es, fr).", choices=["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh","hu","ko","ja"])
    parser.add_argument("--whisper_model", type=str, default="large-v3", help="Whisper model to use for transcription.", choices=["large-v3","faster-whisper-large-v3-turbo", "large-v2", "large", "medium", "small", "tiny"])

    # Training Arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training.") # Default from original script
    parser.add_argument("--grad_acumm", type=int, default=1, help="Gradient accumulation steps.") # Default from original script
    parser.add_argument("--max_audio_length", type=int, default=11, help="Maximum audio segment length in seconds for training.") # Default from original script
    parser.add_argument("--xtts_base_version", type=str, default="v2.0.2", help="Base XTTS model version to fine-tune from.", choices=["v2.0.3", "v2.0.2", "v2.0.1", "v2.0.0", "main"])
    parser.add_argument("--custom_model", type=str, default="", help="(Optional) Path or URL to a custom .pth base model file.")

    # Inference Arguments
    parser.add_argument("--example_text", type=str, default="This is an example sentence generated by the fine tuned model.", help="Text to use for generating the example output WAV.")
    parser.add_argument("--temperature", type=float, default=0.75, help="TTS inference temperature.")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="TTS inference length penalty.")
    parser.add_argument("--repetition_penalty", type=float, default=5.0, help="TTS inference repetition penalty.")
    parser.add_argument("--top_k", type=int, default=50, help="TTS inference top K.")
    parser.add_argument("--top_p", type=float, default=0.85, help="TTS inference top P.")
    parser.add_argument("--no_sentence_split", action="store_true", help="Disable sentence splitting during TTS inference.")


    args = parser.parse_args()

    # --- Setup Paths ---
    input_audio_path = Path(args.input_audio)
    if not input_audio_path.is_file():
        print(f"Error: Input audio file not found at {args.input_audio}")
        sys.exit(1)

    output_name = args.model_name if args.model_name else input_audio_path.stem
    output_dir_base = Path(args.output_dir_base)
    output_dir = output_dir_base / output_name
    output_dir_ready = output_dir / "ready"
    output_dir_dataset = output_dir / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True) # Create base model output dir

    print(f"Starting Headless XTTS Training Pipeline")
    print(f"Input Audio: {input_audio_path}")
    print(f"Output Name: {output_name}")
    print(f"Output Directory: {output_dir}")
    print(f"-------------------------------------------")

    # Create a temporary directory for intermediate audio files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # --- Step 0: Prepare Audio ---
        prepared_audio_path = prepare_audio(input_audio_path, temp_dir, max_duration_minutes=40)
        if prepared_audio_path is None:
            print("Error during audio preparation. Exiting.")
            sys.exit(1)
        # Ensure the path is absolute for consistency downstream
        prepared_audio_path = Path(prepared_audio_path).resolve()

        # --- Step 1: Data Processing ---
        status, train_csv_path, eval_csv_path = preprocess_dataset_headless(
            audio_file_path=prepared_audio_path, # Pass absolute path
            language=args.lang,
            whisper_model_name=args.whisper_model,
            dataset_out_path=str(output_dir_dataset.resolve()) # Pass absolute path
        )
        if "failed" in status.lower() or "error" in status.lower():
            print(f"Error during data processing: {status}")
            sys.exit(1)
        if not train_csv_path or not eval_csv_path:
             print("Error: Data processing did not return valid metadata paths. Exiting.")
             sys.exit(1)
        # Ensure paths are absolute
        train_csv_path = Path(train_csv_path).resolve()
        eval_csv_path = Path(eval_csv_path).resolve()


        # --- Step 2: Training ---
        status, config_path, vocab_path, unoptimized_model_path, speaker_xtts_path, original_speaker_wav = train_model_headless(
             language=args.lang,
             train_csv_path=str(train_csv_path), # Pass absolute path
             eval_csv_path=str(eval_csv_path),   # Pass absolute path
             num_epochs=args.epochs,
             batch_size=args.batch_size,
             grad_acumm=args.grad_acumm,
             output_path_base=str(output_dir.resolve()), # Pass absolute path
             max_audio_length_sec=args.max_audio_length,
             version=args.xtts_base_version,
             custom_model=args.custom_model
        )
        if "failed" in status.lower() or "error" in status.lower():
             print(f"Error during training: {status}")
             sys.exit(1)
        if not config_path or not vocab_path or not unoptimized_model_path or not original_speaker_wav:
             print("Error: Training step did not return all expected file paths. Exiting.")
             sys.exit(1)


        # --- Step 2.1: Create Reference WAVs ---
        # The original_speaker_wav is often a chunk from the dataset, copy/rename/resample it.
        # Ensure original_speaker_wav path is absolute before passing
        success = create_reference_wavs(
             Path(original_speaker_wav).resolve(), # Ensure path is absolute
             output_dir_ready.resolve(), # Ensure path is absolute
             output_name # Basename for the ref files (e.g., Alban.wav)
        )
        if not success:
             print("Warning: Failed to create all reference WAVs.")
             # Decide if this is critical - for now, we continue
        final_reference_wav = output_dir_ready / f"{output_name}.wav" # Path to the main reference

        # --- Step 2.5: Optimization ---
        status, optimized_model_path = optimize_model_headless(str(output_dir.resolve())) # Pass absolute path
        if "failed" in status.lower() or "error" in status.lower():
             print(f"Error during optimization: {status}")
             sys.exit(1) # Optimization is usually desired
        if not optimized_model_path:
             print("Error: Optimization step did not return the optimized model path. Exiting.")
             sys.exit(1)


        # --- Step 3: Load Optimized Model ---
        # Paths returned from training/optimization should now be correct relative to ready_dir
        # Let's resolve them to be sure they are absolute when passed to load_model
        final_config_path = Path(config_path).resolve()
        final_vocab_path = Path(vocab_path).resolve()
        final_speaker_xtts_path = Path(speaker_xtts_path).resolve() if speaker_xtts_path else None
        optimized_model_path = Path(optimized_model_path).resolve()

        status = load_model_headless(
            xtts_checkpoint=str(optimized_model_path),
            xtts_config=str(final_config_path),
            xtts_vocab=str(final_vocab_path),
            xtts_speaker=str(final_speaker_xtts_path) if final_speaker_xtts_path else None
        )
        if "failed" in status.lower() or "error" in status.lower():
            print(f"Error loading trained model: {status}")
            sys.exit(1)

        # --- Step 4: Generate Example TTS ---
        example_output_wav_path = output_dir / f"{output_name}_generated_example.wav"
        status, generated_wav = run_tts_headless(
            lang=args.lang,
            tts_text=args.example_text,
            speaker_audio_file=str(final_reference_wav.resolve()), # Pass absolute path to reference
            output_wav_path=str(example_output_wav_path.resolve()), # Pass absolute path for output
            temperature=args.temperature,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k,
            top_p=args.top_p,
            sentence_split=not args.no_sentence_split
        )
        if "failed" in status.lower() or "error" in status.lower():
            print(f"Error generating example TTS: {status}")
            # Don't necessarily exit, the model might still be fine

    # --- End ---
    print(f"\n-------------------------------------------")
    print(f"Headless XTTS pipeline finished!")
    final_output_dir = output_dir.resolve()
    print(f"All output files are located in: {final_output_dir}")
    print(f"  - Ready model files: {final_output_dir / 'ready'}")
    print(f"  - Dataset files: {final_output_dir / 'dataset'}")
    example_wav_final_path = final_output_dir / f"{output_name}_generated_example.wav"
    print(f"  - Example generation: {example_wav_final_path if 'generated_wav' in locals() and generated_wav else 'Failed or Not Generated'}")
    print(f"-------------------------------------------")

if __name__ == "__main__":
    main()