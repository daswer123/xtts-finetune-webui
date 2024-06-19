import os
import gc
import torchaudio
import pandas as pd
from faster_whisper import WhisperModel
from glob import glob

from tqdm import tqdm

from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners
# Add support for JA train
# from utils.tokenizer import multilingual_cleaners

import torch
import torchaudio
import json
# torch.set_num_threads(1)

from audio_separator.separator import Separator

import shutil


torch.set_num_threads(16)
import os

audio_types = (".wav", ".mp3", ".flac")

def find_latest_best_model(folder_path):
        search_path = os.path.join(folder_path, '**', 'best_model.pth')
        files = glob(search_path, recursive=True)
        latest_file = max(files, key=os.path.getctime, default=None)
        return latest_file


def list_audios(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=audio_types, contains=contains)

def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an audio and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the audio and yield it
                audioPath = os.path.join(rootDir, filename)
                yield audioPath


def save_dataset_info(out_path, audio_files, num_segments, total_duration, target_language):
    info_file_path = os.path.join(out_path, "info.json")

    if os.path.exists(info_file_path):
        with open(info_file_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    dataset_info['audio_files'] = dataset_info.get('audio_files', []) + audio_files
    dataset_info['num_segments'] = dataset_info.get('num_segments', 0) + num_segments
    dataset_info['total_duration'] = dataset_info.get('total_duration', 0) + total_duration
    dataset_info['language'] = target_language

    with open(info_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4)


def save_transcriptions(out_path, audio_file_name, sentence):
    txt_dir = os.path.join(out_path, "txt")
    os.makedirs(txt_dir, exist_ok=True)

    txt_file_path = os.path.join(txt_dir, f"{audio_file_name}")
    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write(sentence)


def create_manifest(out_path, audio_file, txt_file, duration):
    manifest_file_path = os.path.join(out_path, "manifest.csv")

    with open(manifest_file_path, 'a', encoding='utf-8') as f:
        f.write(f"{audio_file}|{txt_file}|{duration}\n")


def format_audio_list(audio_files, asr_model, target_language="en", out_path=None, buffer=0.2, eval_percentage=0.15, speaker_name="coqui", use_separate_audio=True, gradio_progress=None):
    audio_total_size = 0
    num_segments = 0
    os.makedirs(out_path, exist_ok=True)

    lang_file_path = os.path.join(out_path, "lang.txt")
    current_language = None

    if current_language != target_language:
        with open(lang_file_path, 'w', encoding='utf-8') as lang_file:
            lang_file.write(target_language + '\n')
        print("Warning, existing language does not match target language. Updated lang.txt with target language.")
    else:
        print("Existing language matches target language")

    metadata = {"audio_file": [], "text": [], "speaker_name": []}
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")

    existing_metadata = {'train': None, 'eval': None}
    if os.path.exists(train_metadata_path):
        existing_metadata['train'] = pd.read_csv(train_metadata_path, sep="|")
        print("Existing training metadata found and loaded.")

    if os.path.exists(eval_metadata_path):
        existing_metadata['eval'] = pd.read_csv(eval_metadata_path, sep="|")
        print("Existing evaluation metadata found and loaded.")

    if gradio_progress is not None:
        tqdm_object = gradio_progress.tqdm(audio_files, desc="Formatting...")
    else:
        tqdm_object = tqdm(audio_files)

    for audio_path in tqdm_object:
        
        # original_audio_path = audio_path
        separate_audio_path = audio_path
        
        if use_separate_audio:
            # Use separator
            # Load a model
            separator = Separator(output_dir="tmp")
            separator.load_model(model_filename='Kim_Vocal_2.onnx')

            # Separate multiple audio files without reloading the model
            separate_audio_path = separator.separate(audio_path)
            
            # Remove instrumetnal version
            instrumental_path = os.path.join("tmp", separate_audio_path[0])
            os.remove(instrumental_path)
            
            # Use vocal part as main audio
            separate_audio_path = separate_audio_path[1]
            separate_audio_path = os.path.join("tmp", separate_audio_path)
            print(f"Separated {audio_path}")
        
        audio_file_name_without_ext,_ = os.path.splitext(os.path.basename(audio_path))
        prefix_check = f"wavs/{audio_file_name_without_ext}_"

        skip_processing = False
        for key in ['train', 'eval']:
            if existing_metadata[key] is not None:
                mask = existing_metadata[key]['audio_file'].str.startswith(prefix_check)
                if mask.any():
                    print(f"Segments from {audio_file_name_without_ext} have been previously processed; skipping...")
                    skip_processing = True
                    break

        if skip_processing:
            continue

        wav, sr = torchaudio.load(audio_path)
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        segments,_ = asr_model.transcribe(separate_audio_path, vad_filter=True, word_timestamps=True, language=target_language)
        segments = list(segments)
        i = 0
        sentence = ""
        sentence_start = None
        first_word = True
        words_list = []
        for _, segment in enumerate(segments):
            words = list(segment.words)
            words_list.extend(words)

        for word_idx, word in enumerate(words_list):
            if first_word:
                sentence_start = word.start
                if word_idx == 0:
                    sentence_start = max(sentence_start - buffer, 0)
                else:
                    previous_word_end = words_list[word_idx - 1].end
                    sentence_start = max(sentence_start - buffer, (previous_word_end + sentence_start) / 2)

                sentence = word.word
                first_word = False
            else:
                sentence += word.word

            if word.word[-1] in ["!", "。", ".", "?"]:
                sentence = sentence[1:]
                sentence = multilingual_cleaners(sentence, target_language)
                audio_file_name,_ = os.path.splitext(os.path.basename(audio_path))
                audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"

                if word_idx + 1 < len(words_list):
                    next_word_start = words_list[word_idx + 1].start
                else:
                    next_word_start = (wav.shape[0] - 1) / sr

                word_end = min((word.end + next_word_start) / 2, word.end + buffer)

                absolute_path = os.path.join(out_path, audio_file)
                os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
                i += 1
                first_word = True

                audio = wav[int(sr*sentence_start):int(sr*word_end)].unsqueeze(0)
                if audio.size(-1) >= sr / 3:
                    torchaudio.save(absolute_path, audio, sr)
                    
                    txt_filename = f"{audio_file_name}_{str(i).zfill(8)}.txt"
                    
                    # Save transcription
                    save_transcriptions(out_path, txt_filename, sentence)

                    # Update manifest
                    txt_file = f"txt/{txt_filename}"
                    duration = audio.size(-1) / sr
                    create_manifest(out_path, audio_file, txt_file, duration)
                    
                    num_segments += 1
                else:
                    continue

                metadata["audio_file"].append(audio_file)
                metadata["text"].append(sentence)
                metadata["speaker_name"].append(speaker_name)

                df = pd.DataFrame(metadata)

                mode = 'w' if not os.path.exists(train_metadata_path) else 'a'
                header = not os.path.exists(train_metadata_path)
                df.to_csv(train_metadata_path, sep="|", index=False, mode=mode, header=header)

                mode = 'w' if not os.path.exists(eval_metadata_path) else 'a'
                header = not os.path.exists(eval_metadata_path)
                df.to_csv(eval_metadata_path, sep="|", index=False, mode=mode, header=header)

                metadata = {"audio_file": [], "text": [], "speaker_name": []}
                
        # Delete separated audio files
        if use_separate_audio:
            os.remove(separate_audio_path)
                

    if os.path.exists(train_metadata_path) and os.path.exists(eval_metadata_path):
        existing_train_df = existing_metadata['train']
        existing_eval_df = existing_metadata['eval']
    else:
        existing_train_df = pd.DataFrame(columns=["audio_file", "text", "speaker_name"])
        existing_eval_df = pd.DataFrame(columns=["audio_file", "text", "speaker_name"])

    new_data_df = pd.read_csv(train_metadata_path, sep="|")

    combined_train_df = pd.concat([existing_train_df, new_data_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    combined_eval_df = pd.concat([existing_eval_df, new_data_df], ignore_index=True).drop_duplicates().reset_index(drop=True)

    combined_train_df_shuffled = combined_train_df.sample(frac=1)
    num_val_samples = int(len(combined_train_df_shuffled) * eval_percentage)

    final_eval_set = combined_train_df_shuffled[:num_val_samples]
    final_training_set = combined_train_df_shuffled[num_val_samples:]

    final_training_set.sort_values('audio_file').to_csv(train_metadata_path, sep='|', index=False)
    final_eval_set.sort_values('audio_file').to_csv(eval_metadata_path, sep='|', index=False)

    # Save dataset info
    save_dataset_info(out_path, audio_files, num_segments, audio_total_size, target_language)

    return train_metadata_path, eval_metadata_path, audio_total_size



def merge_datasets(dataset1_path, dataset2_path, merged_dataset_path):
    # Создаем директорию для нового датасета
    os.makedirs(merged_dataset_path, exist_ok=True)

    # Копируем все файлы из первого датасета в новый датасет
    for root, dirs, files in os.walk(dataset1_path):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(merged_dataset_path, os.path.relpath(src_path, dataset1_path))
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)

    # Копируем все файлы из второго датасета в новый датасет, пропуская дубликаты
    for root, dirs, files in os.walk(dataset2_path):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(merged_dataset_path, os.path.relpath(src_path, dataset2_path))
            if not os.path.exists(dst_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy(src_path, dst_path)

    # Объединяем файлы metadata_train.csv и metadata_eval.csv, удаляя дубликаты
    for metadata_file in ["metadata_train.csv", "metadata_eval.csv"]:
        metadata1_path = os.path.join(dataset1_path, metadata_file)
        metadata2_path = os.path.join(dataset2_path, metadata_file)
        merged_metadata_path = os.path.join(merged_dataset_path, metadata_file)

        if os.path.exists(metadata1_path) and os.path.exists(metadata2_path):
            metadata1 = pd.read_csv(metadata1_path, sep="|")
            metadata2 = pd.read_csv(metadata2_path, sep="|")
            merged_metadata = pd.concat([metadata1, metadata2]).drop_duplicates(subset="audio_file").reset_index(drop=True)
            merged_metadata.to_csv(merged_metadata_path, sep="|", index=False)
        elif os.path.exists(metadata1_path):
            shutil.copy(metadata1_path, merged_metadata_path)
        elif os.path.exists(metadata2_path):
            shutil.copy(metadata2_path, merged_metadata_path)

    # Объединяем файлы info.json, удаляя дубликаты
    info1_path = os.path.join(dataset1_path, "info.json")
    info2_path = os.path.join(dataset2_path, "info.json")
    merged_info_path = os.path.join(merged_dataset_path, "info.json")

    if os.path.exists(info1_path) and os.path.exists(info2_path):
        with open(info1_path, "r") as f1, open(info2_path, "r") as f2, open(merged_info_path, "w") as f_merged:
            info1 = json.load(f1)
            info2 = json.load(f2)
            merged_info = {
                "audio_files": list(set(info1["audio_files"] + info2["audio_files"])),
                "num_segments": info1["num_segments"] + info2["num_segments"],
                "total_duration": info1["total_duration"] + info2["total_duration"],
                "language": info1["language"]
            }
            json.dump(merged_info, f_merged, indent=4)
    elif os.path.exists(info1_path):
        shutil.copy(info1_path, merged_info_path)
    elif os.path.exists(info2_path):
        shutil.copy(info2_path, merged_info_path)
