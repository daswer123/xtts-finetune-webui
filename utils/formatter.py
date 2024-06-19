import os
import gc
import torchaudio
import pandas
from faster_whisper import WhisperModel
from glob import glob

from tqdm import tqdm

from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners
# Add support for JA train
# from utils.tokenizer import multilingual_cleaners

import torch
import torchaudio
# torch.set_num_threads(1)


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

def format_audio_list(audio_files, asr_model, target_language="en", out_path=None, buffer=0.2, eval_percentage=0.15, speaker_name="coqui", gradio_progress=None):
    audio_total_size = 0
    os.makedirs(out_path, exist_ok=True)

    lang_file_path = os.path.join(out_path, "lang.txt")
    current_language = None
    if os.path.exists(lang_file_path):
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()

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
        existing_metadata['train'] = pandas.read_csv(train_metadata_path, sep="|")
        print("Existing training metadata found and loaded.")

    if os.path.exists(eval_metadata_path):
        existing_metadata['eval'] = pandas.read_csv(eval_metadata_path, sep="|")
        print("Existing evaluation metadata found and loaded.")

    if gradio_progress is not None:
        tqdm_object = gradio_progress.tqdm(audio_files, desc="Formatting...")
    else:
        tqdm_object = tqdm(audio_files)

    for audio_path in tqdm_object:
        audio_file_name_without_ext, _= os.path.splitext(os.path.basename(audio_path))
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

        segments, _= asr_model.transcribe(audio_path, vad_filter=True, word_timestamps=True, language=target_language)
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
                audio_file_name, _= os.path.splitext(os.path.basename(audio_path))
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

                audio = wav[int(sr*sentence_start):int(sr *word_end)].unsqueeze(0)
                if audio.size(-1) >= sr / 3:
                    torchaudio.save(absolute_path, audio, sr)
                else:
                    continue

                metadata["audio_file"].append(audio_file)
                metadata["text"].append(sentence)
                metadata["speaker_name"].append(speaker_name)

                df = pandas.DataFrame(metadata)

                mode = 'w' if not os.path.exists(train_metadata_path) else 'a'
                header = not os.path.exists(train_metadata_path)
                df.to_csv(train_metadata_path, sep="|", index=False, mode=mode, header=header)

                mode = 'w' if not os.path.exists(eval_metadata_path) else 'a'
                header = not os.path.exists(eval_metadata_path)
                df.to_csv(eval_metadata_path, sep="|", index=False, mode=mode, header=header)

                metadata = {"audio_file": [], "text": [], "speaker_name": []}

    if os.path.exists(train_metadata_path) and os.path.exists(eval_metadata_path):
        existing_train_df = existing_metadata['train']
        existing_eval_df = existing_metadata['eval']
    else:
        existing_train_df = pandas.DataFrame(columns=["audio_file", "text", "speaker_name"])
        existing_eval_df = pandas.DataFrame(columns=["audio_file", "text", "speaker_name"])

    new_data_df = pandas.read_csv(train_metadata_path, sep="|")

    combined_train_df = pandas.concat([existing_train_df, new_data_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    combined_eval_df = pandas.concat([existing_eval_df, new_data_df], ignore_index=True).drop_duplicates().reset_index(drop=True)

    combined_train_df_shuffled = combined_train_df.sample(frac=1)
    num_val_samples = int(len(combined_train_df_shuffled)* eval_percentage)

    final_eval_set = combined_train_df_shuffled[:num_val_samples]
    final_training_set = combined_train_df_shuffled[num_val_samples:]

    final_training_set.sort_values('audio_file').to_csv(train_metadata_path, sep='|', index=False)
    final_eval_set.sort_values('audio_file').to_csv(eval_metadata_path, sep='|', index=False)

    return train_metadata_path, eval_metadata_path, audio_total_size
