import os
import json
import shutil
import pandas as pd
from datasets import Dataset, Audio, Value, Features, DatasetDict

def create_readme_from_info(info_file_path, readme_file_path):
    if os.path.exists(info_file_path):
        with open(info_file_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)

        readme_content = "# Dataset Information\n\n"
        readme_content += "## General Info\n"
        readme_content += f"-**Language**: {info_data.get('language', 'N/A')}\n"
        readme_content += f"-**Number of Segments**: {info_data.get('num_segments', 'N/A')}\n"
        readme_content += f"-**Total Duration**: {info_data.get('total_duration', 'N/A')} seconds\n\n"

        with open(readme_file_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

def create_and_upload_dataset(upload_address, dataset_path):
    # Загрузка manifest.csv
    
    # dataset_path_folder = os.path.dirname(dataset_path)
    # dataset_not_abs_path = os.path.join("datasets", dataset_path_folder)
    
    def read_text_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    manifest_df = pd.read_csv(os.path.join(dataset_path, "manifest.csv"), sep="|", header=None, names=["audio_file", "text_file", "duration"])
    
    manifest_df['audio_file'] = manifest_df['audio_file'].apply(lambda x: os.path.join(dataset_path,x))
    manifest_df['text_file'] = manifest_df['text_file'].apply(lambda x: os.path.join(dataset_path,x))
    # manifest_df['text_file'] = manifest_df['text_file'].apply(read_text_file)

    # Разделение данных на train и eval
    train_size = int(0.8 * len(manifest_df))
    train_manifest_df = manifest_df[:train_size].copy()
    eval_manifest_df = manifest_df[train_size:].copy()

    # Переименование столбцов 'audio_file' и 'text_file'
    train_manifest_df = train_manifest_df.rename(columns={'audio_file': 'audio', 'text_file': 'text'})
    eval_manifest_df = eval_manifest_df.rename(columns={'audio_file': 'audio', 'text_file': 'text'})

    # Создание train и eval datasets
    train_dataset = Dataset.from_pandas(train_manifest_df, features=Features({
        "audio": Audio(),
        "text": Value("string"),
        "duration": Value("float32"),
    }))

    eval_dataset = Dataset.from_pandas(eval_manifest_df, features=Features({
        "audio": Audio(),
        "text": Value("string"),
        "duration": Value("float32"),
    }))

    # Обновление путей к аудиофайлам и текстовым файлам
    def update_paths(example):
        # example['audio'] = {'path': os.path.join(dataset_path, example['audio'])}
        with open(os.path.abspath(example['text']), 'r', encoding='utf-8') as f:
            example['text'] = f.read().strip()
        return example

    train_dataset = train_dataset.map(update_paths)
    eval_dataset = eval_dataset.map(update_paths)

    # Создание DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "eval": eval_dataset,
    })

    # Создание README.md из info.json
    info_json_path = os.path.join(dataset_path, "info.json")
    readme_md_path = os.path.join(dataset_path, "README.md")
    create_readme_from_info(info_json_path, readme_md_path)

    # Загрузка датасета на Hugging Face Hub
    dataset_dict.push_to_hub(upload_address)

