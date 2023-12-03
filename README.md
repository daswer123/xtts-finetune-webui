# xtts-finetune-webui
This webui is a slightly modified copy of the [official webui](https://github.com/coqui-ai/TTS/pull/3296) for finetune xtts.

## Key features:
1) Updated faster-whisper to 0.10.0 with the ability to select a larger-v3 model.
2) Added the ability to select the base model for XTTS, as well as when you re-training does not need to download the model again.
3) Added ability to select custom model as base model during training, which will allow finetune already finetune model.
4) Added possibility to get optimized version of the model for 1 click ( step 2.5, put optimized version in output folder).
5) Changed output folder to output folder inside the main folder.
6) Added possibility to customize infer settings during model checking.

## Changes in webui
### 1 - Data processing

![image](https://github.com/daswer123/xtts-finetune-webui/assets/22278673/8f09b829-098b-48f5-9668-832e7319403b)

### 2 - Fine-tuning XTTS Encoder

![image](https://github.com/daswer123/xtts-finetune-webui/assets/22278673/4a07a65c-f807-42b1-8514-6bff73086e31)

### 3 - Inference

![image](https://github.com/daswer123/xtts-finetune-webui/assets/22278673/dade10d3-82f3-4458-a647-c82081ad1ff8)


## Install
1) Make sure you have `Cuda` installed
2) `git clone https://github.com/daswer123/xtts-finetune-webui`
3) `cd xtts-finetune-webui`
4) `pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118`
5) `pip install -r requirements.txt`

### If you're using Windows
1) First start `install.bat`
2) To start the server start `start.bat`
3) Go to the local address `127.0.0.1:5003`
