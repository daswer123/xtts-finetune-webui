# xtts-finetune-webui

This webui is a slightly modified copy of the [official webui](https://github.com/coqui-ai/TTS/pull/3296) for finetune xtts.

If you are looking for an option for normal XTTS use look here [https://github.com/daswer123/xtts-webui](https://github.com/daswer123/xtts-webui)

## TODO
- [ ] Add the ability to use via console 

## Key features:

### Data processing

1. Updated faster-whisper to 0.10.0 with the ability to select a larger-v3 model.
2. Changed output folder to output folder inside the main folder.
3. If there is already a dataset in the output folder and you want to add new data, you can do so by simply adding new audio, what was there will not be processed again and the new data will be automatically added
4. Turn on VAD filter
5. After the dataset is created, a file is created that specifies the language of the dataset. This file is read before training so that the language always matches. It is convenient when you restart the interface

### Fine-tuning XTTS Encoder

1. Added the ability to select the base model for XTTS, as well as when you re-training does not need to download the model again.
2. Added ability to select custom model as base model during training, which will allow finetune already finetune model.
3. Added possibility to get optimized version of the model for 1 click ( step 2.5, put optimized version in output folder).
4. You can choose whether to delete training folders after you have optimized the model
5. When you optimize the model, the example reference audio is moved to the output folder
6. Checking for correctness of the specified language and dataset language

### Inference

1. Added possibility to customize infer settings during model checking.

### Other

1. If you accidentally restart the interface during one of the steps, you can load data to additional buttons
2. Removed the display of logs as it was causing problems when restarted
3. The finished result is copied to the ready folder, these are fully finished files, you can move them anywhere and use them as a standard model
4. Added support for finetune Japanese 

## Changes in webui

### 1 - Data processing

![image](https://github.com/daswer123/xtts-finetune-webui/assets/22278673/8f09b829-098b-48f5-9668-832e7319403b)

### 2 - Fine-tuning XTTS Encoder

![image](https://github.com/daswer123/xtts-finetune-webui/assets/22278673/897540d9-3a6b-463c-abb8-261c289cc929)

### 3 - Inference

![image](https://github.com/daswer123/xtts-finetune-webui/assets/22278673/aa05bcd4-8642-4de4-8f2f-bc0f5571af63)

## Install

1. Make sure you have `Cuda` installed
2. `git clone https://github.com/daswer123/xtts-finetune-webui`
3. `cd xtts-finetune-webui`
4. `pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118`
5. `pip install -r requirements.txt`

### If you're using Windows

1. First start `install.bat`
2. To start the server start `start.bat`
3. Go to the local address `127.0.0.1:5003`

### On Linux

1. Run `bash install.sh`
2. To start the server start `start.sh`
3. Go to the local address `127.0.0.1:5003`
~                                            
