# Dual Subtitles for Video

下载 YouTube 视频（或提供您自己的视频）并使用 Whisper 和翻译API生成双语字幕

This project is a Python script that downloads a YouTube video (or uses a local video file), transcribes it, translates the transcript into a target language, and generates a video with dual subtitles (original and translated). The transcription and translation are powered by the Whisper model and the translation API (M2M100, google, GPT3.5), respectively.

<!-- Note: Embedding the subtitles into the video is not working yet, due to some bugs causing the font in non-English languages to not be found. For now, it will only generate a dual-language SRT file. -->


# Requirements

- Python 3.9 or later
- GPU (recommended for better performance)

Additionally, when running the script for first time, it will download the following pre-trained models:

- [Whisper Model](https://github.com/openai/whisper) (small): ~461 MB
- Facebook M2M100 Model: ~2 GB (Optional, you can also use Google Translate API with googletrans, or Whisper's transcribe)
- OpenAI API key (Optional, if you want to use GPT-3.5 for subtitle translation)

# Installation
1. Clone this repository.
2. Install the required dependencies using ``` pip install -r requirements.txt ```

# Usage
You can provide either a YouTube URL or a local video file for processing. The script will transcribe the video, translate the transcript, and generate dual subtitles in the form of an SRT file.

```
python main.py --youtube_url [YOUTUBE_URL] --target_language [TARGET_LANGUAGE] --model [WHISPER_MODEL] --translation_method [TRANSLATION_METHOD]

```
# Arguments

---youtube_url: The URL of the YouTube video.

--local_video: The path to the local video file.

--target_language: The target language for translation (default: 'zh').

--model: Choose one of the Whisper models (default: 'small', choices: ['tiny', 'base', 'small', 'medium', 'large']).

--translation_method: The method to use for translation. (default: 'm2m100', choices: ['m2m100', 'google', 'whisper', 'gpt']).


Note: You must provide either --youtube_url or --local_video, but not both.

# Example

To download a YouTube video, transcribe it, and generate dual subtitles using the M2M100 translation method:

```
python main.py --youtube_url [YOUTUBE_URL] --target_language 'zh' --model 'small' --translation_method 'google'
```

To process a local video file, transcribe it, and generate dual subtitles using Whisper's transcribe method (you will need to download the large Whisper model if it is not already downloaded):

```
python main.py --local_video [VIDEO_FILE_PATH] --target_language 'zh' --model 'large' --translation_method 'whisper'
```


The script will generate the following output files in the same directory as the input video:

- An SRT file containing the original transcribed subtitles.
- An SRT file containing the translated subtitles.
- An SRT file containing the combined dual subtitles.
- A video file with embedded dual subtitles (not yet work).


# Subtitle Translation using GPT-3.5 （translate_gpt.py）

This script translates subtitles using OpenAI's GPT-3.5 language model. It requires an OpenAI API key to function. In some cases, GPT-based translation might produce better results compared to Google Translate, especially when dealing with context-specific translations or idiomatic expressions. This script aims to provide an alternative method for translating subtitles when traditional translation services like Google Translate do not produce satisfactory results.
### Setup
1. Sign up for an API key from OpenAI at https://platform.openai.com/account/api-keys
2. Once you have the API key, create a file named .env in the same directory as the script.
3. Add the following line to the .env file:
```
OPENAI_API_KEY=your_api_key_here
```
Replace your_api_key_here with the API key you obtained from OpenAI.

### Usage

```
python translate_gpt.py --input_file INPUT_FILE_PATH [--batch_size BATCH_SIZE] [--target_language TARGET_LANGUAGE]

```

### Arguments

- --input_file: The path to the input subtitle file. (Required)
- --batch_size: The number of subtitles to process in a batch. (Optional, default: 3)
- --target_language: The target language for translation. (Optional, default: 'zh' for Simplified Chinese)


[showcase of GPT-3.5 translation](https://www.bilibili.com/video/BV1xv4y1E7ZD/)


# Contributing
Contributions are more than welcome!


# TODO
- [ ] Fix the bug that prevents embedding dual subtitles into the video.
- [ ] Implement a GUI to make the tool more user-friendly.
- [ ] Let GPT summarize a list of word-to-word translation in its response and use it to improve translation consistency, and let human post-process the transcript by modify the dictionary.
- [ ] Use GPT4 to generate Few-shot examples for other language pairs, and make a JSON file for the examples.
- [x] Explore the possibility of using offline small GPT models. (I tried some models on [webGUI](https://github.com/oobabooga/text-generation-webui) with the prompt, but they mostly output random text or translation that are worst than google translate)
- [ ] [Fine-tune Whisper](https://github.com/jumon/whisper-finetuning) on (English audio, Chinese subtitle) dataset to improve x-to-Chinese translation accuracy

# Google Colab Example
You can also try out this script using a Google Colab notebook. Click the link below to access the example:

[Colab](https://colab.research.google.com/drive/1XDLFlgew9BzUqNpTv_kq0HNocTNOSekP?usp=sharing)

Follow the instructions in the notebook to download the necessary packages and models, and to run the script on your desired YouTube video or local video file.