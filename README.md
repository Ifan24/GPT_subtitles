# Dual Subtitles for Video

下载 YouTube 视频（或提供您自己的视频）并使用 Whisper 生成双语字幕

This project is a Python script that downloads a YouTube video (or uses a local video file), transcribes it, translates the transcript into a target language, and generates a video with dual subtitles (original and translated). The transcription and translation are powered by the Whisper model and the M2M100 model, respectively.

Note: Embedding the subtitle into the video is not working yet, due to some bugs causing the font in non-English languages to not be found. For now, it will only generate a dual-language SRT file.


# Requirements

- Python 3.9 or later
- GPU (recommended for better performance)

Additionally, when running the script for first time, it will download the following pre-trained models:

- [Whisper Model](https://github.com/openai/whisper) (small): ~461 MB
- Facebook M2M100 Model: ~2 GB (Optional, you can also use Google Translate API with googletrans, or Whisper's transcribe)

# Installation
1. Clone this repository.
2. Install the required dependencies using ``` pip install -r requirements.txt ```

# Usage
You can provide either a YouTube URL or a local video file for processing. The script will transcribe the video, translate the transcript, and generate dual subtitles in the form of an SRT file.

```
python main.py --youtube_url [YOUTUBE_URL] --target_language [TARGET_LANGUAGE] --model [WHISPER_MODEL] --translation_method [TRANSLATION_METHOD]

```

You can also use GPT-3.5 to translate the transcript. You will need to provide your own API key in .env file.

[showcase of GPT-3.5 translation](https://www.bilibili.com/video/BV1Qc411n7pE/)

# Arguments

---youtube_url: The URL of the YouTube video.

--local_video: The path to the local video file.

--target_language: The target language for translation (default: 'zh').

--model: Choose one of the Whisper models (default: 'small', choices: ['tiny', 'base', 'small', 'medium', 'large']).

--translation_method: The method to use for translation. Options: "m2m100" or "google" or "whisper" (default: 'm2m100', choices: ['m2m100', 'google', 'whisper']).


Note: You must provide either --youtube_url or --local_video, but not both.

# Example

To download a YouTube video, transcribe it, and generate dual subtitles using the M2M100 translation method:

```
python main.py --youtube_url [YOUTUBE_URL] --target_language 'zh' --model 'small' --translation_method 'm2m100'
```

To process a local video file, transcribe it, and generate dual subtitles using Whisper's transcribe method (it will download the large Whisper model if it is not already downloaded):

```
python main.py --local_video [VIDEO_FILE_PATH] --target_language 'zh' --model 'large' --translation_method 'whisper'
```


The script will generate the following output files in the same directory as the input video:

- An SRT file containing the original transcribed subtitles.
- An SRT file containing the translated subtitles.
- An SRT file containing the combined dual subtitles. (having some issues showing the dual subtitles as two lines in the video)
- A video file with embedded dual subtitles (not yet work).

<!-- This will download the specified YouTube video, transcribe it, translate the transcript into Chinese, and generate a video with dual subtitles (English and Chinese). The output video will be saved in the same directory as the original video with the postfix _dual_sub. -->


# Google Colab Example
You can also try out this script using a Google Colab notebook. Click the link below to access the example:

[Colab](https://colab.research.google.com/drive/1XDLFlgew9BzUqNpTv_kq0HNocTNOSekP?usp=sharing)

Follow the instructions in the notebook to download the necessary packages and models, and to run the script on your desired YouTube video or local video file.