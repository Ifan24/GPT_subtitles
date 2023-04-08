# Dual Subtitles for Video

下载 YouTube 视频（或提供您自己的视频）并使用 Whisper 在视频中嵌入双字幕

This project is a Python script that downloads a YouTube video (or uses a local video file), transcribes it, translates the transcript into a target language, and generates a video with dual subtitles (original and translated). The transcription and translation are powered by the Whisper model and the M2M100 model, respectively.

note: Embedding the subtitle into the video is not working yet, because some bugs are causing the font in non-English to be not found, for now, it would only generate a dual-language srt file


# Requirements

- Python 3.9 or later
- GPU (recommended for better performance)

Additionally, when running the script for first time, it will download the following pre-trained models:

- [Whisper Model](https://github.com/openai/whisper) (small): ~461 MB
- Facebook M2M100 Model: ~2 GB

# Installation
1. Clone this repository.
2. Install the required dependencies using ``` pip install -r requirements.txt ```

# Usage
```
python main.py [--youtube_url YOUTUBE_URL] [--local_video LOCAL_VIDEO] [--target_language TARGET_LANGUAGE] [--model MODEL]
```

# Arguments
--youtube_url: The URL of the YouTube video to download and process.

--local_video: The path to the local video file to process.

--target_language: The target language for translation (default: 'zh').

--model: The Whisper model to use for transcription (default: 'small').

Note: You must provide either --youtube_url or --local_video, but not both.

# Example
```
python main.py --youtube_url https://www.youtube.com/watch?v=EXAMPLE
```

The script will generate the following output files in the same directory as the input video:

- An SRT file containing the original transcribed subtitles.
- An SRT file containing the translated subtitles.
- An SRT file containing the combined dual subtitles.
- A video file with embedded dual subtitles (not yet work).

<!-- This will download the specified YouTube video, transcribe it, translate the transcript into Chinese, and generate a video with dual subtitles (English and Chinese). The output video will be saved in the same directory as the original video with the postfix _dual_sub. -->


# Google Colab Example
You can also try out this script using a Google Colab notebook. Click the link below to access the example:

[Colab](https://colab.research.google.com/drive/1XDLFlgew9BzUqNpTv_kq0HNocTNOSekP?usp=sharing)

Follow the instructions in the notebook to download the necessary packages and models, and to run the script on your desired YouTube video or local video file.