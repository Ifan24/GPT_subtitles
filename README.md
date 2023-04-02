# Dual Subtitles for Video

下载 YouTube 视频（或提供您自己的视频）并使用 Whisper 在视频中嵌入双字幕

This project is a Python script that downloads a YouTube video (or uses a local video file), transcribes it, translates the transcript into a target language, and generates a video with dual subtitles (original and translated). The transcription and translation are powered by the Whisper model and the M2M100 model, respectively.

note: Embedding the subtitle into the video is not working yet, because some bugs causing the font in non-english is not found

# Installation
1. Clone this repository.
2. Install the required dependencies using ``` pip install -r requirements.txt ```

# Usage
```
python dual_subtitles.py [--youtube_url YOUTUBE_URL] [--local_video LOCAL_VIDEO] [--target_language TARGET_LANGUAGE] [--model MODEL] [--font_path FONT_PATH]
```

# Arguments
--youtube_url: The URL of the YouTube video to download and process.

--local_video: The path to the local video file to process.

--target_language: The target language for translation (default: 'zh').

--model: The Whisper model to use for transcription (default: 'small').

Note: You must provide either --youtube_url or --local_video, but not both.

# Example
```
python dual_subtitles.py --youtube_url https://www.youtube.com/watch?v=EXAMPLE --target_language zh --model small
```
This will download the specified YouTube video, transcribe it, translate the transcript into Chinese, and generate a video with dual subtitles (English and Chinese). The output video will be saved in the same directory as the original video with the prefix dual_sub_.