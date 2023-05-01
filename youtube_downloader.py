import os
import time
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter
from pytube import YouTube
from pythumb import Thumbnail
import subprocess
from urllib.parse import parse_qs, urlparse

class TranscriptFetcher:
    def __init__(self, video_id):
        self.video_id = video_id
        self.transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
    def fetch_transcript(self, target_language):
        try:
            transcript = self.transcript_list.find_manually_created_transcript([target_language])
        except:
            try:
                transcript = self.transcript_list.find_manually_created_transcript(
                    self.transcript_list._manually_created_transcripts.keys())
            except:
                try:
                    transcript = self.transcript_list.find_generated_transcript([target_language])
                except:
                    transcript = self.transcript_list.find_generated_transcript(
                        self.transcript_list._generated_transcripts.keys())
        
        if target_language != 'en':
            transcript = transcript.translate(target_language)
        transcript_data = transcript.fetch()
        return transcript_data, transcript.language

class SRTDownloader:
    def __init__(self, url, title, output_path):
        self.url = url
        self.title = title
        self.output_path = output_path

    def get_youtube_id(self):
        parsed_url = urlparse(self.url)

        if parsed_url.netloc == "www.youtube.com":
            return parse_qs(parsed_url.query).get('v', [None])[0]
        raise ValueError(f"Invalid YouTube URL: {self.url}")

    def download(self, target_language):
        try:
            video_id = self.get_youtube_id()
            transcript_data, language = TranscriptFetcher(video_id).fetch_transcript(target_language)
            srt_formatted = SRTFormatter().format_transcript(transcript_data)

            filename = f"[{language}] {self.title}.srt"
            srt_file_path = os.path.join(self.output_path, filename)

            with open(srt_file_path, 'w', encoding='utf-8') as file:
                file.write(srt_formatted)

            print(f'SRT file downloaded at: {srt_file_path}')
            return True
        except Exception as e:
            print(f"Failed to download transcript. Error: {e}")
            return False


class YouTubeDownloader:
    def __init__(self, url, target_language='zh-Hans'):
        self.url = url
        self.target_language = target_language
        if target_language == 'zh':
            self.target_language = 'zh-Hans'

    def download_video(self):
        yt = YouTube(self.url)
        while True:
            try:
                title = yt.title
                break
            except:
                print("Failed to get name. Retrying... Press Ctrl+C to exit")
                time.sleep(1)
                yt = YouTube(self.url)
                continue

        print('Downloading video: ' + title)

        # Create a folder called 'videos' if it does not exist
        if not os.path.exists('videos'):
            os.makedirs('videos')

        # Create a folder with the video title inside the "videos" folder
        video_folder = os.path.join("videos", title)
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        # Download English transcript first
        SRTDownloader(self.url, title, video_folder).download('en')
        # Then try downloading the target language transcript
        if self.target_language != 'en':
            SRTDownloader(self.url, title, video_folder).download(self.target_language)


        # Download the thumbnail using pythumb
        thumbnail = Thumbnail(self.url)
        thumbnail.fetch()
        thumbnail.save(dir=video_folder, filename='thumbnail', overwrite=True)
        print(f'Thumbnail saved at: {video_folder}')

        # Download the video using yt-dlp
        output_filename = os.path.join(video_folder, f"{title}.%(ext)s")
        youtube_dl_command = f"yt-dlp -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]' --merge-output-format mp4 -o \"{output_filename}\" {self.url}"
        subprocess.run(youtube_dl_command, shell=True, check=True)

        # Find the downloaded video file
        downloaded_video_path = None
        count = 0
        while downloaded_video_path is None:
            for file in os.listdir(video_folder):
                if file.endswith(".mp4"):
                    downloaded_video_path = os.path.join(video_folder, file)
                    break
            print(f" | Waiting for video to download... | time elapsed: {count} seconds |")
            count += 30
            time.sleep(30)

        print('Download complete: ' + downloaded_video_path)
        print(f'File size: {os.path.getsize(downloaded_video_path) / 1e6} mb')

        return downloaded_video_path
