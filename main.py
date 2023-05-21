import os
import argparse
import whisper
from tqdm import tqdm
from pathlib import Path
import copy
from faster_whisper import WhisperModel
from translation_service import GoogleTranslateService, M2M100TranslateService

# import whisperx

# from dotenv import load_dotenv

# load_dotenv()
# HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

class SegmentMerger:
    def __init__(self, max_text_len=80, max_duration=15, min_text_len=8, max_segment_interval=2):
        self.max_text_len = max_text_len
        self.max_duration = max_duration
        self.min_text_len = min_text_len
        self.max_segment_interval = max_segment_interval
        self.end_of_sentence_symbols = tuple(['.', '。', ',', '，', '!', '！', '?', '？', ':', '：', '”', ')', ']', '}', ';'])

    def process_segments(self, segments):
        """
        Convert word segments to sentences and merge them if necessary.
        :param segments: List of word segments
        :type segments: list of dicts
        :return: List of merged sentence segments
        :rtype: list of dicts
        """
        
        sentence_results = []
        current_sentence = {"text": "", "start": 0, "end": 0}
        current_sentence_template = {"text": "", "start": 0, "end": 0}
        for segment in segments:
            if current_sentence["text"] == "":
                current_sentence["start"] = segment["start"]
            current_sentence["text"] += segment["word"]
            current_sentence["end"] = segment["end"]
            if self._is_end_of_sentence(segment) or self._is_max_length_exceeded(current_sentence) or self._is_max_duration_exceeded(current_sentence):
                current_sentence["text"] = current_sentence["text"].strip()
                sentence_results.append(copy.deepcopy(current_sentence))
                current_sentence = copy.deepcopy(current_sentence_template)
        return self.merge_segments(sentence_results)

    def merge_segments(self, segments):
        """
        Merge sentence segments to one segment, if the length of the text is less than max_text_len.
        :param segments: List of sentence segments
        :type segments: list of dicts
        :return: List of merged sentence segments
        :rtype: list of dicts
        """
        
        merged_segments = []
        current_segment = {"text": "", "start": 0, "end": 0}
        for i, segment in enumerate(segments):
            if current_segment["text"] == "":
                current_segment["start"] = segment["start"]
            if self._can_merge(current_segment, segment):
                current_segment["text"] += ' ' + segment["text"]
                current_segment["end"] = segment["end"]
            else:
                if current_segment["text"] != "":
                    current_segment["text"] = current_segment["text"].strip()
                    if self._is_too_short(current_segment, i, segments):
                        next_segment = segments[i + 1]
                        if self._can_merge(current_segment, next_segment):
                            current_segment["text"] += ' ' + next_segment["text"]
                            current_segment["end"] = next_segment["end"]
                            continue
                    merged_segments.append(copy.deepcopy(current_segment))
                current_segment = copy.deepcopy(segment)
        if current_segment["text"] != "":
            current_segment["text"] = current_segment["text"].strip()
            merged_segments.append(copy.deepcopy(current_segment))
        return merged_segments

    def _is_end_of_sentence(self, segment):
        return segment["word"][-1] in self.end_of_sentence_symbols

    def _is_max_length_exceeded(self, current_sentence):
        return len(current_sentence["text"]) >= self.max_text_len

    def _is_max_duration_exceeded(self, current_sentence):
        return (current_sentence["end"] - current_sentence["start"]) >= self.max_duration

    def _can_merge(self, current_segment, segment):
        return segment["start"] - current_segment["end"] < self.max_segment_interval and \
               len(current_segment["text"] + " " + segment["text"]) < self.max_text_len and \
               (segment["end"] - current_segment["start"]) < self.max_duration

    def _is_too_short(self, current_segment, i, segments):
        return len(current_segment["text"]) < self.min_text_len and i < len(segments) - 1
 
 
class SubtitleProcessor:
    def __init__(self, video_path, target_language, model, translation_method):
        self.video_path = video_path
        self.target_language = target_language
        self.model = model
        self.translation_method = translation_method
        self.video_language = 'en'
        self.segment_merger = SegmentMerger()

        if translation_method == 'google':
            self.translation_service = GoogleTranslateService()
        elif translation_method == 'm2m100':
            self.translation_service = M2M100TranslateService()
        

    def segments_to_srt(self, segs):
        text = []
        for i, s in tqdm(enumerate(segs)):
            text.append(str(i + 1))

            time_start = s['start']
            hours, minutes, seconds = int(time_start / 3600), (time_start / 60) % 60, (time_start) % 60
            timestamp_start = "%02d:%02d:%06.3f" % (hours, minutes, seconds)
            timestamp_start = timestamp_start.replace('.', ',')
            time_end = s['end']
            hours, minutes, seconds = int(time_end / 3600), (time_end / 60) % 60, (time_end) % 60
            timestamp_end = "%02d:%02d:%06.3f" % (hours, minutes, seconds)
            timestamp_end = timestamp_end.replace('.', ',')
            text.append(timestamp_start + " --> " + timestamp_end)

            formatted_text = s['text'].strip().replace('\n', ' ')
            text.append(formatted_text + "\n")

        return "\n".join(text)

    def transcribe_audio(self):
        if self.model == 'large':
            self.model = 'large-v2'
            
        model = WhisperModel(self.model, device="cuda", compute_type="float16")
        # or run on GPU with INT8
        # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        # or run on CPU with INT8
        # model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        print("Transcribing audio...")
        segments, info = model.transcribe(self.video_path, word_timestamps=True)
        
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        self.video_language = info.language

        words_list = []
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            for word in segment.words:
                # print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
                dict_word = {'word': word.word, 'start': word.start, 'end': word.end}
                words_list.append(dict_word)
            
        # print(words_list)
        # Convert word segments to sentences and merge them
        merged_sentence_segments = self.segment_merger.process_segments(words_list)
        
        # print(merged_sentence_segments)
        srt_sub = self.segments_to_srt(merged_sentence_segments)
        srt_file = os.path.join(os.path.dirname(self.video_path), f'{Path(self.video_path).stem}.srt')
        with open(srt_file, 'w') as f:
            f.write(srt_sub)

        print(f"subtitle is saved at {srt_file}")
        result = {'segments' : merged_sentence_segments, 'language': info.language}
        return result, srt_file
    
    def save_translated_srt(self, segs, translated_text):
        """Save the translated text to a separate SRT file."""
        for i, s in enumerate(segs):
            s["text"] = translated_text[i].strip()

        translated_srt = self.segments_to_srt(segs)
        translated_srt_file = os.path.join(os.path.dirname(self.video_path), f'{Path(self.video_path).stem}_{self.target_language}.srt')
        with open(translated_srt_file, 'w') as f:
            f.write(translated_srt)

        print(f"Translated subtitle is saved at {translated_srt_file}")

    def combine_translated(self, segs, text_translated):
        "Combine the translated text into the 'text' field of segments."
        comb = []
        for s, tr in zip(segs, text_translated):
            seg_copy = copy.deepcopy(s)
            # c = f"{tr.strip()}\\N{s['text'].strip()}\n"
            seg_copy['text'] = tr
            comb.append(seg_copy)
            comb.append(s)

        return comb

    def add_dual_subtitles(self, eng_transcript, translated_transcript):

        print("Combining subtitles...")
        segs_tr = copy.deepcopy(eng_transcript['segments'])
        # print(segs_tr)
        # print(translated_transcript)
        segs_tr = self.combine_translated(segs_tr, translated_transcript)
        sub_tr = self.segments_to_srt(segs_tr)
        sub_translated = os.path.join(os.path.dirname(self.video_path), f'{Path(self.video_path).stem}_dual_sub.srt')
        with open(sub_translated, 'w') as f:
            f.write(sub_tr)

        print(f'combine translated subtitle is saved at {sub_translated}')

    def translate_with_whisper(self, language):
            
        model = WhisperModel('large-v2', device="cuda", compute_type="float16")
        
        print("Transcribing audio...")
        segments, info = model.transcribe(self.video_path, word_timestamps=True, language=language)
        text_list = []
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            text_list.append({'text': segment.text, 'start': segment.start, 'end': segment.end})
        srt_sub = self.segments_to_srt(text_list)
        srt_file = os.path.join(os.path.dirname(self.video_path), f'{Path(self.video_path).stem}_{language}_whisper.srt')
        with open(srt_file, 'w') as f:
            f.write(srt_sub)

        print(f"subtitle is saved at {srt_file}")
        
    def process(self):
        # Transcribe the video
        english_transcript, srt_file = self.transcribe_audio()

        if self.translation_method == 'gpt':
            from translate_gpt import translate_with_gpt
            # Translate the transcript to another language using gpt-3.5 or gpt-4 Translate
            translate_with_gpt(input_file=srt_file, target_language=self.target_language)

        elif self.translation_method == 'whisper':
            self.translate_with_whisper(self.target_language)
            
        else:
            # Translate the transcript to another language
            translated_transcript = self.translation_service.translate(english_transcript, src_lang=self.video_language, tr_lang=self.target_language)
            print(translated_transcript)
            # Save the translated subtitles to a separate SRT file
            segs_tr = copy.deepcopy(english_transcript['segments'])
            
            self.save_translated_srt(segs_tr, translated_transcript)
    
            # Add dual subtitles to the video
            self.add_dual_subtitles(english_transcript, translated_transcript)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download YouTube video, transcribe, translate and add dual subtitles.')
    parser.add_argument('--youtube_url', help='The URL of the YouTube video.', type=str)
    parser.add_argument('--local_video', help='The path to the local video file.', type=str)
    parser.add_argument('--target_language', help='The target language for translation.', default='zh')
    parser.add_argument("--model", help="""Choose one of the Whisper model""", default='small', type=str, choices=['tiny', 'base', 'small', 'medium', 'large'])
    parser.add_argument('--translation_method', help='The method to use for translation. Options: "m2m100" or "google" or "whisper" or "gpt"', default='m2m100', choices=['m2m100', 'google', 'whisper', 'gpt'])

    args = parser.parse_args()

    if args.youtube_url and args.local_video:
        raise ValueError("Cannot provide both 'youtube_url' and 'local_video'. Choose one of them.")

    if not args.youtube_url and not args.local_video:
        raise ValueError("Must provide one of 'youtube_url' or 'local_video'.")

    # Download the YouTube video or use the local video file
    if args.youtube_url:
        from youtube_downloader import YouTubeDownloader
        video_filename = YouTubeDownloader(args.youtube_url, args.target_language).download_video()
    else:
        print("Local video file: " + args.local_video)
        video_filename = args.local_video

    # Create SubtitleProcessor instance and process the video
    subtitle_processor = SubtitleProcessor(video_path=video_filename, target_language=args.target_language, model=args.model, translation_method=args.translation_method)
    subtitle_processor.process()