import os
import argparse
import whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
from pathlib import Path
import copy
from googletrans import Translator
import time
from faster_whisper import WhisperModel

# import whisperx

# from dotenv import load_dotenv

# load_dotenv()
# HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


def word_segment_to_sentence(segments, max_text_len=80, max_duration=15):
    """
    Convert word segments to sentences.
    :param segments: [{"word": "Hello,", "start": 1.1, "end": 2.2}, {"word": "World!", "start": 3.3, "end": 4.4}]
    :type segments: list of dicts
    :return: Segments, but with sentences instead of words.
    :rtype: list of dicts  [{"text": "Hello, World!", "start": 1.1, "end": 4.4}]
    """
    
    
    end_of_sentence_symbols = tuple(['.', '。', ',', '，', '!', '！', '?', '？', ':', '：', '”', ')', ']', '}', ';'])
    sentence_results = []
    
    current_sentence = {"text": "", "start": 0, "end": 0}
    current_sentence_template = {"text": "", "start": 0, "end": 0}
    
    for segment in segments:
        if current_sentence["text"] == "":
            current_sentence["start"] = segment["start"]
        current_sentence["text"] += segment["word"]
        current_sentence["end"] = segment["end"]
        
        # Check if the segment ends a sentence, or if the current sentence exceeds the max length or duration
        if segment["word"][-1] in end_of_sentence_symbols or \
           len(current_sentence["text"]) >= max_text_len or \
           (current_sentence["end"] - current_sentence["start"]) >= max_duration:
            # Trim leading/trailing whitespace and add the sentence to the results
            current_sentence["text"] = current_sentence["text"].strip()
            sentence_results.append(copy.deepcopy(current_sentence))
            
            # Reset the current sentence
            current_sentence = copy.deepcopy(current_sentence_template)
            
    return sentence_results


def sentence_segments_merger(segments, min_text_len=8, max_text_len=80, max_segment_interval=2, max_duration=15):
    """
    Merge sentence segments to one segment, if the length of the text is less than max_text_len.
    :param segments: [{"text": "Hello, World!", "start": 1.1, "end": 4.4}, {"text": "Hello, World!", "start": 1.1, "end": 4.4}]
    :type segments: list of dicts
    :param max_text_len: Max length of the text
    :type max_text_len: int
    :return: Segments, but with merged sentences.
    :rtype: list of dicts  [{"text": "Hello, World! Hello, World!", "start": 1.1, "end": 4.4}]
    """
    
    merged_segments = []
    current_segment = {"text": "", "start": 0, "end": 0}
    
    for i, segment in enumerate(segments):
        if current_segment["text"] == "":
            current_segment["start"] = segment["start"]

        # Check if the segment can be merged with the current merged segment
        if segment["start"] - current_segment["end"] < max_segment_interval and \
           len(current_segment["text"] + " " + segment["text"]) < max_text_len and \
           (segment["end"] - current_segment["start"]) < max_duration:
            # Merge the segment with the current merged segment
            current_segment["text"] += ' ' + segment["text"]
            current_segment["end"] = segment["end"]
        else:
            # If the current merged segment is not empty, trim leading/trailing whitespace
            if current_segment["text"] != "":
                current_segment["text"] = current_segment["text"].strip()
                
                # If the current merged segment is too short and there's a next segment
                if len(current_segment["text"]) < min_text_len and i < len(segments) - 1:
                    next_segment = segments[i + 1]
                    
                    # Check if the current merged segment can be merged with the next segment
                    if len(current_segment["text"] + " " + next_segment["text"]) < max_text_len:
                        # Merge the current merged segment with the next segment
                        current_segment["text"] += ' ' + next_segment["text"]
                        current_segment["end"] = next_segment["end"]
                        continue  # Skip the next segment in the next iteration
                
                # Add the current merged segment to the list of merged segments
                merged_segments.append(copy.deepcopy(current_segment))
            
            # Set the current merged segment to the current segment
            current_segment = copy.deepcopy(segment)
    
    # Append the last segment if not empty
    if current_segment["text"] != "":
        current_segment["text"] = current_segment["text"].strip()
        merged_segments.append(copy.deepcopy(current_segment))
        
    return merged_segments
    
class SubtitleProcessor:
    def __init__(self, video_path, target_language, model, translation_method):
        self.video_path = video_path
        self.target_language = target_language
        self.model = model
        self.translation_method = translation_method
        self.video_language = 'en'

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

    def transcribed_text(self, segs):
        texts = [s['text'] for s in segs]
        text = '\n'.join(texts)
        return text
  
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
            
        print(words_list)
        # Convert word segments to sentences
        sentence_segments = word_segment_to_sentence(words_list)
        
        # Merge sentence segments if they meet certain conditions
        merged_sentence_segments = sentence_segments_merger(sentence_segments)
        
        print(merged_sentence_segments)
        srt_sub = self.segments_to_srt(merged_sentence_segments)
        srt_file = os.path.join(os.path.dirname(self.video_path), f'{Path(self.video_path).stem}.srt')
        with open(srt_file, 'w') as f:
            f.write(srt_sub)

        print(f"subtitle is saved at {srt_file}")
        result = {'segments' : merged_sentence_segments, 'language': info.language}
        return result, srt_file

    def translate_text_with_whisper(self):

        print("Translating text with Whisper...")
        model = whisper.load_model(self.model)
        options = whisper.DecodingOptions(fp16=False, language=self.target_language, task="transcribe")
        result = model.transcribe(self.video_path, **options.__dict__, verbose=False)

        srt_sub = self.segments_to_srt(result['segments'])
        srt_file = os.path.join(os.path.dirname(self.video_path), f'{Path(self.video_path).stem}_{self.target_language}.srt')
        with open(srt_file, 'w') as f:
            f.write(srt_sub)

        print(f"translated subtitle is saved at {srt_file}")

        return result

    def batch_text(self, result, gs=32):
        """split list into small groups of group size `gs`."""
        segs = result['segments']
        length = len(segs)
        mb = length // gs
        text_batches = []
        for i in range(mb):
            text_batches.append([s['text'] for s in segs[i * gs:(i + 1) * gs]])
        if mb * gs != length:
            text_batches.append([s['text'] for s in segs[mb * gs:length]])
        return text_batches

    def _translate(self, text, tokenizer, model_tr, src_lang='en', tr_lang='zh'):
        tokenizer.src_lang = src_lang
        encoded_en = tokenizer(text, return_tensors="pt", padding=True)
        generated_tokens = model_tr.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id(tr_lang))
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def batch_translate(self, texts, tokenizer, model_tr, src_lang='en', tr_lang='zh'):
        translated = []
        for t in tqdm(texts):
            tt = self._translate(t, tokenizer, model_tr, src_lang=src_lang, tr_lang=tr_lang)
            translated += tt
        return translated

    def translate_text(self, result):
        print("Translating text...")
        model_tr = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

        texts = self.batch_text(result, gs=32)
        texts_tr = self.batch_translate(texts, tokenizer, model_tr, src_lang=self.video_language, tr_lang=self.target_language)

        return texts_tr

    def translate_text_google(self, text, src_lang='en', tr_lang='zh-cn'):
        translator = Translator()
        translated = []
        for t in text:
            # translation = translator.translate(t, src=src_lang, dest=tr_lang)
            inference_not_done = True
            while inference_not_done:
                try:
                    translation = translator.translate(t, src=src_lang, dest=tr_lang)
                    inference_not_done = False
                except Exception as e:
                    print(f"Waiting 15 seconds")
                    print(f"Error was: {e}")
                    time.sleep(15)

            translated.append(translation.text)
        return translated

    def batch_translate_google(self, result, src_lang='en', tr_lang='zh-cn'):
        if tr_lang == 'zh':
            tr_lang = 'zh-cn'

        texts = self.batch_text(result, gs=25)
        translated = []

        for t in tqdm(texts):
            tt = self.translate_text_google(t, src_lang=src_lang, tr_lang=tr_lang)
            translated += tt
        return translated

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

    def process(self):
        # Transcribe the video
        english_transcript, srt_file = self.transcribe_audio()

        if self.translation_method == 'gpt':
            from translate_gpt import translate_with_gpt
            # Translate the transcript to another language using gpt-3.5 or gpt-4 Translate
            translate_with_gpt(input_file=srt_file, target_language=self.target_language)

        else:
            if self.translation_method == 'whisper':
                # Translate the transcript to another language using Whisper
                # it is better to use large model for translating
                if self.model != 'large':
                    print("Whisper model is not large, it is better to use large model for translating. (default: small)")

                translated_transcript = self.translate_text_with_whisper()

            else:
                if self.translation_method == 'm2m100':
                    # Translate the transcript to another language using M2M100
                    translated_transcript = self.translate_text(english_transcript)
                elif self.translation_method == 'google':
                    # Translate the transcript to another language using Google Translate
                    translated_transcript = self.batch_translate_google(english_transcript, src_lang=self.video_language, tr_lang=self.target_language)
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