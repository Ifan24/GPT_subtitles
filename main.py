import os
import argparse
import whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
from pathlib import Path
import copy
from googletrans import Translator
import time

class SubtitleProcessor:
    def __init__(self, video_path, target_language, model, translation_method):
        self.video_path = video_path
        self.target_language = target_language
        self.model = model
        self.translation_method = translation_method

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
        model = whisper.load_model(self.model)

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(self.video_path)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # detect the spoken language
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        print(f"Detected language: {detected_language}")

        print("Transcribing audio...")
        options = whisper.DecodingOptions(fp16=False, language=detected_language)
        result = model.transcribe(self.video_path, **options.__dict__, verbose=False)

        srt_sub = self.segments_to_srt(result['segments'])
        srt_file = os.path.join(os.path.dirname(self.video_path), f'{Path(self.video_path).stem}.srt')
        with open(srt_file, 'w') as f:
            f.write(srt_sub)

        print(f"subtitle is saved at {srt_file}")

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
        texts_tr = self.batch_translate(texts, tokenizer, model_tr, src_lang=result['language'], tr_lang=self.target_language)

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
                    translated_transcript = self.batch_translate_google(english_transcript, src_lang=english_transcript['language'], tr_lang=self.target_language)
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