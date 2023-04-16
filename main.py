import os
import argparse
import whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
from pathlib import Path
import copy
from pythumb import Thumbnail
import subprocess
from googletrans import Translator
from pytube import YouTube

# import spacy
# spacy.prefer_gpu()
# import pysrt
# from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
# from ffpb import main as ffpb_main


def download_youtube_video(url):
    yt = YouTube(url)
    print('Downloading video: ' + yt.title)

    # Create a folder called 'videos' if it does not exist
    if not os.path.exists('videos'):
        os.makedirs('videos')

    # Create a folder with the video title inside the "videos" folder
    video_folder = os.path.join("videos", yt.title)
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
        
    # Download the thumbnail using pythumb
    thumbnail = Thumbnail(url)
    thumbnail.fetch()
    thumbnail.save(dir=video_folder, filename='thumbnail', overwrite=True)
    print(f'Thumbnail saved at: {video_folder}')

    # Download the video using yt-dlp
    output_filename = os.path.join(video_folder, f"{yt.title}.%(ext)s")
    youtube_dl_command = f"yt-dlp -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]' --merge-output-format mp4 -o \"{output_filename}\" {url}"
    completed_process = subprocess.run(youtube_dl_command, shell=True, check=True, text=True)

    # Find the downloaded video file
    for file in os.listdir(video_folder):
        if file.endswith(".mp4"):
            downloaded_video_path = os.path.join(video_folder, file)
            break
    # downloaded_video_path = os.path.join(video_folder, f"{yt.title}.mp4")
    print('Download complete: ' + downloaded_video_path)
    print(f'File size: {os.path.getsize(downloaded_video_path)/1e6} mb')
    
    return downloaded_video_path
 
def segments_to_srt(segs):
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

def transcribed_text(segs):
    texts = [s['text'] for s in segs]
    text = '\n'.join(texts)
    return text
    
def transcribe_audio(file_path, whisper_model):
    model = whisper.load_model(whisper_model)
    
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)
    
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    # detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"Detected language: {detected_language}")
    
    print("Transcribing audio...")
    options = whisper.DecodingOptions(fp16=False, language=detected_language)
    result = model.transcribe(file_path, **options.__dict__, verbose=False)
    
        
    srt_sub = segments_to_srt(result['segments'])
    srt_file = os.path.join(os.path.dirname(file_path), f'{Path(file_path).stem}.srt')
    with open(srt_file, 'w') as f:
        f.write(srt_sub)
        
    print(f"subtitle is saved at {srt_file}")
    
    
    return result
    
def translate_text_with_whisper(file_path, whisper_model, target_language):

    print("Translating text with Whisper...")
    model = whisper.load_model(whisper_model)
    options = whisper.DecodingOptions(fp16=False, language=target_language, task="transcribe")
    result = model.transcribe(file_path, **options.__dict__, verbose=False)
    
    srt_sub = segments_to_srt(result['segments'])
    srt_file = os.path.join(os.path.dirname(file_path), f'{Path(file_path).stem}_{target_language}.srt')
    with open(srt_file, 'w') as f:
        f.write(srt_sub)
        
    print(f"translated subtitle is saved at {srt_file}")
    
    
    return result

def batch_text(result, gs=32):
    """split list into small groups of group size `gs`."""
    segs = result['segments']
    length = len(segs)
    mb = length // gs
    text_batches = []
    for i in range(mb):
        text_batches.append([s['text'] for s in segs[i*gs:(i+1)*gs]])
    if mb*gs != length:
        text_batches.append([s['text'] for s in segs[mb*gs:length]])
    return text_batches


def _translate(text, tokenizer, model_tr, src_lang='en', tr_lang='zh'):
    tokenizer.src_lang = src_lang
    encoded_en = tokenizer(text, return_tensors="pt", padding=True)
    generated_tokens = model_tr.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id(tr_lang))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def batch_translate(texts, tokenizer, model_tr, src_lang='en', tr_lang='zh'):
    translated = []
    for t in tqdm(texts):
        # preprocessed = preprocess_text(' '.join(t), lang=src_lang)
        tt = _translate(t, tokenizer, model_tr, src_lang=src_lang, tr_lang=tr_lang)
        translated += tt
    return translated


def translate_text(result, target_language):
    print("Translating text...")
    model_tr = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    texts = batch_text(result, gs=32)
    texts_tr = batch_translate(texts, tokenizer, model_tr, src_lang=result['language'], tr_lang=target_language)

    return texts_tr
    
def translate_text_google(text, src_lang='en', tr_lang='zh-cn'):
    translator = Translator()
    translated = []
    for t in text:
        translation = translator.translate(t, src=src_lang, dest=tr_lang)
        translated.append(translation.text)
    return translated

def batch_translate_google(texts, src_lang='en', tr_lang='zh-cn'):
    if tr_lang == 'zh':
        tr_lang = 'zh-cn'
    
    translated = []

    for t in tqdm(texts):
        # preprocessed = preprocess_text(' '.join(t), lang=src_lang)
        tt = translate_text_google(t, src_lang=src_lang, tr_lang=tr_lang)
        translated += tt
    return translated


def save_translated_srt(segs, translated_text, video_path, target_language):
    """Save the translated text to a separate SRT file."""
    for i, s in enumerate(segs):
        s["text"] = translated_text[i].strip()

    translated_srt = segments_to_srt(segs)
    translated_srt_file = os.path.join(os.path.dirname(video_path), f'{Path(video_path).stem}_{target_language}.srt')
    with open(translated_srt_file, 'w') as f:
        f.write(translated_srt)

    print(f"Translated subtitle is saved at {translated_srt_file}")
    
    
def combine_translated(segs, text_translated):
    "Combine the translated text into the 'text' field of segments."
    comb = []
    for s, tr in zip(segs, text_translated):
        seg_copy = copy.deepcopy(s)
        # c = f"{tr.strip()}\\N{s['text'].strip()}\n"
        seg_copy['text'] = tr 
        comb.append(seg_copy)
        comb.append(s)
        
    return comb


def add_dual_subtitles(video_path, eng_transcript, translated_transcript):

    print("Combining subtitles...")
    segs_tr = copy.deepcopy(eng_transcript['segments'])
    # print(segs_tr)
    # print(translated_transcript)
    segs_tr = combine_translated(segs_tr, translated_transcript)
    sub_tr = segments_to_srt(segs_tr)
    sub_translated = os.path.join(os.path.dirname(video_path), f'{Path(video_path).stem}_dual_sub.srt')
    with open(sub_translated, 'w') as f:
        f.write(sub_tr)
    
    print(f'combine translated subtitle is saved at {sub_translated}')
    
    
    # TODO: embed the subtitle into the video is not working yet, because the font in other language is not found
    
    # output_file = 'dual_sub_' + os.path.basename(video_path)
    # print("Writing video to file: " + output_file)

    
    # video = VideoFileClip(video_path)
    # subtitles = pysrt.open('Dune_Silence_Warner_Bros_Entertainment.srt')
    # subtitle_clips = []
    # for sub in subtitles:
    #     txt_clip = TextClip(
    #         sub.text,
    #         fontsize=24,
    #         color='white',
    #         size=video.size,
    #         method="caption",
    #         align="South",
    #         stroke_color="black",
    #         stroke_width=1,
    #     )
    #     txt_clip = txt_clip.set_start(sub.start.to_time()).set_duration(sub.duration.to_time())
    #     subtitle_clips.append(txt_clip)
    
    # composite = CompositeVideoClip([video] + subtitle_clips)
    # composite.write_videofile(output_file, codec="libx264")


    # ffmpeg_args = [
    #     "-i", video_path,
    #     "-vf", f"subtitles={sub_translated}",
    #     output_file
    # ]
    # # font_path = os.path.abspath(args.font_path)
    # # ffmpeg_args = [
    # #     "-i", video_path,
    # #     "-vf", f"subtitles={sub_translated}:fontsdir=./:fontfile={font_path}",
    # #     output_file
    # # ]
    # ffpb_main(argv=ffmpeg_args)

# def preprocess_text(text, lang='en'):
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(text)
#     preprocessed = []
#     for sent in doc.sents:
#         preprocessed.append(sent.text.capitalize())
#     return preprocessed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download YouTube video, transcribe, translate and add dual subtitles.')
    parser.add_argument('--youtube_url', help='The URL of the YouTube video.', type=str)
    parser.add_argument('--local_video', help='The path to the local video file.', type=str)
    parser.add_argument('--target_language', help='The target language for translation.', default='zh')
    parser.add_argument("--model", help="""Choose one of the Whisper model""", default='small', type=str, choices=['tiny', 'base', 'small', 'medium', 'large'])
    # parser.add_argument("--font_path", help="""The path to the local font file for the target language.""", default='msyh.ttc', type=str)
    parser.add_argument('--translation_method', help='The method to use for translation. Options: "m2m100" or "google" or "whisper"', default='m2m100', choices=['m2m100', 'google', 'whisper'])
    # parser.add_argument('--fp16', help='Enable fp16 (mixed precision) decoding. (default: False)', action='store_true')

    args = parser.parse_args()

    if args.youtube_url and args.local_video:
        raise ValueError("Cannot provide both 'youtube_url' and 'local_video'. Choose one of them.")

    if not args.youtube_url and not args.local_video:
        raise ValueError("Must provide one of 'youtube_url' or 'local_video'.")

    # Download the YouTube video or use the local video file
    if args.youtube_url:
        video_filename = download_youtube_video(args.youtube_url)
    else:
        print("Local video file: " + args.local_video)
        video_filename = args.local_video

    # Transcribe the video
    english_transcript = transcribe_audio(video_filename, args.model)

    if args.translation_method == 'whisper':
        # Translate the transcript to another language using Whisper
        # it is better to use large model for translating
        if args.model != 'large':
            print("Whisper model is not large, it is better to use large model for translating. (default: small)")
            
        translated_transcript = translate_text_with_whisper(video_filename, args.model, args.target_language)
        
    else:
        if args.translation_method == 'm2m100':
            # Translate the transcript to another language using M2M100
            translated_transcript = translate_text(english_transcript, args.target_language)
        elif args.translation_method == 'google':
            # Translate the transcript to another language using Google Translate
            texts = batch_text(english_transcript, gs=25)
            translated_transcript = batch_translate_google(texts, src_lang=english_transcript['language'], tr_lang=args.target_language)
    
        # Save the translated subtitles to a separate SRT file
        segs_tr = copy.deepcopy(english_transcript['segments'])
        save_translated_srt(segs_tr, translated_transcript, video_filename, args.target_language)

    # Add dual subtitles to the video
    add_dual_subtitles(video_filename, english_transcript, translated_transcript)

    # Remove the original downloaded video file
    # os.remove(video_filename)

    