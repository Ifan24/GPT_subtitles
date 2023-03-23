import os
import argparse
import pysrt
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
from pathlib import Path
import copy
from ffpb import main as ffpb_main


def download_youtube_video(url):
    yt = YouTube(url)
    video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    print('Downloading video: ' + video.title)
    video.download()
    print('Download complete: ' + video.default_filename)
    print(f'File size: {str(video.filesize/1e6)} mb')
    
    return video.default_filename
 
def segments_to_srt(segs):
    text = []
    for i,s in tqdm(enumerate(segs)):
        text.append(str(i+1))

        time_start = s['start']
        hours, minutes, seconds = int(time_start/3600), (time_start/60) % 60, (time_start) % 60
        timestamp_start = "%02d:%02d:%06.3f" % (hours, minutes, seconds)
        timestamp_start = timestamp_start.replace('.',',')     
        time_end = s['end']
        hours, minutes, seconds = int(time_end/3600), (time_end/60) % 60, (time_end) % 60
        timestamp_end = "%02d:%02d:%06.3f" % (hours, minutes, seconds)
        timestamp_end = timestamp_end.replace('.',',')        
        text.append(timestamp_start + " --> " + timestamp_end)

        text.append(s['text'].strip() + "\n")
            
    return "\n".join(text)

def transcribed_text(segs):
    texts = [s['text'] for s in segs]
    text = '\n'.join(texts)
    return text
    
def transcribe_audio(file_path, args):
    model = whisper.load_model(args.model)
    
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
    srt_file = f'{Path(file_path).stem}.srt'
    with open(srt_file, 'w') as f:
        f.write(srt_sub)
        
    print(f"subtitle is saved at {srt_file}")
    
    
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


def combine_translated(segs, text_translated):
    "Combine the translated text into the 'text' field of segments."
    comb = []
    for s, tr in zip(segs, text_translated):
        c = f"{tr.strip()}\\N\\N{s['text'].strip()}\n"
        s['text'] = c 
        comb.append(s)
    return comb

        
def add_dual_subtitles(video_path, eng_transcript, translated_transcript, font_file):

    print("Combining subtitles...")
    segs_tr = copy.deepcopy(eng_transcript['segments'])
    segs_tr = combine_translated(segs_tr, translated_transcript)
    sub_tr = segments_to_srt(segs_tr)
    sub_translated = f'dual_sub_{Path(video_path).stem}.srt'
    with open(sub_translated, 'w') as f:
        f.write(sub_tr)
    
    print(f'translated subtitle is saved at {sub_translated}')
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download YouTube video, transcribe, translate and add dual subtitles.')
    parser.add_argument('--youtube_url', help='The URL of the YouTube video.', type=str)
    parser.add_argument('--local_video', help='The path to the local video file.', type=str)
    parser.add_argument('--target_language', help='The target language for translation.', default='zh')
    parser.add_argument("--model", help="""Choose one of the Whisper model""", default='small', type=str)
    parser.add_argument("--font_path", help="""The path to the local font file for the target language.""", default='msyh.ttc', type=str)
    
    args = parser.parse_args()

    if args.youtube_url and args.local_video:
        raise ValueError("Cannot provide both 'youtube_url' and 'local_video'. Choose one of them.")

    if not args.youtube_url and not args.local_video:
        raise ValueError("Must provide one of 'youtube_url' or 'local_video'.")

    # Download the YouTube video or use the local video file
    if args.youtube_url:
        from pytube import YouTube
        video_filename = download_youtube_video(args.youtube_url)
    else:
        print("Local video file: " + args.local_video)
        video_filename = args.local_video

    # Transcribe the video
    english_transcript = transcribe_audio(video_filename, args)

    # Translate the transcript to another language
    translated_transcript = translate_text(english_transcript, args.target_language)

    # Add dual subtitles to the video
    add_dual_subtitles(video_filename, english_transcript, translated_transcript, args.font_path)

    # Remove the original downloaded video file
    # os.remove(video_filename)

    