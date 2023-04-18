import openai
import os
from tqdm import tqdm
import re

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def split_subtitles(input_string, batch_size):
    subtitle_blocks = input_string.strip().split('\n\n')
    batches = []

    for i in range(0, len(subtitle_blocks), batch_size):
        batch = '\n\n'.join(subtitle_blocks[i:i+batch_size])
        batches.append(batch)

    return batches


def process_subtitles(subtitles):
    lines = subtitles.split('\n')
    processed_lines = []
    timestamps = []

    for line in lines:
        if re.match(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', line):
            timestamps.append(line)
        else:
            processed_lines.append(line)

    return '\n'.join(processed_lines), timestamps

def merge_subtitles_with_timestamps(translated_subtitles, timestamps):
    translated_lines = translated_subtitles.split('\n')
    merged_lines = []

    timestamp_idx = 0
    for line in translated_lines:
        if re.match(r'\d+\s*$', line):
            merged_lines.append(line)
            merged_lines.append(timestamps[timestamp_idx])
            timestamp_idx += 1
        else:
            merged_lines.append(line)

    return '\n'.join(merged_lines)

# Load subtitle file
def load_subtitles(file_path, batch_size=3):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    subtitle_batches = split_subtitles(content, batch_size)
    processed_batches = []
    timestamps_batches = []
    for batch in subtitle_batches:
        processed_batch, timestamps = process_subtitles(batch)
        processed_batches.append(processed_batch)
        timestamps_batches.append(timestamps)
    return processed_batches, timestamps_batches


# Save translated subtitles
def save_translated_subtitles(file_path, translated_content):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(translated_content)

def count_blocks(subtitle_string):
    if not subtitle_string.endswith('\n'):
        subtitle_string += '\n'
    return len(re.findall(r'(\d+\n(?:.+\n)+)', subtitle_string))


# Translate subtitles mismatch wrapper
def translate_gpt(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, source_language="English", target_language="Chinese", subtitles_length=25, titles="Video Title not found",):
    
    translated_subtitles, used_dollars = send_to_openai(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, source_language=source_language, target_language=target_language, subtitles_length=subtitles_length, titles=titles)
    count = 0
    total_used_dollars = used_dollars
    while count_blocks(translated_subtitles) != subtitles_length and count < 3:
        print(f"Warning: Mismatch in the number of lines, retry count {count}...")
        translated_subtitles, used_dollars = send_to_openai(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, source_language=source_language, target_language=target_language, subtitles_length=subtitles_length, titles=titles, mismatch=translated_subtitles)
        count+=1
        total_used_dollars += used_dollars
        
    return translated_subtitles, total_used_dollars, count

def send_to_openai(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, source_language="English", target_language="Chinese", subtitles_length=25, titles="Video Title not found", mismatch=None):
    if prev_subtitle:
        prompt = f"Previous subtitle: {prev_subtitle}\n"
    else:
        prompt = ""

    if next_subtitle:
        prompt += f"Next subtitle: {next_subtitle}\n"
    
    if prev_translated_subtitle:
        prompt += f"If you need to merge the subtitles with the previous line, simply repeat the previous translation. Previous translated subtitle: {prev_translated_subtitle}\n"
        
    prompt += f"Translate the following {source_language} subtitles to {target_language} line by line for the video titled '{titles}'. If a sentence is unfinished, translate the unfinished sentence without merging it with the next line. Please ensure that each translated line corresponds to the same numbered line in the English subtitles, without repetition, and maintain the original sentence structure even if it's unfinished. The translated subtitles should have the same number of lines ({subtitles_length}) as the source subtitles, and the numbering should be maintained:\n{subtitles}\n"

    if mismatch:
        mismatch_message = f"In a previous request sent to OpenAI, the number of lines in the translated subtitles did not match the number of lines in the source subtitles. Please ensure that each translated line corresponds to the same numbered line in the English subtitles, without merging lines or altering the original sentence structure, even if it's unfinished. Here is the previous mismatch response, please do not create the subtitles that are not matching the corresponding line and learn from your mistake:\n{mismatch}\n"
        prompt = mismatch_message + prompt

    # input_data = {
    #     "subtitles": subtitles,
    #     "previous_subtitle": prev_subtitle,
    #     "next_subtitle": next_subtitle,
    #     "source_language": source_language,
    #     "target_language": target_language,
    #     "subtitles_length": subtitles_length,
    #     "titles": titles,
    #     "mismatch": mismatch_message if mismatch else None
    # }

    messages = [
        {"role": "system", "content": f"You are a program responsible for translating subtitles. Your task is to output the specified target language based on the input text. Please do not create the following subtitles on your own. Please do not output any text other than the translation. You will receive the subtitles as an array that needs to be translated, as well as the previous translation results and next subtitle. If you need to merge the subtitles with the following line, simply repeat the translation.\n"},
        {"role": "user", "content": prompt}
    ]
    print("========Messages========\n")
    print(messages)
    print("========End of Messages========\n")
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3,
        top_p=1,
        max_tokens=2048,
    )

    translated_subtitles = response.choices[0].get("message").get("content").encode("utf8").decode()
    print("========Response========\n")
    print(translated_subtitles)
    print("========End of Response========\n")
    
    used_tokens = response['usage']['total_tokens']
    used_dollars = used_tokens / 1000 * 0.002
    
    return translated_subtitles, used_dollars

    
def batch_translate_gpt(result, timestamps, src_lang='en', tr_lang='zh', titles='Video Title not found'):
    if tr_lang == "zh":
        tr_lang = "Simplified Chinese"
    if src_lang == 'en':
        src_lang = 'English'
        
    translated = []
    total_dollars = 0
    number_of_retry = 0
    prev_translated_subtitle = None
    for i, t in enumerate(tqdm(result)):
        prev_subtitle = result[i-1] if i > 0 else None
        next_subtitle = result[i+1] if i < len(result) - 1 else None
        tt, used_dollars, retry_count = translate_gpt(t, prev_subtitle, next_subtitle, prev_translated_subtitle, source_language=src_lang, target_language=tr_lang, subtitles_length=count_blocks(t), titles=titles)
        prev_translated_subtitle = tt
        tt_merged = merge_subtitles_with_timestamps(tt, timestamps[i])
        total_dollars += used_dollars
        number_of_retry += retry_count
        print("========Batch summary=======\n")
        print(f"total dollars used: {total_dollars:.3f}\n")
        print(f"total number of retry: {number_of_retry}\n")
        print(tt_merged)
        print("========End of Batch summary=======\n")
        translated.append(tt_merged)
        
    translated = '\n\n'.join(translated)
    print(translated)
    print("========Translate summary=======\n")
    print(f"total dollars used: {total_dollars:.3f}\n")
    print(f"total number of retry: {number_of_retry}\n")
    print("========End of Translate summary=======\n")
    return translated


    
# Main function
def main():
    input_file = 'videos/4090 ITX Overkill – New Dan C4-SFX/4090 ITX Overkill – New Dan C4-SFX.srt'
    output_file = "videos/4090 ITX Overkill – New Dan C4-SFX/4090 ITX Overkill – New Dan C4-SFX_zh_gpt1.srt"
    
    subtitles_batch, timestamps_batch = load_subtitles(input_file, batch_size=2)
    # print(subtitles_batch)
    translated_subtitles = batch_translate_gpt(subtitles_batch, timestamps_batch)
    save_translated_subtitles(output_file, translated_subtitles)
if __name__ == "__main__":
    main()
