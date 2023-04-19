import openai
import json
import os
from tqdm import tqdm
import re
import argparse

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

def check_response(subtitle_string):
    if not subtitle_string.endswith('\n'):
        subtitle_string += '\n'
    
    subtitle_blocks = re.findall(r'(\d+\n(?:.+\n)+)', subtitle_string)
    additional_content = re.sub(r'\d+\n(?:.+\n)+', '', subtitle_string).strip()
    
    problematic_blocks = []
    for block in subtitle_blocks:
        lines = block.strip().split('\n')
        if len(lines) > 2:
            problematic_blocks.append(block)
            continue

        english_chars = sum(c.isascii() and c.isalpha() for c in lines[1])
        total_chars = len(lines[1])
        english_ratio = english_chars / total_chars

        if english_ratio > 0.7:
            problematic_blocks.append(block)
    
    return len(subtitle_blocks), additional_content, problematic_blocks


# Translate subtitles check_response wrapper
def translate_gpt(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, source_language="English", target_language="Chinese", subtitles_length=25, titles="Video Title not found",):
    
    translated_subtitles, used_dollars = send_to_openai(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, source_language=source_language, target_language=target_language, subtitles_length=subtitles_length, titles=titles)
    count = 0
    total_used_dollars = used_dollars
    blocks, additional_content, problematic_blocks = check_response(translated_subtitles)
    cumulative_warning = ""
    wasted_dollars = 0
    while (blocks != subtitles_length or additional_content or problematic_blocks) and count < 5:
        warning_message = f"Warning: Mismatch in the number of lines ({blocks} != {subtitles_length}), or additional content found ({additional_content}), or problematic blocks ({problematic_blocks}), retry count {count}..."
        print(warning_message)
        cumulative_warning = cumulative_warning + warning_message + "\n"
        translated_subtitles, used_dollars = send_to_openai(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, source_language=source_language, target_language=target_language, subtitles_length=subtitles_length, titles=titles, warning_message=cumulative_warning)
        count += 1
        wasted_dollars = total_used_dollars
        total_used_dollars += used_dollars
        blocks, additional_content, problematic_blocks = check_response(translated_subtitles)
        
    return translated_subtitles, total_used_dollars, count, wasted_dollars


def send_to_openai(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, source_language="English", target_language="Chinese", subtitles_length=25, titles="Video Title not found", warning_message=None):
    prompt = ""
    if warning_message:
        prompt = f"In a previous request sent to OpenAI, the response is problematic. Please ensure that each translated line corresponds to the same numbered line in the English subtitles, without merging lines or altering the original sentence structure, even if it's unfinished. please do not create the subtitles that are not matching the corresponding line and (!!important) make sure that your reply only contain the {subtitles_length} lines, the translated subtitles have the same number of lines ({subtitles_length}) as the source subtitles. Learn from your mistake. Here is the warning message generated based on your previous response:\n{warning_message}\n"
    
    if prev_subtitle:
        prompt += f"Previous subtitle: {prev_subtitle}\n"

    if next_subtitle:
        prompt += f"Next subtitle: {next_subtitle}\n"
    
    if prev_translated_subtitle:
        prompt += f"If you need to merge the subtitles with the previous line, simply repeat the previous translation. Previous translated subtitle: {prev_translated_subtitle}\n"
        
        
    prompt += f"Translate the following {source_language} subtitles to {target_language} line by line for the video titled '{titles}'. (If a sentence is unfinished, translate the unfinished sentence without merging it with the next line. Please ensure that each translated line corresponds to the same numbered line in the English subtitles, without repetition, and maintain the original sentence structure even if it's unfinished. The translated subtitles should have the same number of lines ({subtitles_length}) as the source subtitles, and the numbering should be maintained.) All the previous text is the prompt, here is the subtitles you need to translate:\n{subtitles}\n"

    # prompt = f"Translate the following {source_language} subtitles to {target_language} line by line for the video titled '{titles}'. If a sentence is unfinished, translate the unfinished sentence without merging it with the next line. Please ensure that each translated line corresponds to the same numbered line in the English subtitles, without repetition, and maintain the original sentence structure even if it's unfinished. The translated subtitles should have the same number of lines ({subtitles_length}) as the source subtitles, and the numbering should be maintained. Only Reply with the translation of the value of 'subtitles'"
    
    # input_data = {
    #     "warning_message": warning_message if warning_message else None,
    #     "previous_subtitle": prev_subtitle,
    #     "next_subtitle": next_subtitle,
    #     "source_language": source_language,
    #     "target_language": target_language,
    #     "subtitles_length": subtitles_length,
    #     "titles": titles,
    #     # "prompt": prompt,
    #     "subtitles": subtitles,
    # }, 

    messages = [
        {"role": "system", "content": f"You are a program responsible for translating subtitles. Your task is to output the specified target language based on the input text. Please do not create the following subtitles on your own. Please do not output any text other than the translation. You will receive the subtitles as an array that needs to be translated, as well as the previous translation results and next subtitle. If you need to merge the subtitles with the following line, simply repeat the translation.\n"},
        {"role": "user", "content": prompt}
        # {"role": "system", "content": f"You are a program responsible for translate the following {source_language} subtitles to {target_language} line by line for the video titled '{titles}'. Your task is to output the specified target language based on the input text. Please do not create the following subtitles on your own. Please do not output any text other than the translation. You will receive a JSON object that contains following keys ('warning_message', 'previous_subtitle', 'next_subtitle', 'source_language', 'target_language', 'subtitles_length', 'titles', 'prompt', 'subtitles'), but you only need to translate the value of key 'subtitles', and reply with the translation only. You will have access to the previous translation results and next few lines of subtitle. If you need to merge the subtitles with the following line, simply repeat the translation. If a sentence is unfinished, translate the unfinished sentence without merging it with the next line. Please ensure that each translated line corresponds to the same numbered line in the English subtitles, without repetition, and maintain the original sentence structure even if it's unfinished. The translated subtitles should have the same number of lines ({subtitles_length}) as the source subtitles, and the numbering should be maintained.\n"},
        # {"role": "user", "content": json.dumps(input_data)}
    ]
    print("========Messages========\n")
    print(messages)
    print("========End of Messages========\n")
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.4,
    )
    # translated_subtitles = response.choices[0].get("message").get("content").replace('\\n', '\n')
    translated_subtitles = response.choices[0].get("message").get("content").encode("utf8").decode()
    print("========Response========\n")
    print(translated_subtitles)
    
    used_tokens = response['usage']['total_tokens']
    used_dollars = used_tokens / 1000 * 0.002
    print(f"Used tokens: {used_tokens}, Used dollars: {used_dollars}")
    print("========End of Response========\n")
    
    return translated_subtitles, used_dollars

    
def batch_translate_gpt(result, timestamps, batch_size, source_language='en', target_language='zh', titles='Video Title not found'):
    if target_language == "zh":
        target_language = "Simplified Chinese"
    if source_language == 'en':
        source_language = 'English'
        
    translated = []
    total_dollars = 0
    number_of_retry = 0
    total_wasted_dollars = 0
    prev_translated_subtitle = None
    for i, t in enumerate(tqdm(result)):
        prev_subtitle = result[i-1] if i > 0 else None
        next_subtitle = result[i+1] if i < len(result) - 1 else None
        tt, used_dollars, retry_count, wasted_dollars = translate_gpt(t, prev_subtitle, next_subtitle, prev_translated_subtitle, source_language=source_language, target_language=target_language, subtitles_length=batch_size, titles=titles)
        prev_translated_subtitle = tt
        tt_merged = merge_subtitles_with_timestamps(tt, timestamps[i])
        total_dollars += used_dollars
        number_of_retry += retry_count
        total_wasted_dollars += wasted_dollars
        print("========Batch summary=======\n")
        print(f"total dollars used: {total_dollars:.3f}\n")
        print(f"total number of retry: {number_of_retry}\n")
        print(f"total wasted dollars: {total_wasted_dollars:.3f}\n")
        print("==============\n")
        print(t)
        print("==============\n")
        print(tt_merged)
        print("========End of Batch summary=======\n")
        translated.append(tt_merged)
        
    translated = '\n\n'.join(translated)
    print(translated)
    print("========Translate summary=======\n")
    print(f"total dollars used: {total_dollars:.3f}\n")
    print(f"total number of retry: {number_of_retry}\n")
    print(f"total wasted dollars: {total_wasted_dollars:.3f}\n")
    print("========End of Translate summary=======\n")
    return translated


    
def translate_with_gpt(input_file, batch_size, target_language):
    # Extract the file name without the extension
    file_name = os.path.splitext(os.path.basename(input_file))[0]

    # Create the output file name
    output_file = os.path.join(os.path.dirname(input_file), f"{file_name}_{target_language}_gpt.srt")
    
    subtitles_batch, timestamps_batch = load_subtitles(input_file, batch_size=batch_size)
    translated_subtitles = batch_translate_gpt(subtitles_batch, timestamps_batch, batch_size, target_language=target_language, titles=file_name)
    save_translated_subtitles(output_file, translated_subtitles)

# Main function
def main():
    parser = argparse.ArgumentParser(description='Translate subtitles using GPT-3.5')
    parser.add_argument('--input_file', help='The path to the input subtitle file.', type=str, required=True)
    # parser.add_argument('--output_file', help='The path to the output subtitle file.', type=str, required=True)
    parser.add_argument('--batch_size', help='The number of subtitles to process in a batch.', type=int, default=2)
    parser.add_argument('--target_language', help='The target language for translation.', default='zh')
    args = parser.parse_args()

    translate_with_gpt(args.input_file, args.batch_size, args.target_language)
    
# python translate_gpt.py --input_file 'videos/WHAT LIFEWEAVER EXCELS AT IN OVERWATCH 2/WHAT LIFEWEAVER EXCELS AT IN OVERWATCH 2.srt' > output2.txt
    
    
if __name__ == "__main__":
    main()
