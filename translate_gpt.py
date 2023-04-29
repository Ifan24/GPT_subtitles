import openai
import json
import os
from tqdm import tqdm
import re
import argparse
import time
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
    # Add a newline to the end of the string if it doesn't already have one
    if not subtitle_string.endswith('\n'):
        subtitle_string += '\n'
    return len(re.findall(r'(\d+\n(?:.+\n)+)', subtitle_string))
    
def check_response(input_subtitles, translated_subtitles):
    if not translated_subtitles.endswith('\n'):
        translated_subtitles += '\n'
    
    input_blocks = re.findall(r'(\d+\n(?:.+\n)+)', input_subtitles)
    translated_blocks = re.findall(r'(\d+\n(?:.+\n)+)', translated_subtitles)
    additional_content = re.sub(r'\d+\n(?:.+\n)+', '', translated_subtitles).strip()
    
    problematic_blocks = []
    for i, (input_block, translated_block) in enumerate(zip(input_blocks, translated_blocks)):
        input_lines = input_block.strip().split('\n')
        translated_lines = translated_block.strip().split('\n')

        if len(input_lines) != len(translated_lines):
            problematic_blocks.append((i, translated_block))
            continue

        input_line_number = int(input_lines[0])
        translated_line_number = int(translated_lines[0])

        if input_line_number != translated_line_number:
            problematic_blocks.append((i, translated_block))
    
    return len(translated_blocks), additional_content, problematic_blocks



# Translate subtitles check_response wrapper
def translate_gpt(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, target_language="Chinese", subtitles_length=25, titles="Video Title not found", video_information=None, model='gpt-3.5-turbo'):

    translated_subtitles, used_dollars = send_to_openai(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, target_language=target_language, subtitles_length=subtitles_length, titles=titles, video_information=video_information, model=model)
    
    count = 0
    total_used_dollars = used_dollars
    blocks, additional_content, problematic_blocks = check_response(subtitles, translated_subtitles)

    cumulative_warning = ""
    wasted_dollars = 0
    while (blocks != subtitles_length or additional_content or problematic_blocks) and count < 5:
        warning_message = f"Warning: Mismatch in the number of lines ({blocks} != {subtitles_length}), or additional content found ({additional_content}), or problematic blocks ({problematic_blocks}), retry count {count}..."
        print(warning_message)
        cumulative_warning = cumulative_warning + warning_message + "\n"
        translated_subtitles, used_dollars = send_to_openai(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, target_language=target_language, subtitles_length=subtitles_length, titles=titles, warning_message=cumulative_warning, prev_response=translated_subtitles, video_information=video_information, model=model)
        count += 1
        wasted_dollars = total_used_dollars
        total_used_dollars += used_dollars
        blocks, additional_content, problematic_blocks = check_response(subtitles, translated_subtitles)

        
    return translated_subtitles, total_used_dollars, count, wasted_dollars

def send_to_openai(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, target_language="Chinese", subtitles_length=25, titles="Video Title not found", model='gpt-3.5-turbo', warning_message=None, prev_response=None, video_information=None):
    prompt = ""
    if warning_message:
        prompt = f"In a previous request sent to OpenAI, the response is problematic. Please ensure that each translated line corresponds to the same numbered line in the English subtitles, without merging lines or altering the original sentence structure, even if it's unfinished. please do not create the subtitles that are not matching the corresponding line and (!!important) make sure that your reply only contain the {subtitles_length} lines, the translated subtitles have the same number of lines ({subtitles_length}) as the source subtitles. Learn from your mistake. Here is the warning message generated based on your previous response:\n{warning_message}\n\n"
    #     prompt += f"The following lines are for reference only, Previous Response:\n{prev_response}\n\n"
        example = """
    Here is an example of translating English subtitle into Simplified Chinese subtitle:
    1
    Let's talk about the messiah of small bean cinema, otherwise known as Wes Anderson.
    
    2
    By now most of you have likely been seduced at one time or another by this man's meticulously
    
    3
    crafted mise-en-scene, intensely symmetrical composition, or quirky, somewhat disaffected
    
    4
    twee characters.
    
    5
    He piqued our interest with Rushmore, dazzled us with Fantastic Mr. Fox, and stole our little
    
    6
    hearts with Moonrise Kingdom.
    
    1
    让我们来谈谈被誉为小豆电影救世主的韦斯·安德森。
    
    2
    到目前为止，你们中的大多数人可能已经被这位导演的精心
    
    3
    制作的布景、高度对称的构图或者古怪、有些冷漠的
    
    4
    小清新角色所吸引过。
    
    5
    他用《独立思考 (Rushmore)》激起了我们的兴趣，用《了不起的狐狸爸爸 (Fantastic Mr. Fox)》使我们眼花缭乱，
    
    6
    用《月亮王国之恋 (Moonrise Kingdom)》偷走了我们的小心心。
        """
        prompt += example + "\n\n"
        
    if prev_subtitle:
        prompt += f"The following lines are for reference only, Previous subtitle: {prev_subtitle}\n\n"

    if prev_translated_subtitle:
        prompt += f"The following lines are for reference only, Previous translated subtitle: {prev_translated_subtitle}\n\n"
    
    if video_information:
        prompt += f"Additional video information: {video_information}\n\n"
        
    if next_subtitle:
        prompt += f"The following lines are for reference only, Next subtitle: {next_subtitle}\n\n"
    

    # prompt += f"Translate the following subtitles to {target_language} line by line for the video titled '{titles}'. (If a sentence is unfinished, translate the unfinished sentence without merging it with the next line. Please ensure that each translated line corresponds to the same numbered line in the English subtitles, without repetition, and maintain the original sentence structure even if it's unfinished. The translated subtitles should have the same number of lines ({subtitles_length}) as the source subtitles, and the numbering should be maintained.) All the previous text is the prompt, here is the subtitles you need to translate:\n{subtitles}\n"

    # prompt = f"Translate the following subtitles to {target_language} line by line for the video titled '{titles}'. If a sentence is unfinished, translate the unfinished sentence without merging it with the next line. Please ensure that each translated line corresponds to the same numbered line in the English subtitles, without repetition, and maintain the original sentence structure even if it's unfinished. The translated subtitles should have the same number of lines ({subtitles_length}) as the source subtitles, and the numbering should be maintained. Only Reply with the translation of the value of 'subtitles'"
    
    
    # input_data = {
    #     "warning_message": warning_message if warning_message else None,
    #     "previous_subtitle": prev_subtitle,
    #     "next_subtitle": next_subtitle,
    #     "target_language": target_language,
    #     "subtitles_length": subtitles_length,
    #     "titles": titles,
    #     # "prompt": prompt,
    #     "current_subtitles": subtitles,
    # }, 


    prompt += f"Translate the following subtitle:\n{subtitles}\n\n"
    
    
    system_content = ("You are a program responsible for translating subtitles. Your task is to "
     f"translate the subtitles into {target_language} line by line for the "
     f"video titled '{titles}'. Please do not create "
     "the following subtitles on your own. Please do not output any text other than "
     "the translation. You will receive a few lines of previous subtitles, a few "
     "lines of the next subtitle, the translation of the previous subtitle and the "
     "current subtitles. Please ensure that each translated line corresponds to the "
     "same numbered line in the Original subtitles, without repetition. The translated "
     f"subtitles should have the same number of lines ({subtitles_length}) as the source subtitles and "
     "the numbering should be maintained. If the last sentence in the current subtitle "
     "is incomplete, you may combine the translation of the first few words in the "
     "next subtitle to make the sentence complete.  If the first sentence in the "
     "current subtitle is incomplete, you may combine the translation of the last few "
     "words in the last subtitle to make the sentence complete. If you need to merge the subtitles with the following line, "
     f"simply repeat the translation, do not leave the line empty or use a placeholder. Target language: {target_language}")
     
     
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
        # {"role": "system", "content": f"You are a program responsible for translate the following subtitles to {target_language} line by line for the video titled '{titles}'. Your task is to output the specified target language based on the input text. Please do not create the following subtitles on your own. Please do not output any text other than the translation. You will receive a JSON object that contains following keys ('warning_message', 'previous_subtitle', 'next_subtitle', 'target_language', 'subtitles_length', 'titles', 'prompt', 'subtitles'), but you only need to translate the value of key 'subtitles', and reply with the translation only. You will have access to the previous translation results and next few lines of subtitle. If you need to merge the subtitles with the following line, simply repeat the translation. If a sentence is unfinished, translate the unfinished sentence without merging it with the next line. Please ensure that each translated line corresponds to the same numbered line in the English subtitles, without repetition, and maintain the original sentence structure even if it's unfinished. The translated subtitles should have the same number of lines ({subtitles_length}) as the source subtitles, and the numbering should be maintained.\n"},
        # {"role": "user", "content": json.dumps(input_data)}
    ]
    print("========Messages========\n")
    print(messages)
    print("========End of Messages========\n")
    
    inference_not_done = True
    while inference_not_done:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.3,
                top_p=0.5
            )
            inference_not_done = False
        except Exception as e:
            print(f"Waiting 60 seconds")
            print(f"Error was: {e}")
            time.sleep(60)
            
            
    # translated_subtitles = response.choices[0].get("message").get("content").replace('\\n', '\n')
    translated_subtitles = response.choices[0].get("message").get("content").encode("utf8").decode()
    print("========Response========\n")
    print(translated_subtitles)
    
    if model == "gpt-3.5-turbo":
        used_tokens = response['usage']['total_tokens']
        used_dollars = used_tokens / 1000 * 0.002
        print(f"Used tokens: {used_tokens}, Used dollars: {used_dollars}")
    elif model == "gpt-4":
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        used_dollars = (prompt_tokens / 1000 * 0.03) + (completion_tokens / 1000 * 0.06)
        print(f"prompt tokens: {prompt_tokens}, completion tokens: {completion_tokens}, Used dollars: {used_dollars}")
        
    
    print("========End of Response========\n")
    
    return translated_subtitles, used_dollars

def batch_translate_gpt(result, timestamps, batch_size, target_language='zh', model='gpt-3.5-turbo', titles='Video Title not found', video_information=None):
# def batch_translate_gpt(result, timestamps, batch_size, target_language='zh', titles='Video Title not found', video_information=None):
    if target_language == "zh":
        target_language = "Simplified Chinese"
        
    translated = []
    raw_translated = []
    total_dollars = 0
    number_of_retry = 0
    total_wasted_dollars = 0
    prev_translated_subtitle = None
    for i, t in enumerate(tqdm(result)):
        prev_subtitle = result[i-1] if i > 0 else None
        next_subtitle = result[i+1] if i < len(result) - 1 else None
        # t = merge_subtitles_with_timestamps(t, timestamps[i])
        tt, used_dollars, retry_count, wasted_dollars = translate_gpt(t, prev_subtitle, next_subtitle, prev_translated_subtitle, target_language=target_language, subtitles_length=count_blocks(t), titles=titles, model=model, video_information=video_information)
        prev_translated_subtitle = tt
        raw_translated.append(tt)
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
    raw_translated = '\n\n'.join(raw_translated)
    print(translated)
    print("========Translate summary=======\n")
    print(f"total dollars used: {total_dollars:.3f}\n")
    print(f"total number of retry: {number_of_retry}\n")
    print(f"total wasted dollars: {total_wasted_dollars:.3f}\n")
    print("========End of Translate summary=======\n")
    return translated, raw_translated


    
def translate_with_gpt(input_file, batch_size, target_language, model, video_information=None):
    # Extract the file name without the extension
    file_name = os.path.splitext(os.path.basename(input_file))[0]

    # Create the output file name
    output_file = os.path.join(os.path.dirname(input_file), f"{file_name}_{target_language}_gpt.srt")
    
    subtitles_batch, timestamps_batch = load_subtitles(input_file, batch_size=batch_size)
    translated_subtitles, raw_translated_subtitles = batch_translate_gpt(subtitles_batch, timestamps_batch, batch_size, target_language=target_language, model=model, titles=file_name, video_information=video_information)
    save_translated_subtitles(output_file, translated_subtitles)

# Main function
def main():
    parser = argparse.ArgumentParser(description='Translate subtitles using GPT-3.5')
    parser.add_argument('--input_file', help='The path to the input subtitle file.', type=str, required=True)
    # parser.add_argument('--output_file', help='The path to the output subtitle file.', type=str, required=True)
    parser.add_argument('--batch_size', help='The number of subtitles to process in a batch.', type=int, default=3)
    parser.add_argument('--target_language', help='The target language for translation.', default='zh')
    parser.add_argument("-i", "--additional_info", type=str, default="", help="Additional information about the video.")
    parser.add_argument('--model', default='gpt-3.5-turbo', help='Model for OpenAI API', type=str, choices=['gpt-3.5-turbo', 'gpt-4'])
    
    args = parser.parse_args()

    translate_with_gpt(args.input_file, args.batch_size, args.target_language, args.model, args.additional_info)
    

if __name__ == "__main__":
    main()
