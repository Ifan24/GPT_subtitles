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


class Subtitle:
    def __init__(self, file_path):
        self.file_path = file_path
        self.content = self.load_subtitles()

    def load_subtitles(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def save_subtitles(self, file_path, content):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def split_subtitles(self, batch_size):
        subtitle_blocks = self.content.strip().split('\n\n')
        batches = []

        for i in range(0, len(subtitle_blocks), batch_size):
            batch = '\n\n'.join(subtitle_blocks[i:i + batch_size])
            batches.append(batch)

        return batches

    def process_subtitles(self, subtitles):
        lines = subtitles.split('\n')
        processed_lines = []
        timestamps = []

        for line in lines:
            if re.match(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', line):
                timestamps.append(line)
            else:
                processed_lines.append(line)

        return '\n'.join(processed_lines), timestamps


    def get_processed_batches_and_timestamps(self, batch_size):
        subtitle_batches = self.split_subtitles(batch_size)
        processed_batches = []
        timestamps_batches = []
        for batch in subtitle_batches:
            processed_batch, timestamps = self.process_subtitles(batch)
            processed_batches.append(processed_batch)
            timestamps_batches.append(timestamps)
        return processed_batches, timestamps_batches



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
        
def count_blocks(subtitle_string):
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
        

class Translator:
    def __init__(self, model='gpt-3.5-turbo', batch_size=3, target_language='zh', titles='Video Title not found', video_info=None):
        self.model = model
        self.batch_size = batch_size
        self.target_language = target_language
        self.titles = titles
        self.video_info = video_info
        
        if target_language == "zh":
            self.target_language = "Simplified Chinese"
        

    def send_to_openai(self, subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, subtitles_length, warning_message=None, prev_response=None):
        prompt = ""
        if warning_message:
            prompt = f"In a previous request sent to OpenAI, the response is problematic. Please ensure that each translated line corresponds to the same numbered line in the English subtitles, without merging lines or altering the original sentence structure, even if it's unfinished. please do not create the subtitles that are not matching the corresponding line and (!!important) make sure that your reply only contain the {subtitles_length} lines, the translated subtitles have the same number of lines ({subtitles_length}) as the source subtitles. Learn from your mistake. Here is the warning message generated based on your previous response:\n---{warning_message}---\n\n"
            # prompt += f"Previous Response:\n---{prev_response}---\n\n"
            
            example = """
        Here is an example of translating English subtitles into Simplified Chinese subtitles:
        Input:
        1
        Let's talk about the messiah of small bean cinema, otherwise known as Wes Anderson.
        
        2
        By now most of you have likely been seduced at one time or another by this man's meticulously
        
        3
        crafted mise-en-scene, intensely symmetrical composition, or quirky, somewhat disaffected twee characters.
        
        4
        He piqued our interest with Rushmore, dazzled us with Fantastic Mr. Fox, and stole our little
        
        5
        hearts with Moonrise Kingdom.
        
        Output:
        1
        让我们来谈谈被誉为小豆电影救世主的韦斯·安德森。
        
        2
        到目前为止，你们中的大多数人可能已经被这位导演的精心
        
        3
        制作的布景、高度对称的构图或者古怪、有些冷漠的小清新角色所吸引过。
        
        4
        他用《独立思考》(Rushmore) 激起了我们的兴趣，用《了不起的狐狸爸爸》(Fantastic Mr. Fox) 使我们眼花缭乱，
        
        5
        用《月亮王国之恋》(Moonrise Kingdom) 偷走了我们的小心心。
        
            """
            prompt += f"---{example}---\n\n"

        if prev_subtitle:
            prompt += f"Previous subtitle: ---{prev_subtitle}---\n\n"

        if prev_translated_subtitle:
            prompt += f"Previous translated subtitle: ---{prev_translated_subtitle}:---\n\n"

        if self.video_info:
            prompt += f"Additional video information: ---{self.video_info}---\n\n"

        if next_subtitle:
            prompt += f"Next subtitle: ---{next_subtitle}---\n\n"

        prompt += f"Translate the following subtitle:\n```{subtitles}```\n\n"

        system_content = ("You are a program responsible for translating subtitles. Your task is to "
                          f"translate the subtitles delimited by triple backticks into {self.target_language} line by line for the "
                          f"video titled '{self.titles}'. Please do not create "
                          "the following subtitles on your own. Please do not output any text other than "
                          "the translation. You will receive some additional information for your reference only delimited by triple dashes, such as a few lines of previous subtitles, "
                          "a few lines of the next subtitle, the translation of the previous subtitle and maybe error messages."
                          "Please ensure that each translated line corresponds to the "
                          "same numbered line in the Original subtitles, without repetition. The translated "
                          f"subtitles should have the same number of lines ({subtitles_length}) as the source subtitles and "
                          "the numbering should be maintained. If the last sentence in the current subtitle "
                          "is incomplete, you may combine the translation of the first few words in the "
                          "next subtitle to make the sentence complete.  If the first sentence in the "
                          "current subtitle is incomplete, you may combine the translation of the last few "
                          "words in the last subtitle to make the sentence complete. If you need to merge the subtitles with the following line, "
                          f"simply repeat the translation, do not leave the line empty or use a placeholder. Target language: {self.target_language}")

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
        print("========Messages========\n")
        print(messages)
        print("========End of Messages========\n")
        
        inference_not_done = True
        while inference_not_done:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    top_p=0.5
                )
                inference_not_done = False
            except Exception as e:
                print(f"Waiting 60 seconds")
                print(f"Error was: {e}")
                time.sleep(60)

        translated_subtitles = response.choices[0].get("message").get("content").encode("utf8").decode()
        print("========Response========\n")
        print(translated_subtitles)
    
        if self.model == "gpt-3.5-turbo":
            used_tokens = response['usage']['total_tokens']
            used_dollars = used_tokens / 1000 * 0.002
            print(f"Used tokens: {used_tokens}, Used dollars: {used_dollars}")
        elif self.model == "gpt-4":
            prompt_tokens = response['usage']['prompt_tokens']
            completion_tokens = response['usage']['completion_tokens']
            used_dollars = (prompt_tokens / 1000 * 0.03) + (completion_tokens / 1000 * 0.06)
            print(f"prompt tokens: {prompt_tokens}, completion tokens: {completion_tokens}, Used dollars: {used_dollars}")

        return translated_subtitles, used_dollars
        
    # Translate subtitles check_response wrapper
    def translate_subtitles(self, subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle):
        subtitles_length = count_blocks(subtitles)
        translated_subtitles, used_dollars = self.send_to_openai(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, subtitles_length)

        count = 0
        total_used_dollars = used_dollars
        blocks, additional_content, problematic_blocks = check_response(subtitles, translated_subtitles)

        cumulative_warning = ""
        wasted_dollars = 0
        while (blocks != subtitles_length or additional_content or problematic_blocks) and count < 5:
            warning_message = f"Warning: Mismatch in the number of lines ({blocks} != {subtitles_length}), or additional content found ({additional_content}), or problematic blocks ({problematic_blocks}), retry count {count}..."
            print(warning_message)
            cumulative_warning = cumulative_warning + warning_message + "\n"
            translated_subtitles, used_dollars = self.send_to_openai(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, subtitles_length, warning_message=cumulative_warning, prev_response=translated_subtitles)
            count += 1
            wasted_dollars = total_used_dollars
            total_used_dollars += used_dollars
            blocks, additional_content, problematic_blocks = check_response(subtitles, translated_subtitles)

        return translated_subtitles, total_used_dollars, count, wasted_dollars

    def batch_translate(self, subtitle_batches, timestamps_batches):


        translated = []
        raw_translated = []
        total_dollars = 0
        number_of_retry = 0
        total_wasted_dollars = 0
        prev_translated_subtitle = None
        for i, t in enumerate(tqdm(subtitle_batches)):
            prev_subtitle = subtitle_batches[i - 1] if i > 0 else None
            next_subtitle = subtitle_batches[i + 1] if i < len(subtitle_batches) - 1 else None
            def extract_line(text):
                if text is None:
                    return None
                lines = text.split('\n', 3)
                first_two_lines = '\n'.join(lines[:3])
                return first_two_lines
            next_subtitle = extract_line(next_subtitle)
            
            tt, used_dollars, retry_count, wasted_dollars = self.translate_subtitles(t, prev_subtitle, next_subtitle, prev_translated_subtitle)
            prev_translated_subtitle = tt
            raw_translated.append(tt)
            tt_merged = merge_subtitles_with_timestamps(tt, timestamps_batches[i])
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

def translate_with_gpt(input_file, batch_size, target_language, model, video_info=None):
    # Extract the file name without the extension
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    
    subtitle = Subtitle(input_file)
    translator = Translator(model=model, batch_size=batch_size, target_language=target_language, titles=file_name, video_info=video_info)

    subtitle_batches, timestamps_batches = subtitle.get_processed_batches_and_timestamps(batch_size)
    translated_subtitles, raw_translated_subtitles = translator.batch_translate(subtitle_batches, timestamps_batches)

    output_file = os.path.join(os.path.dirname(input_file), f"{os.path.splitext(os.path.basename(input_file))[0]}_{target_language}_gpt.srt")
    subtitle.save_subtitles(output_file, translated_subtitles)
    
    
def main():
    parser = argparse.ArgumentParser(description='Translate subtitles using GPT')
    parser.add_argument('-i', '--input_file', help='The path to the input subtitle file.', type=str, required=True)
    # parser.add_argument('-o', '--output_file', help='The path to the output subtitle file.', type=str, required=True)
    parser.add_argument('-b', '--batch_size', help='The number of subtitles to process in a batch.', type=int, default=3)
    parser.add_argument('-l', '--language', help='The target language for translation.', default='zh')
    parser.add_argument('-v', "--video_info", type=str, default="", help="Additional information about the video.")
    parser.add_argument('-m', '--model', default='gpt-3.5-turbo', help='Model for OpenAI API', type=str, choices=['gpt-3.5-turbo', 'gpt-4'])
    
    args = parser.parse_args()

    translate_with_gpt(args.input_file, args.batch_size, args.language, args.model, args.video_info)



if __name__ == "__main__":
    main()