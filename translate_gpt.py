import openai
import ujson
import os
from tqdm import tqdm
import re
import argparse
import time
from dotenv import load_dotenv
import logging

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def count_token(str):
    num_tokens = len(encoding.encode(str))
    return num_tokens
 
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
    def __init__(self, model='gpt-3.5-turbo', batch_size=40, target_language='zh', titles='Video Title not found', video_info=None, input_path=None):
        self.model = model
        self.batch_size = batch_size
        self.target_language = target_language
        self.titles = titles
        self.video_info = video_info
        
        if target_language == "zh":
            self.target_language = "Simplified Chinese"
        
        # Setting up the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(os.path.join(os.path.dirname(input_path), 'translator.log'))
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Setup logger for OpenAI response
        self.openai_logger = logging.getLogger('OpenAI_Response')
        self.openai_logger.setLevel(logging.DEBUG)

        # Create another file handler for OpenAI response
        openai_file_handler = logging.FileHandler(os.path.join(os.path.dirname(input_path), 'response.log'))
        openai_file_handler.setLevel(logging.DEBUG)

        # Create another formatter for OpenAI response
        openai_formatter = logging.Formatter('%(message)s')

        # Set formatter for OpenAI response logger
        openai_file_handler.setFormatter(openai_formatter)

        # Add handler to OpenAI response logger
        self.openai_logger.addHandler(openai_file_handler)
        

    def send_to_openai(self, subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, subtitles_length, warning_message=None, prev_response=None):


        system_content = f"""You are a program responsible for translating subtitles. Your task is to translate the current batch of subtitles into {self.target_language} for the video titled '{self.titles}' and follow the guidelines below.
Guidelines:
- Keep in mind that each index should correspond exactly with the original text, and your translation should be faithful to the context of each sentence.
- Translate with informal slang if necessary, ensuring that the translation is accurate and reflects the context and terminology. Please do not output any text other than the translation. 
- You will also receive some additional information for your reference only, such as the previous batch of subtitles, the translation of the previous batch, the next batch of subtitle, and maybe error messages. 
- Please ensure that each line in the current batch has a corresponding translated line. 
- If the last sentence in the current batch is incomplete, you may ignore the last sentence. If the first sentence in the current batch is incomplete, you may combine the last sentence in the last batch to make the sentence complete. 
- Please only output the translation of the current batch of subtitles (current_batch_subtitles_translation).
- Do not put the translation of the next whole sentence in the current sentence.
- Each index in the current batch of subtitles must correspond to the exact original text and translation. Do not combine sentences from different indices.
- Ensure that the number of lines in the current batch of subtitles is the same as the number of lines in the translation.
- You may translate with conversational language if the original text is informal.
- Additional information for the video: {self.video_info}
- Please ensure that the translation and the original text are matched correctly.
- You may receive original text in other languages, but please only output {self.target_language} translation.

- Target language: {self.target_language}

- Please output proper JSON with this format:
{{
    "current_batch_subtitles_translation": [
        {{
            "index": <int>,
            "original_text": <str>,
            "translation": <str>
        }}
    ]
}}"""

        
        example_user_input1 = '''{
    "previous_batch_subtitles": [
        {
            "index": 23,
            "original_text": "with this issue, though they didn't elaborate on what exactly that would mean.",
            "translation": "有关这个问题，尽管他们没有详细说明这究竟意味着什么。"
        },
        {
            "index": 24,
            "original_text": "Another change they mentioned was the corpse explosion from necromancer that leaves",
            "translation": "他们提到的另一个变化是死灵法师的尸爆"
        }
    ],
    "current_batch_subtitles": [
        {
            "index": 25,
            "original_text": "the shadow dot all over the screen that makes it so people can't see anything that's"
        },
        {
            "index": 26,
            "original_text": "going to be patched so that players can also be able to see what's going on in the"
        },
        {
            "index": 27,
            "original_text": "ground and not stand in harmful AoEs. Another really hot topic in"
        }
    ],
    "next_batch_subtitles": [
        {
            "index": 28,
            "original_text": "this livestream was dungeon elite packs and XP and how"
        }
    ]
}'''
        
        example_assistant_output1 = '''{
    "current_batch_subtitles_translation": [
        {
            "index": 25,
            "original_text": "the shadow dot all over the screen that makes it so people can't see anything that's",
            "translation": "尸爆会让屏幕上布满阴影点，使人们无法看清屏幕上的任何东西，"
            
        },
        {
            "index": 26,
            "original_text": "going to be patched so that players can also be able to see what's going on in the",
            "translation": "这将会被修补，以便玩家能够看清地面上发生的情况，"
            
        },
        {
            "index": 27,
            "original_text": "ground and not stand in harmful AoEs.",
            "translation": "这样不会站在有害的AoEs（范围伤害技能）中。"
            
        }
    ]
}'''
        
        example_user_input2 = '''{
    "current_batch_subtitles": [
        {
            "index": 1,
            "original_text": "So Diablo 2 had a secret cow level which you could only unlock after you found some"
        },
        {
            "index": 2,
            "original_text": "items in the world,"
        }
    ]
}'''
        
        example_assistant_output2 = '''{
    "current_batch_subtitles_translation": [
        {
            "index": 1,
            "original_text": "So Diablo 2 had a secret cow level which you could only unlock after you found some",
            "translation": "暗黑破坏神2里有一个秘密奶牛关卡，你只有在找到一些物品之后才能解锁，"
        },
        {
            "index": 2,
            "original_text": "items in the world,",
            "translation": "这些物品需要在世界中找到，"
        }
    ]
}'''

        example_user_input3 = '''{
    "current_batch_subtitles": [
        {
            "index": 102,
            "original_text": "You may find a chest piece for example that has like three different resistances"
        },
        {
            "index": 103,
            "original_text": "with like 50% or something like that and you think wow this is a lot of damage reduction"
        }
    ]
}'''
        
        example_assistant_output3 = """{
    "current_batch_subtitles_translation": [
        {
            "index": 102,
            "original_text": "You may find a chest piece for example that has like three different resistances",
            "translation": "你可能会看到一个胸甲有3个50%左右的抗性，"
        },
        {
            "index": 103,
            "original_text": "with like 50% or something like that and you think wow this is a lot of damage reduction",
            "translation": "然后你会觉得哇，这是很大的伤害减免，"
        }
    ]
}"""
        
        
        def process_line(line):
            subtitles = []
            lines = line.split("\n")
            
            i = 0
            while i < len(lines):
                # Skip empty lines
                if lines[i].strip() == "":
                    i += 1
                    continue
                
                # Extract number and text
                number = int(lines[i])
                i += 1
                original_text = lines[i]
                
                # Add to subtitles list
                subtitles.append({"index": number, "original_text": original_text})
                
                # Move to next subtitle
                i += 2
                
            return subtitles
        
        user_input = {}
        
        if prev_subtitle:
            previous_subtitles = process_line(prev_subtitle)
            if prev_translated_subtitle:
                translated_subtitles = process_line(prev_translated_subtitle)
                # self.logger.info(previous_subtitles)
                # self.logger.info(f"length of previous_subtitles: {len(previous_subtitles)}")
                # self.logger.info(translated_subtitles)
                # self.logger.info(f"length of translated_subtitles: {len(translated_subtitles)}")
                
                # Create a dictionary for faster lookup
                index_to_translation = {item["index"]: item["original_text"] for item in translated_subtitles}
                
                # Loop through each item in previous_subtitles
                for item in previous_subtitles:
                    # Check if the index exists in translated_subtitles
                    if item["index"] in index_to_translation:
                        # Add the translation to previous_subtitles
                        item["translation"] = index_to_translation[item["index"]]
                    else:
                        # self.logger.info error if index doesn't exist in translated_subtitles
                        self.logger.info(f"Error: index {item['index']} not found in translated subtitles")
                                
                    user_input["previous_batch_subtitles"] = previous_subtitles
        
        if subtitles:
            user_input["current_batch_subtitles"] = process_line(subtitles)
        
        if next_subtitle:
            user_input["next_batch_subtitles"] = process_line(next_subtitle)
            
        if warning_message:
            user_input["Warning_message"] = f"In a previous request sent to OpenAI, the response is problematic. Please double-check your answer. Warning message: {warning_message}"
            
        messages = [
            {"role": "system", "content": system_content},
            
            # {"role": "user", "content": example_user_input1},
            # {"role": "assistant", "content": example_assistant_output1},
            
            # {"role": "user", "content": example_user_input2},
            # {"role": "assistant", "content": example_assistant_output2},
            
            # {"role": "user", "content": example_user_input3},
            # {"role": "assistant", "content": example_assistant_output3},
            
            {"role": "user", "content": ujson.dumps(user_input)},
        ]
        
        self.logger.info("========Messages========\n")
        self.logger.info(messages)
        self.logger.info("========End of Messages========\n")
        
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                answer = ''
                translated_subtitles = ''
                delay_time = 0.01
                start_time = time.time()
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    top_p=0.5,
                    stream=True,
                )
                terminator = self.openai_logger.handlers[0].terminator
                self.openai_logger.handlers[0].terminator = ''
                for event in response: 
                    # STREAM THE ANSWER
                    # print(answer, end='', flush=True) 
                    self.openai_logger.info(answer) 
                    self.openai_logger.handlers[0].flush()
                    # RETRIEVE THE TEXT FROM THE RESPONSE
                    event_time = time.time() - start_time  
                    event_text = event['choices'][0]['delta']
                    answer = event_text.get('content', '')
                    translated_subtitles += answer
                    time.sleep(delay_time)
                
                self.openai_logger.handlers[0].terminator = terminator
                self.openai_logger.info("===========================") 
                self.logger.info("========Response========\n")
                self.logger.info(translated_subtitles)
                
                # Parse JSON string into a Python dictionary
                json_string_with_double_quotes = re.sub(r"(\s*[\{\}:,]\s*)'([^']*)'", r'\1"\2"', translated_subtitles)
                pattern = re.compile(r',\s*}')
                cleaned_json_string = re.sub(pattern, '}', json_string_with_double_quotes)

                data = ujson.loads(cleaned_json_string)
                
                # Extract translations and construct the output string
                output_string = ""
                for subtitle in data["current_batch_subtitles_translation"]:
                    index = subtitle["index"]
                    translation = subtitle["translation"]
                    output_string += f"{index}\n{translation}\n\n"
                
                prompt_tokens = count_token(str(messages))
                completion_tokens = count_token(translated_subtitles)
                if self.model == "gpt-3.5-turbo":
                    # used_tokens = response['usage']['total_tokens']
                    used_dollars = (prompt_tokens / 1000 * 0.0015) + (completion_tokens / 1000 * 0.002)
                    self.logger.info(f"prompt tokens: {prompt_tokens}, completion tokens: {completion_tokens}, Used dollars: {used_dollars}")
                    
                elif self.model == "gpt-3.5-turbo-16k-0613":
                    used_dollars = (prompt_tokens / 1000 * 0.003) + (completion_tokens / 1000 * 0.004)
                    self.logger.info(f"prompt tokens: {prompt_tokens}, completion tokens: {completion_tokens}, Used dollars: {used_dollars}")
                    
                elif self.model == "gpt-4":
                    # prompt_tokens = response['usage']['prompt_tokens']
                    # completion_tokens = response['usage']['completion_tokens']
                    used_dollars = (prompt_tokens / 1000 * 0.03) + (completion_tokens / 1000 * 0.06)
                    self.logger.info(f"prompt tokens: {prompt_tokens}, completion tokens: {completion_tokens}, Used dollars: {used_dollars}")
        
                return output_string, used_dollars

            except ujson.JSONDecodeError as e:
                retry_count += 1
                self.logger.error(f"An error occurred while parsing JSON: {e}. Retrying {retry_count} of {max_retries}.")
                warning_message = "Your response is not in a valid JSON format. Please double-check your answer."
                time.sleep(10) 
            
            except openai.error.APIError as e:
                # Handle API error here, e.g. retry or log
                self.logger.error(f"Waiting 30 seconds")
                self.logger.error(f"OpenAI API returned an API Error: {e}")
                time.sleep(30)
                
            except openai.error.APIConnectionError as e:
                # Handle connection error here
                self.logger.error(f"Waiting 30 seconds, please double-check your internet connection")
                self.logger.error(f"Failed to connect to OpenAI API: {e}")
                time.sleep(30)
                
            except openai.error.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                self.logger.error(f"Waiting 60 seconds")
                self.logger.error(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(60)
  
            except Exception as e:
                self.logger.error(f"An unexpected error occurred: {e}")
                return None, None 

        self.logger.error("Max retries reached. Unable to get valid JSON response.")
        return None, None
        
    # Translate subtitles check_response wrapper
    def translate_subtitles(self, subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle):
        subtitles_length = count_blocks(subtitles)
        translated_subtitles, used_dollars = self.send_to_openai(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, subtitles_length)

        count = 0
        total_used_dollars = used_dollars
        blocks, additional_content, problematic_blocks = check_response(subtitles, translated_subtitles)

        cumulative_warning = ""
        wasted_dollars = 0
        while (blocks != subtitles_length or additional_content or problematic_blocks) and count < 3:
            self.logger.info("========Retrying========\n")
            self.logger.info(translated_subtitles)
            warning_message = f"Warning: Mismatch in the number of lines ({blocks} != {subtitles_length}), or additional content found ({additional_content}), or problematic blocks ({problematic_blocks}), retry count {count}..."
            self.logger.info(warning_message)
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
        def extract_line(text, num_lines, is_next=False):
            """
            Extracts a specific number of lines from the given text.
    
            The function splits the input text into separate lines, and then
            returns either the first or last num_lines lines, based on the value
            of the is_next flag.
            """
    
            if text is None:
                return None
            entries = text.split('\n\n')
            if is_next:
                selected_entries = '\n\n'.join(entries[:num_lines])
            else:
                selected_entries = '\n\n'.join(entries[-num_lines:])
            return selected_entries
            
        for i, t in enumerate(tqdm(subtitle_batches)):
            prev_subtitle = subtitle_batches[i - 1] if i > 0 else None
            next_subtitle = subtitle_batches[i + 1] if i < len(subtitle_batches) - 1 else None
            
            # choose the last 2 lines of the previous subtitle and the first line of the next subtitle
            prev_subtitle = extract_line(prev_subtitle, 2)
            next_subtitle = extract_line(next_subtitle, 2, is_next=True)

            tt, used_dollars, retry_count, wasted_dollars = self.translate_subtitles(t, prev_subtitle, next_subtitle, prev_translated_subtitle)
            prev_translated_subtitle = tt
            raw_translated.append(tt)
            tt_merged = merge_subtitles_with_timestamps(tt, timestamps_batches[i])
            total_dollars += used_dollars
            number_of_retry += retry_count
            total_wasted_dollars += wasted_dollars
            self.logger.info("========Batch summary=======\n")
            self.logger.info(f"total dollars used: {total_dollars:.3f}\n")
            self.logger.info(f"total number of retry: {number_of_retry}\n")
            self.logger.info(f"total wasted dollars: {total_wasted_dollars:.3f}\n")
            self.logger.info("==============\n")
            self.logger.info(t)
            self.logger.info("==============\n")
            self.logger.info(tt_merged)
            self.logger.info("========End of Batch summary=======\n")
            translated.append(tt_merged)

        translated = ''.join(translated)
        raw_translated = ''.join(raw_translated)

        self.logger.info(translated)
        self.logger.info("========Translate summary=======\n")
        self.logger.info(f"total dollars used: {total_dollars:.3f}\n")
        self.logger.info(f"total number of retry: {number_of_retry}\n")
        self.logger.info(f"total wasted dollars: {total_wasted_dollars:.3f}\n")
        self.logger.info("========End of Translate summary=======\n")
        return translated, raw_translated

def translate_with_gpt(input_file, target_language, batch_size=40, model='gpt-3.5-turbo', video_info=None):
    # Extract the file name without the extension
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    
    subtitle = Subtitle(input_file)
    translator = Translator(model=model, batch_size=batch_size, target_language=target_language, titles=file_name, video_info=video_info, input_path=input_file)

    subtitle_batches, timestamps_batches = subtitle.get_processed_batches_and_timestamps(batch_size)
    translated_subtitles, raw_translated_subtitles = translator.batch_translate(subtitle_batches, timestamps_batches)

    output_file = os.path.join(os.path.dirname(input_file), f"{os.path.splitext(os.path.basename(input_file))[0]}_{target_language}_gpt.srt")
    subtitle.save_subtitles(output_file, translated_subtitles)
    
    
def main():
    parser = argparse.ArgumentParser(description='Translate subtitles using GPT')
    parser.add_argument('-i', '--input_file', help='The path to the input subtitle file.', type=str, required=True)
    # parser.add_argument('-o', '--output_file', help='The path to the output subtitle file.', type=str, required=True)
    parser.add_argument('-b', '--batch_size', help='The number of subtitles to process in a batch.', type=int, default=12)
    parser.add_argument('-l', '--language', help='The target language for translation.', default='zh')
    parser.add_argument('-v', "--video_info", type=str, default="", help="Additional information about the video.")
    parser.add_argument('-m', '--model', default='gpt-3.5-turbo', help='Model for OpenAI API', type=str, choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613'])
    
    args = parser.parse_args()

    translate_with_gpt(args.input_file, args.language, args.batch_size , args.model, args.video_info)



if __name__ == "__main__":
    main()