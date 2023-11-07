import openai
import ujson
import os
from tqdm import tqdm
import re
import argparse
import time
from dotenv import load_dotenv
import logging

def check_for_errors(log_file_path, starting_line):
    # First, check if the log file exists
    if not os.path.exists(log_file_path):
        print(f"No log file found at {log_file_path}")
        return False
    
    error_occurred = False
    with open(log_file_path, 'r') as log_file:
        # Skip to the starting line
        for _ in range(starting_line):
            next(log_file, None)  # Skip the line safely
        
        # Check for errors in the new lines
        for line in log_file:
            if "- ERROR -" in line:  # Adjusted to match the specific error format
                error_occurred = True
                break
    return error_occurred
    
def count_log_lines(log_file_path):
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as file:
            return sum(1 for line in file)
    else:
        return 0  # If the file doesn't exist, return 0 lines
        
        
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



class TranslationMapping:
    def __init__(self, max_size):
        self.max_size = max_size
        self.mapping_dict = {}
        self.translations = set()
        self.all_mappings = []
        self.current_index = 0  # Keep track of the most recent subtitle index

    def add_mapping(self, new_mapping, translations):
        for subtitle in translations:
            # Ensure 'index' is present and is an integer
            if not isinstance(subtitle["index"], int):
                # Handle the error or convert to int as appropriate
                continue  # Skip this subtitle if the index is not valid
                
            index = subtitle["index"]
            self.current_index = max(self.current_index, index)  # Update the most recent subtitle index
            translation = subtitle["translation"]
            original_text = subtitle["original_text"]

            words = re.findall(r'\b\w+\b', original_text)

            for word in words:
                if word in self.mapping_dict:
                    self.mapping_dict[word]['frequency'] += 1
                    self.mapping_dict[word]['index'] = index
                    self.calculate_score(word)

        for term, translation in new_mapping.items():
            # Preprocess term
            proper_noun = term.lower().strip()

            if proper_noun not in self.mapping_dict and translation not in self.translations:
                if len(self.mapping_dict) == self.max_size:
                    # Remove the mapping with the lowest score
                    proper_noun_to_remove = min(self.mapping_dict, key=lambda x: self.mapping_dict[x]['score'])
                    removed_translation = self.mapping_dict[proper_noun_to_remove]['translation']
                    del self.mapping_dict[proper_noun_to_remove]
                    self.translations.remove(removed_translation)

                self.mapping_dict[proper_noun] = {'translation': translation, 'frequency': 1, 'index': self.current_index, 'score': 0}
                self.translations.add(translation)

            self.all_mappings.append((proper_noun, translation))

        # sort the mapping by key so it is easier to check for human
        self.mapping_dict = dict(sorted(self.mapping_dict.items(), key=lambda item: item[0]))
        self.all_mappings = sorted(self.all_mappings, key=lambda item: item[0])

    def calculate_score(self, proper_noun):
        # Here, we give equal weight to frequency and recency. 
        # Adjust the weights depending on which factor you want to prioritize.
        frequency_score = self.mapping_dict[proper_noun]['frequency']
        index_difference = self.current_index - self.mapping_dict[proper_noun]['index'] + 1
        recency_score = 1 / index_difference
        self.mapping_dict[proper_noun]['score'] = frequency_score * recency_score

    def get_mappings(self):
        return {proper_noun: mapping['translation'] for proper_noun, mapping in self.mapping_dict.items()}

    def get_all_mappings(self):
        unique_mappings = list(set(self.all_mappings))  # Removes duplicates
        sorted_mappings = sorted(unique_mappings, key=lambda item: item[0])
        return "\n".join(f"{proper_noun} : {translation}" for proper_noun, translation in sorted_mappings)

    def get_current_mappings(self):
        sorted_mappings = sorted(self.mapping_dict.items(), key=lambda item: item[0])
        return "\n".join(f"{proper_noun} : {mapping['translation']}" for proper_noun, mapping in sorted_mappings)


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
    def __init__(self, model='gpt-3.5-turbo-16k', batch_size=40, target_language='zh', source_language='en', titles='Video Title not found', video_info=None, input_path=None, no_translation_mapping=False, load_from_tmp=False):
        self.model = model
        self.batch_size = batch_size
        self.target_language = target_language
        self.source_language = source_language
        self.titles = titles
        self.video_info = video_info
        self.input_path = input_path
        
        # LRFU (Least Recently/Frequently Used) 
        self.translation_mapping = TranslationMapping(max_size=40)
        
        self.no_translation_mapping = no_translation_mapping
        
        self.load_from_tmp = load_from_tmp
        
        self.translate_max_retry = 2
        
        
        with open('few_shot_examples.json', 'r') as f:
            few_shot_examples = ujson.load(f)
        
        try:
            self.few_shot_examples = few_shot_examples[f"{self.source_language}-to-{self.target_language}"]
            
        except KeyError:
            print("No few shot examples found for this language pair. Please add some examples to few_shot_examples.json. Use default examples (en-to-zh)")
            self.few_shot_examples = few_shot_examples["en-to-zh"]            
            
        
        # TODO: use a mapping to transform language code
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
        
    def process_line(self, line):
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
            
    def send_to_openai(self, subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, subtitles_length, warning_message=None, prev_response=None):
        total_used_dollars = 0

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

- Please translate the following subtitles and summarize all the proper nouns that appear to generate a mapping.
- Please only output the proper nouns and their translation that appear in the current batch of subtitles, do not repeat from the input
- You may receive translation_mapping as input, which is a mapping of proper nouns to their translation in {self.target_language}. 
- Please follow this mapping to translate the subtitles to improve translation consistency.

- Target language: {self.target_language}

- Please output proper JSON with this format:
{{
    "current_batch_subtitles_translation": [
        {{
            "index": <int>,
            "original_text": <str>,
            "translation": <str>
        }}
    ],
    "translation_mapping": {{
        "proper nouns": <translation in target language>
    }}
}}"""

        user_input = self.process_user_input(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, warning_message)
        
        messages = [
            {"role": "system", "content": system_content},
        ]
        
        # append few-shot examples
        for example in self.few_shot_examples["examples"]:
            messages.append({"role": "user", "content": ujson.dumps(example["input"], ensure_ascii=False, indent=2)})
            messages.append({"role": "assistant", "content": ujson.dumps(example["output"], ensure_ascii=False, indent=2)})
        
        messages.append({"role": "user", "content": ujson.dumps(user_input, ensure_ascii=False, indent=2)})
        
        self.logger.info("========Messages========\n")
        self.logger.info(messages)
        self.logger.info("========End of Messages========\n")
        
        max_retries = self.translate_max_retry
        retry_count = 0
        translated_subtitles = ''
        finish_reasons = set()
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
                    # not very useful, the output format is JSON but does not follow the prompt
                    # response_format={"type": "json_object" } if '1106' in self.model else {"type": "text"},
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
                    finish_reason = event['choices'][0]['finish_reason']
                    finish_reasons.add(str(finish_reason))
                    answer = event_text.get('content', '')
                    translated_subtitles += answer
                    time.sleep(delay_time)
                
                self.openai_logger.handlers[0].terminator = terminator
                self.openai_logger.info("===========================") 
                used_dollars = self.count_used_dollars(translated_subtitles, messages)
                
                total_used_dollars += used_dollars
                
                # Parse JSON string into a Python dictionary
                json_string_with_double_quotes = re.sub(r"(\s*[\{\}:,]\s*)'([^']*)'", r'\1"\2"', translated_subtitles)
                pattern = re.compile(r',\s*}')
                cleaned_json_string = re.sub(pattern, '}', json_string_with_double_quotes)
                
                self.logger.info("========Response========\n")
                self.logger.info(ujson.dumps(translated_subtitles, ensure_ascii=False, indent=4))
          
                self.logger.info(f"finish_reasons: {finish_reasons}")
                
                data = ujson.loads(cleaned_json_string)
        
                # Extract translations and construct the output string
                output_string = ""
                for subtitle in data["current_batch_subtitles_translation"]:
                    
                    # If the 'translation' key is not present, use 'original_text' as a fallback
                    if 'translation' not in subtitle:
                        self.logger.error(f"Missing 'translation' in subtitle: {subtitle}")
                        subtitle['translation'] = subtitle.get('original_text', '')
                        
                    # Ensure 'index' is present and is an integer
                    if 'index' not in subtitle or not isinstance(subtitle['index'], int):
                        self.logger.error(f"Missing or invalid 'index' in subtitle: {subtitle}")
                    
                    index = subtitle["index"]
                    translation = subtitle["translation"]
                    output_string += f"{index}\n{translation}\n\n"
                
                translation_mapping = data["translation_mapping"]
                self.translation_mapping.add_mapping(translation_mapping, data["current_batch_subtitles_translation"])
                
                # self.logger.info(self.translation_mapping.get_all_mappings())
                self.logger.info(ujson.dumps(self.translation_mapping.mapping_dict, ensure_ascii=False, indent=4))
                
                return output_string, total_used_dollars

            except ujson.JSONDecodeError as e:
                retry_count += 1
                self.logger.error(f"An error occurred while parsing JSON: {e}. Retrying {retry_count} of {max_retries}.")
                warning_message = f"Your response is not in a valid JSON format. Please double-check your answer. Error:{e}"
                if warning_message:
                    user_input["Warning_message"] = f"In a previous request sent to OpenAI, the response is problematic. Please double-check your answer. Warning message: {warning_message} Retry count: {retry_count} of {max_retries}"
                
                self.logger.info("========Messages========\n")
                self.logger.info(ujson.dumps(user_input, ensure_ascii=False, indent=4))
                self.logger.info("========End of Messages========\n")
                # replace the last message with the warning message
                messages[-1]["content"] = ujson.dumps(user_input)
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
                return translated_subtitles, total_used_dollars 

        self.logger.error("Max retries reached. Unable to get valid JSON response.")
        return translated_subtitles, total_used_dollars

    def process_user_input(self, subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, warning_message):
        user_input = {}
        
        if prev_subtitle:
            previous_subtitles = self.process_line(prev_subtitle)
            if prev_translated_subtitle:
                translated_subtitles = self.process_line(prev_translated_subtitle)
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
            user_input["current_batch_subtitles"] = self.process_line(subtitles)
        
        if next_subtitle:
            user_input["next_batch_subtitles"] = self.process_line(next_subtitle)
            
        if warning_message:
            user_input["Warning_message"] = f"In a previous request sent to OpenAI, the response is problematic. Please double-check your answer. Warning message: {warning_message}"
        
        if len(self.translation_mapping.get_mappings()) != 0 and not self.no_translation_mapping:
            user_input["translation_mapping"] = self.translation_mapping.get_mappings()
            
        return user_input

    def count_used_dollars(self, translated_subtitles, messages):
        prompt_tokens = count_token(str(messages))
        completion_tokens = count_token(translated_subtitles)
        used_dollars = 0
        if 'gpt-3.5-turbo' in self.model:
            used_dollars = (prompt_tokens / 1000 * 0.001) + (completion_tokens / 1000 * 0.002)
            self.logger.info(f"prompt tokens: {prompt_tokens}, completion tokens: {completion_tokens}, Used dollars: {used_dollars}")
                    
        elif 'gpt-4' in self.model:
            # prompt_tokens = response['usage']['prompt_tokens']
            # completion_tokens = response['usage']['completion_tokens']
            used_dollars = (prompt_tokens / 1000 * 0.01) + (completion_tokens / 1000 * 0.03)
            self.logger.info(f"prompt tokens: {prompt_tokens}, completion tokens: {completion_tokens}, Used dollars: {used_dollars}")
        return used_dollars
   
    # Translate subtitles check_response wrapper
    def translate_subtitles(self, subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle):
        subtitles_length = count_blocks(subtitles)
        translated_subtitles, used_dollars = self.send_to_openai(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, subtitles_length)

        count = 0
        total_used_dollars = used_dollars
        blocks, additional_content, problematic_blocks = check_response(subtitles, translated_subtitles)

        cumulative_warning = ""
        wasted_dollars = 0
        while (blocks != subtitles_length or additional_content or problematic_blocks) and count < self.translate_max_retry:
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
            
        # if tmp_file exists, load the previous translated subtitles
        tmp_file = os.path.join(os.path.dirname(self.input_path), 'tmp_subtitles.json')
        skip_length = 0
        if os.path.exists(tmp_file):
            with open(tmp_file, 'r') as f:
                previous_subtitles = ujson.load(f)
        
            # skip the first n batch if load from tmp file
            if self.load_from_tmp:
                translated = previous_subtitles
                skip_length = len(translated)
        
        for i, t in enumerate(tqdm(subtitle_batches)):
            if skip_length > 0:
                skip_length -= 1
                continue
                
            prev_subtitle = subtitle_batches[i - 1] if i > 0 else None
            next_subtitle = subtitle_batches[i + 1] if i < len(subtitle_batches) - 1 else None
            
            # choose the last 2 lines of the previous subtitle and the first line of the next subtitle
            prev_subtitle = extract_line(prev_subtitle, 2)
            next_subtitle = extract_line(next_subtitle, 2, is_next=True)

            tt, used_dollars, retry_count, wasted_dollars = self.translate_subtitles(t, prev_subtitle, next_subtitle, prev_translated_subtitle)
            prev_translated_subtitle = tt
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
            
            with open(tmp_file, 'w') as f:
                ujson.dump(translated, f, ensure_ascii=False, indent=2) 

        translated = ''.join(translated)

        self.logger.info(translated)
        self.logger.info("========Translate summary=======\n")
        self.logger.info(f"total dollars used: {total_dollars:.3f}\n")
        self.logger.info(f"total number of retry: {number_of_retry}\n")
        self.logger.info(f"total wasted dollars: {total_wasted_dollars:.3f}\n")
        self.logger.info("========End of Translate summary=======\n")
        self.logger.info("========translation mapping=======\n")
        self.logger.info(self.translation_mapping.get_all_mappings())
        
        
        return translated

def translate_with_gpt(input_file, target_language='zh', source_language='en', batch_size=40, model='gpt-3.5-turbo-16k', video_info=None, no_translation_mapping=False, load_from_tmp=False):
    # check log file
    log_file_path = os.path.join(os.path.dirname(input_file), 'translator.log')
    starting_line = count_log_lines(log_file_path)
    
    # Extract the file name without the extension
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    
    subtitle = Subtitle(input_file)
    translator = Translator(model=model, batch_size=batch_size, target_language=target_language, source_language=source_language, 
        titles=file_name, video_info=video_info, input_path=input_file, no_translation_mapping=no_translation_mapping, load_from_tmp=load_from_tmp)

    subtitle_batches, timestamps_batches = subtitle.get_processed_batches_and_timestamps(batch_size)
    translated_subtitles = translator.batch_translate(subtitle_batches, timestamps_batches)

    output_file = os.path.join(os.path.dirname(input_file), f"{os.path.splitext(os.path.basename(input_file))[0]}_{target_language}_gpt.srt")
    subtitle.save_subtitles(output_file, translated_subtitles)
    
    # check if an error was logged
    if check_for_errors(log_file_path, starting_line):
        print("An error was logged. Please search '- ERROR -' in translator.log for more details.")
    
def main():
    parser = argparse.ArgumentParser(description='Translate subtitles using GPT')
    parser.add_argument('-i', '--input_file', help='The path to the input subtitle file.', type=str, required=True)
    # parser.add_argument('-o', '--output_file', help='The path to the output subtitle file.', type=str, required=True)
    parser.add_argument('-b', '--batch_size', help='The number of subtitles to process in a batch.', type=int, default=12)
    parser.add_argument('-l', '--target_language', help='The target language for translation.', default='zh')
    parser.add_argument('-s', '--source_language', help='The source language for translation.', default='en')
    parser.add_argument('-v', "--video_info", type=str, default="", help="Additional information about the video.")
    parser.add_argument('-m', '--model', default='gpt-3.5-turbo-16k', help='Model for OpenAI API, default to gpt-3.5-turbo-16k', type=str)
    parser.add_argument('-um', "--no_mapping", action='store_true', help="don't use translation mapping as input to the model" )
    parser.add_argument('-lt', "--load_tmp_file", action='store_true', help="load the previous translated subtitles, assume previous tmp file generated with the same setting as the current run")
    
    args = parser.parse_args()

    translate_with_gpt(args.input_file, args.target_language, args.source_language, 
                args.batch_size , args.model, args.video_info, args.no_mapping, args.load_tmp_file)



if __name__ == "__main__":
    main()