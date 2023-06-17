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


import nltk
from collections import defaultdict
import tiktoken
from main import SegmentMerger

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0613":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
def count_token(str):
    num_tokens = len(encoding.encode(str))
    return num_tokens
    
    
def encode_timestamps(word_segments):
    # Create a set of all unique timestamps
    timestamps = sorted(set([seg['start'] for seg in word_segments] + [seg['end'] for seg in word_segments]))
    
    # Create a dictionary that maps each timestamp to its index in the list
    timestamp_to_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    
    # Replace the timestamps in the word segments with their corresponding index
    encoded_word_segments = []
    for seg in word_segments:
        encoded_seg = seg.copy()
        encoded_seg['start'] = timestamp_to_index[seg['start']]
        encoded_seg['end'] = timestamp_to_index[seg['end']]
        encoded_word_segments.append(encoded_seg)
    
    return encoded_word_segments, timestamps

# def encode_timestamps(word_segments):
#     # Create a set of all unique timestamps
#     timestamps = sorted(set([seg['start'] for seg in word_segments] + [seg['end'] for seg in word_segments]))
    
#     # Create a dictionary that maps each timestamp to its index in the list
#     timestamp_to_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    
#     # Replace the timestamps in the word segments with their corresponding index
#     for seg in word_segments:
#         seg['start'] = timestamp_to_index[seg['start']]
#         seg['end'] = timestamp_to_index[seg['end']]
    
#     return word_segments, timestamps

def decode_timestamps(word_segments, timestamps):
    # Replace the indices in the word segments with their corresponding timestamp
    for seg in word_segments:
        seg['start'] = timestamps[seg['start']]
        seg['end'] = timestamps[seg['end']]
    
    return word_segments


def get_batches(word_segments, max_tokens=7000):

        
    # Convert word segments to sentences
    # words = [segment['word'] for segment in word_segments]
    segment_merger = SegmentMerger(max_text_len=150)
    
    sentences = segment_merger.process_segments(word_segments)
    sentences = [sentence['words'] for sentence in sentences]
    
    # for sentence in sentences:
    #     print("=================")
    #     sentence, timestamps_encode = encode_timestamps(sentence)
    #     # print(sentence)
    #     print(str(sentence).replace('\'', ''))
    
    # Batch the sentences
    batches = []
    batch = []
    batch_tokens = 0
    for sentence in sentences:
        encoded_word_segments, _ = encode_timestamps(sentence)
        tokens = count_token(str(encoded_word_segments).replace('\'', ''))
        if batch_tokens + tokens > max_tokens:
            # print("Batch tokens: ", batch_tokens)
            # print("Batch: ", batch)
            # print("================")
            # Start a new batch
            batches.append(batch)
            batch = sentence
            batch_tokens = tokens
            
        else:
            # Add to the current batch
            batch.extend(sentence)
            batch_tokens += tokens
    # Add the last batch
    if batch:
        batches.append(batch)
    
    
    return batches

def send_to_openai(word_segments):
    model = 'gpt-3.5-turbo-16k-0613'
    encoded_word_segments, timestamps_encode = encode_timestamps(word_segments)
    encoded_word_segments = str(encoded_word_segments).replace('\'', '')
    tokens = count_token(encoded_word_segments)

    # if video_info:
    #     prompt += f"Additional video information: ---{video_info}---\n\n"


    system_content = '''Guidelines:
- Accurate content, max 42 chars/line
- Use ellipses, hyphens, dates, numbers as per guidelines
- Correct line treatment, simple punctuation, double quotes for quoted words
- Reading speed limit: 20 chars/sec, max 5 secs/line
- Break down long sentences into shorter, more readable ones, even if there's no punctuation to indicate a natural break
- Segment the sentence into different text lines at punctuation marks when improving the subtitle segmentation, not just adding punctuation within a line
- Strictly limit the difference between the start and end of a text to less than 15. This is a crucial rule.

Task:
1. Merge word segments into sentences. Do not invent sentences that do not exist in the input. Ensure that the start and end times of the sentences match the start of the first word and end of the last word in the sentence. Remember to strictly adhere to the 15-second rule between the start and end of a text. Do not change the content of the subtitles. Double-check that the start and end times are correct and that the text does not contain any content not present in the word segment list.
2. Translate the sentences into Chinese. When translating the subtitles, make the language more natural, consider cultural nuances, and avoid common translation errors. Ensure that the start and end times of the translated sentences match those of the original sentences.'''
    
    example_user = '''Word Segments:
```[{'word': ' Welcome', 'start': 1, 'end': 2}, {'word': ' back', 'start': 2, 'end': 3}, {'word': ' to', 'start': 3, 'end': 4}, {'word': ' my', 'start': 4, 'end': 5}, {'word': ' channel.', 'start': 5, 'end': 6}, {'word': ' And', 'start': 7, 'end': 8}, {'word': ' if', 'start': 8, 'end': 9}, {'word': " you're", 'start': 9, 'end': 10}, {'word': ' new,', 'start': 10, 'end': 11}, {'word': ' welcome.', 'start': 12, 'end': 13}, {'word': ' I', 'start': 14, 'end': 15}, {'word': ' hope', 'start': 15, 'end': 16}, {'word': ' to', 'start': 16, 'end': 17}, {'word': ' earn', 'start': 17, 'end': 18}, {'word': ' your', 'start': 18, 'end': 19}, {'word': ' subscription', 'start': 19, 'end': 20}, {'word': ' today.', 'start': 20, 'end': 21}, {'word': " I'm", 'start': 21, 'end': 22}, {'word': ' out', 'start': 22, 'end': 23}, {'word': ' here', 'start': 23, 'end': 24}, {'word': ' today', 'start': 24, 'end': 25}, {'word': ' prospecting,', 'start': 25, 'end': 26}, {'word': ' looking', 'start': 26, 'end': 27}, {'word': ' for', 'start': 27, 'end': 28}, {'word': ' a', 'start': 28, 'end': 29}, {'word': ' new', 'start': 29, 'end': 29}, {'word': ' deposit', 'start': 29, 'end': 30}, {'word': ' of', 'start': 30, 'end': 31}, {'word': ' Gemmy', 'start': 31, 'end': 32}, {'word': ' Gemmy', 'start': 32, 'end': 33}, {'word': ' garnets.', 'start': 33, 'end': 34}, {'word': ' So', 'start': 35, 'end': 36}, {'word': ' wish', 'start': 36, 'end': 37}, {'word': ' me', 'start': 37, 'end': 38}, {'word': ' luck', 'start': 38, 'end': 39}, {'word': ' and', 'start': 39, 'end': 40}, {'word': ' I', 'start': 40, 'end': 41}, {'word': ' hope', 'start': 41, 'end': 42}, {'word': ' you', 'start': 42, 'end': 43}, {'word': ' enjoy.', 'start': 43, 'end': 44}]```
'''
    
    example_assistant = '''{
    "Subtitles": [
        {"text": "Welcome back to my channel.", "start": 1, "end": 6},
        {"text": "And if you're new, welcome.", "start": 7, "end": 13},
        {"text": "I hope to earn your subscription today.", "start": 14, "end": 21},
        {"text": "I'm out here today prospecting, looking for a new deposit of Gemmy Gemmy garnets.", "start": 21, "end": 34},
        {"text": "So wish me luck and I hope you enjoy.", "start": 35, "end": 44}
    ],
    "Translation": [
        {"text": "欢迎回到我的频道。", "start": 1, "end": 6},
        {"text": "如果你是新来的，欢迎。", "start": 7, "end": 13},
        {"text": "我希望今天能赢得你的订阅。", "start": 14, "end": 21},
        {"text": "我今天在这里勘探，寻找新的Gemmy Gemmy石榴石矿床。", "start": 21, "end": 34},
        {"text": "所以祝我好运，我希望你能喜欢。", "start": 35, "end": 44},
    ]
}'''
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": example_assistant},
        {"role": "user", "content": f'Word Segments:```{encoded_word_segments}```'},
    ]
    
    print("========Messages========\n")
    print(messages)
    print("========End of Messages========\n")
    
    inference_not_done = True
    answer = ''
    translated_subtitles = ''
    while inference_not_done:
        try:
            delay_time = 0.01
            start_time = time.time()
            print(f"Sending to OpenAI API...")
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
                # top_p=0.33,
                max_tokens=5500,
                stream=True,
            )
            for event in response: 
                # STREAM THE ANSWER
                print(answer, end='', flush=True) # Print the response    
                # RETRIEVE THE TEXT FROM THE RESPONSE
                event_time = time.time() - start_time  # CALCULATE TIME DELAY BY THE EVENT
                event_text = event['choices'][0]['delta'] # EVENT DELTA RESPONSE
                answer = event_text.get('content', '') # RETRIEVE CONTENT
                translated_subtitles += answer
                time.sleep(delay_time)
    
            inference_not_done = False
        except Exception as e:
            print(f"Waiting 60 seconds")
            print(f"Error was: {e}")
            time.sleep(60)

    # translated_subtitles = response.choices[0].get("message").get("content").encode("utf8").decode()
    print("========Response========\n")
    # print(response)
    print(translated_subtitles)
    
    try:
        data = json.loads(translated_subtitles)
        # print(data["Subtitles"])
        # print(data["Improved Subtitles"])
        translation = decode_timestamps(data["Translation"], timestamps_encode)
        print("========Translation========\n")
        print(translation)
        
    except json.JSONDecodeError:
        print("The JSON data is not well formatted.")

    if model == 'gpt-3.5-turbo-16k-0613':
        # prompt_tokens = response['usage']['prompt_tokens']
        # completion_tokens = response['usage']['completion_tokens']
        prompt_tokens = count_token(str(messages))
        completion_tokens = count_token(translated_subtitles)
        used_dollars = (prompt_tokens / 1000 * 0.003) + (completion_tokens / 1000 * 0.004)
        print(f"prompt tokens: {prompt_tokens}, completion tokens: {completion_tokens}, Used dollars: {used_dollars}")
    
    # elif model == "gpt-4":
    #     prompt_tokens = response['usage']['prompt_tokens']
    #     completion_tokens = response['usage']['completion_tokens']
    #     used_dollars = (prompt_tokens / 1000 * 0.03) + (completion_tokens / 1000 * 0.06)
    #     print(f"prompt tokens: {prompt_tokens}, completion tokens: {completion_tokens}, Used dollars: {used_dollars}")

    return translation, used_dollars
        
        
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
        
def main():
    parser = argparse.ArgumentParser(description='Translate word level segment using GPT')
    parser.add_argument('-i', '--input_file', help='The path to the input word level segment file.', type=str, required=True)
    # parser.add_argument('-o', '--output_file', help='The path to the output subtitle file.', type=str, required=True)
    parser.add_argument('-l', '--language', help='The target language for translation.', default='zh')
    parser.add_argument('-v', "--video_info", type=str, default="", help="Additional information about the video.")
    parser.add_argument('-m', '--model', default='gpt-3.5-turbo', help='Model for OpenAI API', type=str, choices=['gpt-3.5-turbo', 'gpt-4'])
    
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        word_segments = json.load(f)
    
    # segment_merger = SegmentMerger(max_text_len=150)
    
    # sentences = segment_merger.process_segments(word_segments)
    # sentences = [sentence['words'] for sentence in sentences]
    
    # for sentence in sentences:
    #     print("=================")
    #     sentence, timestamps_encode = encode_timestamps(sentence)
    #     # print(sentence)
    #     print(str(sentence).replace('\'', ''))
        
    batches = get_batches(word_segments)
    translation = []
    for batch in batches:
        print("=================")
        # encoded_word_segments, _ = encode_timestamps(batch)
        # encoded_word_segments = str(encoded_word_segments).replace('\'', '')
        # tokens = count_token(encoded_word_segments)
        # print("Tokens: ", tokens)
        # print("Batch: ", encoded_word_segments)
        batch_translation = send_to_openai(batch)
        translation.extend(batch_translation)

    srt_text = segments_to_srt(translation)
    with open('subtitles.srt', 'w', encoding='utf-8') as f:
        f.write(srt_text)
        
    # print("Batches len: ", len(batches))
    
    # print(loaded_word_segments)
    # translate_with_gpt(args.input_file, args.language, args.batch_size , args.model, args.video_info)



if __name__ == "__main__":
    main()
    