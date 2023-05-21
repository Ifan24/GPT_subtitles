from abc import ABC, abstractmethod
import whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from googletrans import Translator
import time
from translate_gpt import translate_with_gpt
from tqdm import tqdm

def batch_text(result, gs=32):
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
        
class ITranslationService(ABC):
    @abstractmethod
    def translate(self, text, src_lang, tr_lang):
        pass

class GoogleTranslateService(ITranslationService):
    def translate(self, result, src_lang='en', tr_lang='zh-cn'):
        if tr_lang == 'zh':
            tr_lang = 'zh-cn'
        translator = Translator()
        batch_texts = batch_text(result, gs=25)
        translated = []
        
        for texts in tqdm(batch_texts):
            batch_translated = []
            for text in texts:
                inference_not_done = True
                while inference_not_done:
                    try:
                        translation = translator.translate(text, src=src_lang, dest=tr_lang)
                        inference_not_done = False
                    except Exception as e:
                        print(f"Waiting 15 seconds")
                        print(f"Error was: {e}")
                        time.sleep(15)
    
                batch_translated.append(translation.text)
            translated += batch_translated
        return translated

class M2M100TranslateService(ITranslationService):
    def translate(self, result, src_lang='en', tr_lang='zh'):
        model_name = "facebook/m2m100_418M"
        model = M2M100ForConditionalGeneration.from_pretrained(model_name).to('cuda')
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        tokenizer.src_lang = src_lang
        translated = []
        batch_texts = batch_text(result, gs=32)
        for texts in tqdm(batch_texts):
            batch_translated = []
            for text in texts:
                encoded = tokenizer(text, return_tensors="pt", padding=True).to('cuda')
                generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tr_lang))
                translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                batch_translated += translated_text
            translated += batch_translated
        return translated

