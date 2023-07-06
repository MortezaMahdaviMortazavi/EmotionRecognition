# import pandas as pd
import re
import config
import re
# from hazm import Normalizer, word_tokenize, sent_tokenize, stopwords_list
# import parsivar
# import nltk
import hazm
import utils

import re

class Preprocessing(object):
    """
        1-Normalization
        2-remove any english character(optional now)
        3-Removing letters that frequently misspelled in order to emphasize certain for example خیلللیییی rather than خیلی
        4-Removing from text any Arabic diacritics that Parsivar didn't remove
        5-after all of the above stages, remove non-Persian characters
        6-Removing # hashtags from text and retain their information
    """

    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Preprocessing, cls).__new__(cls)
        return cls._instance


    def __init__(self,dataset):
        assert dataset in ['arman','emopars','else']

        self.dataset = dataset
        self.labels = {}
        self.label_idx = 0            

    def label_getter(self,text):
        _list = hazm.word_tokenize(text)
        last_item = _list[-1]
        if re.search('[a-zA-Z]', last_item):
            if last_item not in self.labels:
                self.labels[last_item] = self.label_idx
                self.label_idx += 1

            return last_item
        
        return None

    def __call__(self, text):
        normalized_text = self.normalize(text)
        
        if self.dataset == 'arman':
            target = self.label_getter(normalized_text)

        cleaned_text = self.remove_english_chars(normalized_text)
        cleaned_text = self.remove_misspelled_letters(cleaned_text)
        cleaned_text = self.remove_arabic_diacritics(cleaned_text)
        cleaned_text = self.remove_non_persian_chars(cleaned_text)
        cleaned_text = self.remove_hashtags(cleaned_text)
        return cleaned_text, target
    
    def get_labels(self):
        return self.labels

    def normalize(self, text):
        normalizer = hazm.Normalizer()
        normalized_text = normalizer.normalize(text)
        return normalized_text

    def remove_english_chars(self, text):
        english_chars_pattern = re.compile(r'[a-zA-Z]')
        cleaned_text = re.sub(english_chars_pattern, '', text)
        return cleaned_text

    def remove_misspelled_letters(self, text):
        misspelled_letters_mapping = {
            'لل': 'لی',
            # Add more mappings as needed
        }
        for misspelled, corrected in misspelled_letters_mapping.items():
            text = text.replace(misspelled, corrected)
        return text

    def remove_arabic_diacritics(self, text):
        """
            Some common Arabic diacritical marks include:
                Fatha (ً): Represents the short vowel "a" or "u" when placed above a letter.
                Kasra (ٍ): Represents the short vowel "i" when placed below a letter.
                Damma (ٌ): Represents the short vowel "u" when placed above a letter.
                Sukun (ـْ): Indicates the absence of any vowel sound.
                Shadda (ّ): Represents consonant doubling or gemination.
                Tanween (ًٌٍ): Represents the nunation or the "n" sound at the end of a word.
        """

        """
            The regular expression [\u064B-\u065F] represents a character range that covers the Unicode code points for Arabic diacritics.
        """
        # مرحبا بكم <== "مَرْحَبًا بِكُمْ"
        arabic_diacritics_pattern = re.compile(r'[\u064B-\u065F]')
        cleaned_text = re.sub(arabic_diacritics_pattern, '', text)
        return cleaned_text

    def remove_non_persian_chars(self, text):
        persian_chars_pattern = re.compile(r'[^\u0600-\u06FF\uFB8A\u067E\u0686\u06AF\u200C\u200F]+')
        cleaned_text = re.sub(persian_chars_pattern, ' ', text)
        return cleaned_text

    def remove_hashtags(self, text):
        hashtags_pattern = re.compile(r'#\w+')
        cleaned_text = re.sub(hashtags_pattern, '', text)
        return cleaned_text

    


if __name__ == "__main__":
    # Create an instance of Preprocessing
    preprocessor = Preprocessing(dataset='arman')

    # Test text
    text = "This is a test text. ما همچنین فارسی را پشتیبانی می‌کنیم."

    # Preprocess the text
    cleaned_text = preprocessor(text)

    # Print the cleaned text
    print("Cleaned Text:")
    