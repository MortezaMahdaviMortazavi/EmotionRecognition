import pandas as pd
import re
import config
import re
# from hazm import Normalizer, word_tokenize, sent_tokenize, stopwords_list
# import parsivar
import nltk


class Preprocessing(object):
    """
        1-Normalization
        2-remove any english character(optional now)
        3-Removing letters that frequently misspelled in order to emphasize certain for examle خیلللیییی rather than خیلی
        4-Removing from text any arabic diacritics that parsivar didnt remove
        5-after all of above stages,remove non persian characters
        6-Removing # hashtags from text and retain their Information

    """

    def __init__(self) -> None:
        pass

def normalize(text):
    # normalizer = Normalizer()
    # normalized_text = normalizer.normalize(text)
    # return normalized_text
    pass

def remove_english_chars(text):
    english_chars_pattern = re.compile(r'[a-zA-Z]')
    cleaned_text = re.sub(english_chars_pattern, '', text)
    return cleaned_text

def remove_misspelled_letters(text):
    misspelled_letters_mapping = {
        'لل': 'لی',
        # Add more mappings as needed
    }
    for misspelled, corrected in misspelled_letters_mapping.items():
        text = text.replace(misspelled, corrected)
    return text

def remove_arabic_diacritics(text):
    arabic_diacritics_pattern = re.compile(r'[\u064B-\u065F]')
    cleaned_text = re.sub(arabic_diacritics_pattern, '', text)
    return cleaned_text

def remove_non_persian_chars(text):
    persian_chars_pattern = re.compile(r'[^\u0600-\u06FF\uFB8A\u067E\u0686\u06AF\u200C\u200F]+')
    cleaned_text = re.sub(persian_chars_pattern, ' ', text)
    return cleaned_text

def remove_hashtags(text):
    hashtags_pattern = re.compile(r'#\w+')
    cleaned_text = re.sub(hashtags_pattern, '', text)
    return cleaned_text

# # Example usage
# text = "خیلللیییی #example text"
# text = normalize(text)
# # text = remove_english_chars(text)
# text = remove_misspelled_letters(text)
# text = remove_arabic_diacritics(text)
# text = remove_non_persian_chars(text)
# text = remove_hashtags(text)

# print(text)