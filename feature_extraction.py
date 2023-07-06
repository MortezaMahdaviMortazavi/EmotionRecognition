import emoji
import prettytable
import symspellpy
import jellyfish
import re

import dadmatools.pipeline.language as language


from symspellpy.symspellpy import SymSpell, Verbosity


class BertSPellChecker():
    def __init__(self, max_edit_distance_dictionary, prefix_length, bert_model):
        self.sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
        self.model, self.tokenizer = self.load_bert_model(bert_model)

    def is_number(self, word):
        new_word = re.sub("\d+", "", word)
        if len(new_word) == 0 or len(new_word) / len(word) < 0.8:
            return True
        else:
            return False


class FeatureExtraction(object):

    def __init__(self,tokenizer):

        self.tokenizer = tokenizer
        self.features = {}

    # Emoji
    def get_emojies(self,text):
        emojies = []
        for word in self.tokenizer(text):
            if emoji.emoji_count(word):
                emojies.append(word)
        return emojies

    # Parts of SPeech(pos)
    def get_pos(self,text):
        pass

    # Hashtags
    def get_hashtags(self,text):
        pass

    # Misspelled words
    def get_misspelled_words(self,text):
        pass

    
    def __call__(self,text):
        self.features['emojies'] = self.get_emojies(text)
        self.features['pos'] = None
        self.features['hashtags'] = None
        self.features['misspelled'] = None

def pos_tag(text):
    tagger = POSTagger()
    tagged_text = tagger.tag(hazm.word_tokenize(text))
    return tagged_text

# if __name__ == "__main__":
#     # spell_checker = BertSPellChecker(None,None,None)
#     # spell_checker.is_number('مسئولان')
#     # here lemmatizer and pos tagger will be loaded
#     # as tokenizer is the default tool, it will be loaded as well even without calling
#     from dadmatools.models.normalizer import Normalizer

#     normalizer = Normalizer(
#         full_cleaning=False,
#         unify_chars=True,
#         refine_punc_spacing=True,
#         remove_extra_space=True,
#         remove_puncs=False,
#         remove_html=False,
#         remove_stop_word=False,
#         replace_email_with="<EMAIL>",
#         replace_number_with=None,
#         replace_url_with="",
#         replace_mobile_number_with=None,
#         replace_emoji_with=None,
#         replace_home_number_with=None
#     )

#     text = """
#     <p>
#     دادماتولز اولین نسخش سال ۱۴۰۰ منتشر شده. 
#     امیدواریم که این تولز بتونه کار با متن رو براتون شیرین‌تر و راحت‌تر کنه
#     لطفا با ایمیل dadmatools@dadmatech.ir با ما در ارتباط باشید
#     آدرس گیت‌هاب هم که خب معرف حضور مبارک هست:
#     https://github.com/Dadmatech/DadmaTools
#     </p>
#     """
#     normalized_text = normalizer.normalize(text)
#     #<p> دادماتولز اولین نسخش سال 1400 منتشر شده. امیدواریم که این تولز بتونه کار با متن رو براتون شیرین‌تر و راحت‌تر کنه لطفا با ایمیل <EMAIL> با ما در ارتباط باشید آدرس گیت‌هاب هم که خب معرف حضور مبارک هست: </p>

#     #full cleaning
#     normalizer = Normalizer(full_cleaning=True)
#     normalized_text = normalizer.normalize(text)
#     #دادماتولز نسخش سال منتشر تولز بتونه کار متن براتون شیرین‌تر راحت‌تر کنه ایمیل ارتباط آدرس گیت‌هاب معرف حضور مبارک
#     print(normalized_text)

#     from hazm import POSTagger
#     import hazm


#     # Example usage
#     text = "من دارم به پارک می‌روم."
#     tagged_text = pos_tag(text)
#     print(tagged_text)



from hazm import POSTagger, word_tokenize

def pos_tag(text):
    tagger = POSTagger(model='resources/postagger.model')
    tagged_text = tagger.tag(word_tokenize(text))
    return tagged_text

# Example usage
text = "من دارم به پارک می‌روم."
tagged_text = pos_tag(text)
print(tagged_text)
