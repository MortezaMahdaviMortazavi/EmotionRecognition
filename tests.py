from feature_extraction import FeatureExtraction
from preprocessing import Preprocessing
from vocabulary import Vocabulary
from augmentation import augment
from models import *

import config
import utils



def test_feature_extraction():
    testfile = 'asset/feature_extraction_test.txt'
    # Create an instance of FeatureExtraction
    feature_extraction = FeatureExtraction()

    # Test with an English sample
    english_text = "I love ❤️ OpenAI"
    # print("English Sample:")
    print("Text:", english_text)

    # Test get_emojies method
    emojies = feature_extraction.get_emojies(english_text)
    print("Emojies:", emojies)
    utils.write_text_to_file('emojies: '+str(emojies)+'\n', testfile)

    # Test get_pos method
    pos = feature_extraction.get_pos(english_text)
    print("Parts of Speech:", pos)
    utils.write_text_to_file('part of speech(pos): '+str(pos)+'\n', testfile)

    # Test get_hashtags method
    hashtags = feature_extraction.get_hashtags(english_text)
    # print("Hashtags:", hashtags)
    utils.write_text_to_file('hashtags: '+str(hashtags)+'\n', testfile)

    # Test get_misspelled_words method
    misspelled_words = feature_extraction.get_misspelled_words(english_text)
    # print("Misspelled Words:", misspelled_words)
    utils.write_text_to_file('misspelled words: '+str(misspelled_words)+'\n', testfile)

    # Test __call__ method
    feature_extraction(english_text)
    features = feature_extraction.features
    print("Features:")
    for key, value in features.items():
        print(key + ":", value)

    utils.write_text_to_file('features: '+str(features)+'\n', testfile)

    # Test with a Persian sample
    persian_text = "من دوست دارم ❤️ OpenAI"
    print("Text:", persian_text)

    # Test get_emojies method
    persian_emojies = feature_extraction.get_emojies(persian_text)
    # print("Emojies:", persian_emojies)
    utils.write_text_to_file('emojies: '+str(persian_emojies)+'\n', testfile)

    # Test get_pos method
    persian_pos = feature_extraction.get_pos(persian_text)
    utils.write_text_to_file('Part of speech(pos): '+str(persian_pos)+'\n', testfile)

    # Test get_hashtags method
    persian_hashtags = feature_extraction.get_hashtags(persian_text)
    # print("Hashtags:", persian_hashtags)
    utils.write_text_to_file('hashtags: '+str(persian_hashtags)+'\n', testfile)

    # Test get_misspelled_words method
    persian_misspelled_words = feature_extraction.get_misspelled_words(persian_text)
    # print("Misspelled Words:", persian_misspelled_words)
    utils.write_text_to_file('misspelled words: '+str(persian_misspelled_words)+'\n', testfile)

    # Test __call__ method
    feature_extraction(persian_text)
    persian_features = feature_extraction.features
    print("Features:")
    for key, value in persian_features.items():
        print(key + ":", value)
    utils.write_text_to_file('persian_features'+str(persian_features)+'\n', testfile)



def test_preprocessing():
    testfile = 'asset/processing_test.txt'
    # Create an instance of Preprocessing
    preprocessing = Preprocessing(dataset='arman')

    # Test normalize method
    text = "سلام! من یک تست متن پیچیده هستم. می‌خواهم #هشتگ‌ها را از متنم حذف کنم و بعضی از کلمات را درست کنم. برای مثال خیلللییی می‌خواهم به خیلی تبدیل شود. همچنین علامت‌های تشدید عربی را هم باید حذف کنم و کاراکترهای غیرفارسی را نیز حذف کنم. در نهایت می‌خواهم متن نرمال شده را برگردانم."
    utils.write_text_to_file('original text: ' + text+'\n',testfile)
    
    normalized_text = preprocessing.normalize(text)
    utils.write_text_to_file('normalized text: ' + normalized_text+'\n',testfile)


    # Test remove_english_chars method
    english_chars_removed = preprocessing.remove_english_chars(normalized_text)
    utils.write_text_to_file('english_chars_removed: '+english_chars_removed+'\n',testfile)

    # Test remove_misspelled_letters method
    misspelled_removed = preprocessing.remove_misspelled_letters(english_chars_removed)
    utils.write_text_to_file('misspelled_removed: '+misspelled_removed+'\n',testfile)

    # Test remove_arabic_diacritics method
    arabic_diacritics_removed = preprocessing.remove_arabic_diacritics(misspelled_removed)
    utils.write_text_to_file('arabic_diacritics_removed: '+arabic_diacritics_removed+'\n',testfile)

    # Test remove_non_persian_chars method
    non_persian_chars_removed = preprocessing.remove_non_persian_chars(arabic_diacritics_removed)
    utils.write_text_to_file('non_persian_chars_removed: '+non_persian_chars_removed+'\n',testfile)

    # Test remove_hashtags method
    hashtags_removed = preprocessing.remove_hashtags(non_persian_chars_removed)
    utils.write_text_to_file('hashtags_removed: '+hashtags_removed+'\n',testfile)

    # Test label_getter method
    label_text = "من دوست دارم OpenAI"
    label = preprocessing.label_getter(label_text)
    utils.write_text_to_file('label: '+label+'\n',testfile)

    # Test __call__ method
    cleaned_text, target = preprocessing(text)
    utils.write_text_to_file('cleaned_text: '+cleaned_text+'\n',testfile)
    utils.write_text_to_file('label: '+ label+'\n',testfile)

    # Test get_labels method
    labels = preprocessing.get_labels()
    utils.write_text_to_file('labels: '+str(labels)+'\n',testfile)



def test_vocabulary():
    testfile = 'asset/vocabulary_test.txt'
    # Create an instance of Vocabulary
    texts = [
        'خیلی کوچیک هستن و سایزشون بدرد نمیخوره میخوام پس بدم	SAD',
        'از صدای پرنده دم دمای صبح متنفرم متنفرم متنفرم	HATE',
        'کیفیتش خیلی خوبه با شک خریدم ولی واقعا راضیم بعد از حدود 2 ماه استفاده«متأسفانه باخبر شدیم» که فردی در ایرانشهر به حداقل 41 دختر تجاوز کرده. امیدواریم با همکاری نمایندگان محترم مجلس و دستگاه قضا، دیگه به این راحتی باخبر نشیم.	SAD',
        'چون همش با دوربین ثبت شده ، ایا میشه اعتراض زد؟؟ و اصن تاثیر داره؟ کسی اگه اطلاعی داره ممنون میشم راهنمایی کنید	OTHER',
        'اين وضع ب طرز خنده داري گريه داره ...	SAD',
    ]
    vocab = Vocabulary(texts, vocab_threshold=1, name='arman', load=False)

    # Test __call__ method
    word = "کوچیک"
    index = vocab(word)
    utils.write_text_to_file(f"Index of '{word}': {index}\n", testfile)

    # Test __repr__ method
    utils.write_text_to_file(str(vocab) + '\n', testfile)

    # Test get_word_index method
    word = "خنده"
    word_index = vocab.get_word_index(word)
    utils.write_text_to_file(f"Index of '{word}': {word_index}\n", testfile)

    # Test get_index_word method
    index = 2
    word = vocab.get_index_word(index)
    utils.write_text_to_file(f"Word at index {index}: {word}\n", testfile)

    # Test get_vocab_size method
    vocab_size = vocab.get_vocab_size()
    utils.write_text_to_file("Vocabulary Size: " + str(vocab_size) + '\n', testfile)

    # Test get_labels method
    labels = vocab.get_labels()
    utils.write_text_to_file("Labels: " + str(labels) + '\n', testfile)




def main():
    test_feature_extraction()
    test_preprocessing()
    test_vocabulary()


if __name__ == "__main__":
    main()


