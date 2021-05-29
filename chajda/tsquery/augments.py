'''
'''

from chajda.tsvector import lemmatize, Config
from chajda.tsquery import to_tsquery


def augments_gensim(lang, word, config=Config(), n=5):
    '''
    Returns n words that are "similar" to the input word in the target language.
    These words can be used to augment a search with the Query class.

    >>> to_tsquery('en', 'baby boy', augment_with=augments_gensim)
    '(baby:A | boy:B | girl:B | newborn:B | pregnant:B | mom:B) & (boy:A | girl:B | woman:B | man:B | kid:B | mother:B)'

    >>> to_tsquery('en', '"baby boy"', augment_with=augments_gensim)
    'baby:A <1> boy:A'

    >>> to_tsquery('en', '"baby boy" (school | home) !weapon', augment_with=augments_gensim)
    '(baby:A <1> boy:A) & ((school:A | college:B | campus:B | graduate:B | elementary:B | student:B) | (home:A | leave:B | rest:B | come:B)) & !(weapon:A | explosive:B | weaponry:B | gun:B | ammunition:B | device:B)'

    >>> augments_gensim('en','baby', n=5)
    ['boy', 'girl', 'newborn', 'pregnant', 'mom']

    >>> augments_gensim('en','school', n=5)
    ['college', 'campus', 'graduate', 'elementary', 'student']

    >>> augments_gensim('en','weapon', n=5)
    ['explosive', 'weaponry', 'gun', 'ammunition', 'device']
    '''

    # find the most similar words;
    try:
        topn = augments_gensim.model.most_similar(word, topn=n+1)
        words = ' '.join([ word for (word,rank) in topn ])

    # gensim raises a KeyError when the input word is not in the vocabulary;
    # we return an empty list to indicate that there are no similar words
    except KeyError:
        return []

    # lemmatize the results so that they'll be in the search document's vocabulary
    words = lemmatize(lang, words, add_positions=False, config=config).split()
    words = list(filter(lambda w: len(w)>1 and w != word, words))[:n]

    return words


# load the model if it's not already loaded
try:
    augments_gensim.model
except AttributeError:
    import gensim.downloader
    augments_gensim.model = gensim.downloader.load("glove-wiki-gigaword-50")


import fasttext
import fasttext.util

fasttext_models = {}

def augments_fasttext(lang, word, config=Config(), n=5):
    ''' 
    Returns n words that are "similar" to the input word in the target language.
    These words can be used to augment a search with the Query class.

    >>> to_tsquery('en', 'baby boy', augment_with=augments_fasttext)
    '(baby:A | newborn:B | infant:B | babytobe:B | babya:B | babythe:B) & (boy:A | girl:B | boyhe:B | boyit:B | boybut:B | boythis:B)'
    >>> to_tsquery('en', '"baby boy"', augment_with=augments_fasttext)
    'baby:A <1> boy:A'

    >>> to_tsquery('en', '"baby boy" (school | home) !weapon', augment_with=augments_fasttext)
    '(baby:A <1> boy:A) & ((school:A | schoo:B | schoolthe:B | schoool:B | kindergarten:B | shcool:B) | (home:A | house:B | homethe:B | homewhen:B | homethis:B | homee:B)) & !(weapon:A | weaponthe:B | weopon:B | weaponit:B | weaponry:B | wepon:B)'

    >>> augments_fasttext('ja','さようなら', n=5)
    ['さよなら', 'バイバイ', 'サヨウナラ', 'さらば', 'おしまい']

    >>> augments_fasttext('es','escuela', n=5)
    ['escuelala', 'academia', 'universidad', 'laescuela', 'escula']

    >>> augments_fasttext('en','weapon', n=5)
    ['weaponthe', 'weopon', 'weaponit', 'weaponry', 'wepon']
    '''


    #load the model based on lang if it's not already loaded
   # if lang not in fasttext_models:
   #     fasttext.util.download_model(lang, if_exists='ignore')
   #     fasttext_models[lang] = fasttext.load_model('cc.{0}.300.bin'.format(lang))

    try:
        fasttext_models[lang]
    except:
        fasttext.util.download_model(lang, if_exists='ignore')
        fasttext_models[lang] = fasttext.load_model('cc.{0}.300.bin'.format(lang))

   # print('fasttext dimension =', augments_fasttext.model.get_dimension())

   # print('fasttext similar words =', augments_fasttext.model.get_nearest_neighbors(word, k=n+1))

    #find the most similar words
    try:
        topn = fasttext_models[lang].get_nearest_neighbors(word, k=n+5)
        words = ' '.join([ word for (rank, word) in topn ])
       # print('fasttext words bf lemma = ', words)
    except KeyError:
        return []

    # lemmatize the results so that they'll be in the search document's vocabulary
    words = lemmatize(lang, words, add_positions=False, config=config).split()

    #todo: figure out a better way to filter through the typo words that fasttext produces:
    words = list(filter(lambda w: len(w)>1 and w != word, words))[:n]


   # print('returned fasttext words = ', words)
    return words


#print('tsquery fasttext baby boy = ', to_tsquery('en', '"baby boy"', augment_with=augments_fasttext)
#print('tsquery fasttext baby boy school home !weapon = ', to_tsquery('en', '"baby boy" (school | home) !weapon', augment_with=augments_fasttext))

     

     
     






