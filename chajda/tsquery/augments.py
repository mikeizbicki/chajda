'''
'''

from chajda.tsvector import lemmatize, Config
from chajda.tsquery.__init__ import to_tsquery

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

    # load the model if it's not already loaded
    try:
        augments_gensim.model
    except AttributeError:
        import gensim.downloader
        with suppress_stdout_stderr():
            augments_gensim.model = gensim.downloader.load("glove-wiki-gigaword-50")

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


import fasttext
import fasttext.util

fasttext_models = {}

def augments_fasttext(lang, word, config=Config(), n=5):
    ''' 
    Returns n words that are "similar" to the input word in the target language.
    These words can be used to augment a search with the Query class.

    >>> to_tsquery('en', 'baby boy', augment_with=augments_fasttext)
    '(baby:A | newborn:B | infant:B) & (boy:A | girl:B | boyhe:B | boyit:B)'

    >>> to_tsquery('en', '"baby boy"', augment_with=augments_fasttext)
    'baby:A <1> boy:A'

    >>> to_tsquery('en', '"baby boy" (school | home) !weapon', augment_with=augments_fasttext)
    '(baby:A <1> boy:A) & ((school:A | schoo:B | schoolthe:B | schoool:B | kindergarten:B) | (home:A | house:B | homethe:B | homewhen:B | homethis:B)) & !(weapon:A | weaponthe:B | weopon:B)'

    >>> augments_fasttext('en','weapon', n=5)
    ['weaponthe', 'weopon']

    >>> augments_fasttext('en','king', n=5)
    ['queen', 'kingthe']

    NOTE:
    Due to the size of fasttext models (>1gb),
    testing multiple languages in the doctest requires more space than github actions allows for.
    For this reason, tests involving languages other than English have been commented out below.

    #>>> augments_fasttext('ja','さようなら', n=5)
    #['さよなら', 'バイバイ', 'サヨウナラ', 'さらば', 'おしまい']

    #>>> augments_fasttext('es','escuela', n=5)
    #['escuelala', 'academia', 'universidad', 'laescuela']
    '''

    # download and load the fasttext model if it's not already loaded
    try:
        fasttext_models[lang]
    except KeyError:
        with suppress_stdout_stderr():
            fasttext.util.download_model(lang, if_exists='ignore')
        fasttext_models[lang] = fasttext.load_model('cc.{0}.300.bin'.format(lang))

    # find the most similar words
    topn = fasttext_models[lang].get_nearest_neighbors(word, k=n)
    words = ' '.join([ word for (rank, word) in topn ])

    # lemmatize the results so that they'll be in the search document's vocabulary
    words = lemmatize(lang, words, add_positions=False, config=config).split()
    words = list(filter(lambda w: len(w)>1 and w != word, words))[:n]

    return words

# The following imports as well as suppress_stdout_stderr() are taken from stackoverflow:
# https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
# In this project, suppress_stdout_stderr() is used for redirecting stdout and stderr when downloading gensim and fasttext models.
# Without this, the doctests fail due to stdout and stderr from gensim and fasttext model downloads.
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """Context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
