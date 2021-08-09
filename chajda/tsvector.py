'''
In the postgres database, the `tsvector` type is used to represent a document for full text search.
This file uses the spacy library to convert input text into a `tsvector`.
The main user-facing function is `lemmatize`.
'''

import pkgutil
import importlib
import inspect
import numpy as np
import spacy
from chajda.embeddings import get_test_embedding

################################################################################
# global variables
################################################################################

# initialize logging
import logging
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    )
logger = logging.getLogger(__name__)

# calculate the valid languages for the installed spacy version
valid_langs = []
for _, lang_iso, is_lang in pkgutil.iter_modules(spacy.lang.__path__):
    if is_lang:
        valid_langs.append(lang_iso)
logger.info("valid_langs=" + str(valid_langs))

# the korean list of stop words isn't big enough,
# so we add some more stop words here
# FIXME:
# should these changes get pushed upstream into spacy?
import spacy.lang.ko
for stopword in ['이거', '이것', '는', '은', '가', '이', '을', '를', '기', '에']:
    spacy.lang.ko.stop_words.STOP_WORDS.add(stopword)

# create a table of Unicode special characters for filtering
# this variable is used within the lemmatize function,
# but we precompute it outside the function for efficiency
import unicodedata
import sys
unicode_CPS = dict.fromkeys(i for i in range(0, sys.maxunicode + 1) if unicodedata.category(chr(i)).startswith(('P', 'S', 'C')))

# the nlp dictionary will hold the loaded spacy models;
# it is populated by the load_lang function;
# an entry of None indicates that the model still needs to be loaded
from collections import defaultdict,Counter
nlp = defaultdict(lambda: None)


################################################################################
# function definitions
################################################################################


def load_lang(lang_iso):
    '''
    This function returns a spacy language model.
    Language models take up a lot of RAM, and they take a long time to load.
    Therefore, we do not automatically load all models on program startup.
    Instead, this function is used to load them lazily as needed.

    FIXME:
    I'm pretty sure there should be an easier way to load a language model from the iso code,
    but I couldn't figure out a native way to do this.
    '''

    logger.info('load_lang ' + lang_iso)
    module_name = 'spacy.lang.' + lang_iso
    lang_module = importlib.import_module(module_name)

    # Each module has a class within it responsible for NLP,
    # but it also has many other classes.
    # Our goal is to filter through these other classes to find the NLP class.
    # First, we get the list of all classes.
    module_classes = inspect.getmembers(lang_module, inspect.isclass)

    # Next, we remove all classes that are not actually defined in the module.
    # >>> pprint.pprint(module_classes)
    # [('Chinese', <class 'spacy.lang.zh.Chinese'>),
    #  ('ChineseDefaults', <class 'spacy.lang.zh.ChineseDefaults'>),
    #  ('ChineseTokenizer', <class 'spacy.lang.zh.ChineseTokenizer'>),
    #  ('Doc', <class 'spacy.tokens.doc.Doc'>),
    #  ('DummyTokenizer', <class 'spacy.util.DummyTokenizer'>),
    #  ('Language', <class 'spacy.language.Language'>),
    #  ('OrderedDict', <class 'collections.OrderedDict'>),
    #  ('Path', <class 'pathlib.Path'>)]
    filtered_classes = list(filter(lambda x: module_name in str(x[1]), module_classes))

    # Finally, the class we want is always the class with the shortest name.
    # >>> pprint.pprint(filtered_classes)
    # [('Chinese', <class 'spacy.lang.zh.Chinese'>),
    #  ('ChineseDefaults', <class 'spacy.lang.zh.ChineseDefaults'>),
    #  ('ChineseTokenizer', <class 'spacy.lang.zh.ChineseTokenizer'>)]
    shortest_name_length = min([len(name) for name, obj in filtered_classes])
    nlp_constructor = list(filter(lambda x: len(x[0]) == shortest_name_length, filtered_classes))[0][1]

    # once we have the nlp_constructor,
    # we want to create the model with non-lemmatization related components disabled
    # this will speedup calculations and save memory
    return nlp_constructor(disable=['ner', 'parser'])


def load_all_langs(langs=None):
    '''
    This function immediately loads all languages.
    In typical usage, we want spacy's languages loaded lazily, and so should not call this function.
    When debugging and testing, however, it can be useful to force the immediate loading of all languages.
    '''
    if langs is None:
        langs = valid_langs

    for lang in langs:
        nlp[lang] = load_lang(lang)


class Config:
    '''
    The `Config` class sets the options used for full text search.
    In order for a query in postgres to be accurate,
    the `tsvector` and `tsquery` objects need to be created with the same configuration settings and the same language.
    '''

    def __init__(self, lower_case=True, remove_special_chars=True, remove_stop_words=True, max_lemma_size=20):
        self.lower_case=lower_case
        self.remove_special_chars=remove_special_chars
        self.remove_stop_words=remove_stop_words
        self.max_lemma_size = max_lemma_size


def lemmatize(
        lang,
        text,
        add_positions=True,
        config=Config()
        ):
    '''
    Lemmatizes the input text according to the input language and config options.
    The output of this function is a string, but it can be cast directly into a tsvector within postgres.

    >>> lemmatize('en', 'The United States of America is a country.')
    'unite:2 state:3 america:5 country:8'

    >>> lemmatize('en', 'The United States of America is a country.', add_positions=False)
    'unite state america country'

    >>> lemmatize('xx', 'The United States of America is a country.', add_positions=False)
    'the united states of america is a country'

    >>> lemmatize('xx', 'reallybigword1 verybigword2 reallyreallyreallybigword3', config=Config(max_lemma_size=5))
    'reall:1 veryb:2 reall:3'

    There are extensive tests for all configuration options and languages in the `tests/` folder.
    '''

    # if any input is None (NULL in postgres),
    # then we return None
    if lang is None or text is None:
        return None

    # if the language is not yet loaded, then load it
    # if the language is not supported, then use spacy's multilingual model ('xx')
    if nlp[lang] is None:
        if lang in valid_langs:
            nlp[lang] = load_lang(lang)
        else:
            logger.warn('lang="' + lang + '" not in valid_langs, using lang="xx"')
            nlp[lang] = load_lang('xx')

    # process the text according to input flags
    if config.lower_case:
        text = text.lower()

    if config.remove_special_chars:
        text = text.translate(unicode_CPS)

    try:
        doc = nlp[lang](text)
    except ValueError as e:
        # FIXME:
        # How should we handle parsing errors?
        # Currently we simply panic and return None.
        # This means that the text will not be indexable from within postgres.
        # A more sophisticated strategy might try to remove the offending portion of text
        # so that the remainder of the text can still be indexed.
        #
        # Currently, the only known parsing error is that the Korean parser
        # panics when there is an emoji in the input.
        # It would be easy to manually remove emojis before passing to the Korean parser.
        logger.error(str(e) + ' ; lang=' + lang + ', text=' + text)
        return None

    def format_token(token, i):
        lemma = token.lemma_[:config.max_lemma_size]
        if add_positions:
            if lemma == ' ':
                return ' '
            else:
                return lemma + ':' + str(i + 1)
        else:
            return lemma

    def include_token(token):
        if config.remove_stop_words:
            return not token.is_stop
        else:
            return True

    lemmas = [format_token(token, i) for i, token in enumerate(doc) if include_token(token)]
    lemmas_joined = ' '.join(lemmas)

    # NOTE:
    # Some lemmatizers capitalize proper nouns even if the input text is lowercase.
    # That's why we need to lower case here and before doing the lemmatization.
    if config.lower_case and lang in ['ja', 'hr']:
        lemmas_joined = lemmas_joined.lower()

    return lemmas_joined


def tsvector_to_ngrams(tsv, n, uniq=True):
    '''
    Groups the ordered lexemes in a tsvector into an unordered set of n-grams.
    The results are intended to be used for computing statistics about the use of n-grams in a corpus of documents.

    The following examples show the output when combined with chajda's lemmatize function:

    >>> tsvector_to_ngrams(lemmatize('en', 'fancy apple pie crust is the most delicious fancy pie that I have ever eaten; I love pie.'), 1, False)
    ['fancy', 'apple', 'pie', 'crust', 'delicious', 'fancy', 'pie', 'eat', 'love', 'pie']

    >>> tsvector_to_ngrams(lemmatize('en', 'fancy apple pie crust is the most delicious fancy pie that I have ever eaten; I love pie.'), 2, False)
    ['fancy', 'apple', 'fancy apple', 'pie', 'apple pie', 'crust', 'pie crust', 'delicious', 'fancy', 'delicious fancy', 'pie', 'fancy pie', 'eat', 'love', 'pie', 'love pie']
    >>> tsvector_to_ngrams(lemmatize('en', 'fancy apple pie crust is the most delicious fancy pie that I have ever eaten; I love pie.'), 3, False)
    ['fancy', 'apple', 'fancy apple', 'pie', 'apple pie', 'fancy apple pie', 'crust', 'pie crust', 'apple pie crust', 'delicious', 'fancy', 'delicious fancy', 'pie', 'fancy pie', 'delicious fancy pie', 'eat', 'love', 'pie', 'love pie']

    These test cases use the same example string above, but tsv was generated from the postgres function to_tsvector,
    which lemmatizes differently than chajda's lemmatize function.

    >>> tsvector_to_ngrams("'appl':2 'crust':4 'delici':8 'eaten':15 'ever':14 'fanci':1,9 'love':17 'pie':3,10,18", 1, False)
    ['fanci', 'appl', 'pie', 'crust', 'delici', 'fanci', 'pie', 'ever', 'eaten', 'love', 'pie']
    >>> tsvector_to_ngrams("'appl':2 'crust':4 'delici':8 'eaten':25 'ever':14 'fanci':1,9 'love':17 'pie':3,10,18", 2, False)
    ['fanci', 'appl', 'fanci appl', 'pie', 'appl pie', 'crust', 'pie crust', 'delici', 'fanci', 'delici fanci', 'pie', 'fanci pie', 'ever', 'love', 'pie', 'love pie', 'eaten']
    >>> tsvector_to_ngrams("'appl':2 'crust':4 'delici':8 'eaten':25 'ever':14 'fanci':1,9 'love':17 'pie':3,10,18", 3, False)
    ['fanci', 'appl', 'fanci appl', 'pie', 'appl pie', 'fanci appl pie', 'crust', 'pie crust', 'appl pie crust', 'delici', 'fanci', 'delici fanci', 'pie', 'fanci pie', 'delici fanci pie', 'ever', 'love', 'pie', 'love pie', 'eaten']

    NOTE:
    Test cases specify that `uniq=False` because the output with `uniq=True` is non-deterministic.
    In practice inside postgres, we'll be using `uniq=True`.
    '''
    positioned_lexemes = _get_positioned_lexemes(tsv)
    ngrams = []
    for i,(pos,lexeme) in enumerate(positioned_lexemes):
        ngrams.append(lexeme)
        ngram = lexeme
        for j in range(1, min(n,i+1)):
            prev_pos,prev_lexeme = positioned_lexemes[i-j]
            if prev_pos == pos - j:
                ngram = prev_lexeme + ' ' + ngram
                ngrams.append(ngram)
            else:
                break
    if uniq:
        ngrams = set(ngrams)
    return ngrams


def tsvector_to_contextvectors(embedding, tsv, n=3, windowsize=10, method='weighted', a=1e-3, normalize=False):
    '''

    >>> assert len(tsvector_to_contextvectors(get_test_embedding('en'), lemmatize('en','war and peace'))) == 2

    FIXME:
    we should also be returning the number of times that a word is used; possibly also its stddev?
    '''

    # compute contextvectors from wordcontext
    wordcontext = tsvector_to_wordcontext(tsv, n, windowsize)
    contextvectors = defaultdict(lambda: 0.0)
    count_total = defaultdict(lambda: 0)
    for word,context,count in wordcontext:
        try:
            contextvector = embedding.kv[context]
            updatevector = contextvector*count
            if method == 'weighted':
                updatevector *= a/(a + embedding.word_frequency(word))
            contextvectors[word] += updatevector
            count_total[word] += count
        except KeyError:
            pass
    if normalize:
        for word in count_total.keys():
            contextvectors[word] /= count_total[word]

    return dict(contextvectors)


def tsvector_to_wordcontext(tsv, n, windowsize):
    '''
    Converts a document into a dictionary of (focus_word, context_words) pairs suitable for word2vec type training.
    The context_words are dictionaries of (word, count) pairs.

    FIXME:
    tsvectors do not contain any information about sentence boundaries;
    therefore, we cannot limit the context to include only text from the same sentence

    params:
        tsv: a tsvector (represented as a str in python)
        n: the length of ngrams for focus words; contexts will only use 1-grams
        windowsize: the size of the context to the left and right of the focus word

    The following examples show how the wordcontexts are derived: 
    >>> tsvector_to_wordcontext(lemmatize('en', 'aaa'), 5, 1)
    []
    >>> tsvector_to_wordcontext(lemmatize('en', 'aaa bbb'), 5, 1)
    [('aaa', 'bbb', 1), ('bbb', 'aaa', 1)]
    >>> tsvector_to_wordcontext(lemmatize('en', 'aaa bbb ccc ddd eee'), 1, 2)
    [('aaa', 'bbb', 1), ('aaa', 'ccc', 1), ('bbb', 'aaa', 1), ('bbb', 'ccc', 1), ('bbb', 'ddd', 1), ('ccc', 'aaa', 1), ('ccc', 'bbb', 1), ('ccc', 'ddd', 1), ('ccc', 'eee', 1), ('ddd', 'bbb', 1), ('ddd', 'ccc', 1), ('ddd', 'eee', 1), ('eee', 'ccc', 1), ('eee', 'ddd', 1)]
    >>> tsvector_to_wordcontext(lemmatize('en', 'aaa bbb ccc ddd eee'), 2, 2)
    [('aaa', 'bbb', 1), ('aaa', 'ccc', 1), ('aaa bbb', 'ccc', 1), ('aaa bbb', 'ddd', 1), ('bbb', 'aaa', 1), ('bbb', 'ccc', 1), ('bbb', 'ddd', 1), ('bbb ccc', 'aaa', 1), ('bbb ccc', 'ddd', 1), ('bbb ccc', 'eee', 1), ('ccc', 'aaa', 1), ('ccc', 'bbb', 1), ('ccc', 'ddd', 1), ('ccc', 'eee', 1), ('ccc ddd', 'aaa', 1), ('ccc ddd', 'bbb', 1), ('ccc ddd', 'eee', 1), ('ddd', 'bbb', 1), ('ddd', 'ccc', 1), ('ddd', 'eee', 1), ('ddd eee', 'bbb', 1), ('ddd eee', 'ccc', 1), ('eee', 'ccc', 1), ('eee', 'ddd', 1)]

    Note that stop words are removed before any processing and have no effect on the results:

    >>> tsvector_to_wordcontext(lemmatize('en', 'aaa ccc ddd eee'), 2, 2)
    [('aaa', 'ccc', 1), ('aaa', 'ddd', 1), ('aaa ccc', 'ddd', 1), ('aaa ccc', 'eee', 1), ('ccc', 'aaa', 1), ('ccc', 'ddd', 1), ('ccc', 'eee', 1), ('ccc ddd', 'aaa', 1), ('ccc ddd', 'eee', 1), ('ddd', 'aaa', 1), ('ddd', 'ccc', 1), ('ddd', 'eee', 1), ('ddd eee', 'aaa', 1), ('ddd eee', 'ccc', 1), ('eee', 'ccc', 1), ('eee', 'ddd', 1)]
    >>> tsvector_to_wordcontext(lemmatize('en', 'aaa and ccc ddd eee'), 2, 2)
    [('aaa', 'ccc', 1), ('aaa', 'ddd', 1), ('aaa ccc', 'ddd', 1), ('aaa ccc', 'eee', 1), ('ccc', 'aaa', 1), ('ccc', 'ddd', 1), ('ccc', 'eee', 1), ('ccc ddd', 'aaa', 1), ('ccc ddd', 'eee', 1), ('ddd', 'aaa', 1), ('ddd', 'ccc', 1), ('ddd', 'eee', 1), ('ddd eee', 'aaa', 1), ('ddd eee', 'ccc', 1), ('eee', 'ccc', 1), ('eee', 'ddd', 1)]

    Here's a more realistic example:

    >>> tsvector_to_wordcontext(lemmatize('en', 'fancy apple pie crust is the most delicious fancy pie that I have ever eaten; I love pie.'), 2, 2)
    [('fancy', 'apple', 1), ('fancy', 'pie', 2), ('fancy', 'crust', 1), ('fancy', 'delicious', 1), ('fancy', 'eat', 1), ('fancy apple', 'pie', 1), ('fancy apple', 'crust', 1), ('apple', 'fancy', 1), ('apple', 'pie', 1), ('apple', 'crust', 1), ('apple pie', 'fancy', 1), ('apple pie', 'crust', 1), ('apple pie', 'delicious', 1), ('pie', 'fancy', 2), ('pie', 'apple', 1), ('pie', 'crust', 1), ('pie', 'delicious', 2), ('pie', 'eat', 2), ('pie', 'love', 2), ('pie crust', 'fancy', 2), ('pie crust', 'apple', 1), ('pie crust', 'delicious', 1), ('crust', 'apple', 1), ('crust', 'pie', 1), ('crust', 'delicious', 1), ('crust', 'fancy', 1), ('crust delicious', 'apple', 1), ('crust delicious', 'pie', 2), ('crust delicious', 'fancy', 1), ('delicious', 'pie', 2), ('delicious', 'crust', 1), ('delicious', 'fancy', 1), ('delicious fancy', 'pie', 2), ('delicious fancy', 'crust', 1), ('delicious fancy', 'eat', 1), ('fancy pie', 'crust', 1), ('fancy pie', 'delicious', 1), ('fancy pie', 'eat', 1), ('fancy pie', 'love', 1), ('pie eat', 'delicious', 1), ('pie eat', 'fancy', 1), ('pie eat', 'love', 1), ('pie eat', 'pie', 1), ('eat', 'fancy', 1), ('eat', 'pie', 2), ('eat', 'love', 1), ('eat love', 'fancy', 1), ('eat love', 'pie', 2), ('love', 'pie', 2), ('love', 'eat', 1), ('love pie', 'pie', 1), ('love pie', 'eat', 1)]
    '''
    positioned_lexemes = _get_positioned_lexemes(tsv)
    positioned_lexemes.sort()
    ordered_lexemes = [lexeme for position,lexeme in positioned_lexemes]

    # wordcontext is a dictionary of dictionaries;
    # the outer key is the focus word, the inner key is the context word, and the inner value is the count
    wordcontext = defaultdict(lambda: Counter())
    for i,lexeme in enumerate(ordered_lexemes):
        for j in range(min(n, len(ordered_lexemes)-i)):
            word = ' '.join(ordered_lexemes[i:i+j+1])
            context_left = ordered_lexemes[max(0,i-windowsize):i]
            context_right = ordered_lexemes[i+j+1:i+j+1+windowsize]
            wordcontext[word] += Counter(context_left + context_right)

    # convert the dictionary of dictionaries into a flat list suitable for storing in a SQL table
    output = []
    for focusword,context in wordcontext.items():
        for contextword,count in context.items():
            output.append((focusword,contextword,count))
    return output


def _get_positioned_lexemes(tsv):
    '''
    This helper functions converts a tsvector input into the original stream of words that generated the tsvector.
    Since stop words are removed when creating a tsvector, the words are labelled with their position in the original text.

    >>> _get_positioned_lexemes(lemmatize('en', 'aaa'))
    [(1, 'aaa')]
    >>> _get_positioned_lexemes(lemmatize('en', 'aaa bbb'))
    [(1, 'aaa'), (2, 'bbb')]
    >>> _get_positioned_lexemes(lemmatize('en', 'aaa bbb ccc'))
    [(1, 'aaa'), (2, 'bbb'), (3, 'ccc')]
    >>> _get_positioned_lexemes(lemmatize('en', 'aaa bbb ccc aaa'))
    [(1, 'aaa'), (2, 'bbb'), (3, 'ccc'), (4, 'aaa')]
    >>> _get_positioned_lexemes(lemmatize('en', 'fancy apple pie crust is the most delicious fancy pie that I have ever eaten; I love pie.'))
    [(1, 'fancy'), (2, 'apple'), (3, 'pie'), (4, 'crust'), (8, 'delicious'), (9, 'fancy'), (10, 'pie'), (15, 'eat'), (17, 'love'), (18, 'pie')]
    '''
    positioned_lexemes = []
    for item in tsv.split():
        try:
            lexeme, positions = item.split(':')
            for position in positions.split(','):
                try:
                    position = int(position)
                    positioned_lexemes.append((position,lexeme.strip("'")))
                except ValueError:
                    logger.error('ValueError: position not an int')

        # FIXME: 
        # there are some items without colons, causing the split() call to fail;
        # this should never happen, and I don't know why it is;
        # we must catch this error so that postgres doesn't crash when the error is thrown
        except ValueError:
            logger.error('ValueError: malformed tsvector lexeme')

    positioned_lexemes.sort()
    return positioned_lexemes
