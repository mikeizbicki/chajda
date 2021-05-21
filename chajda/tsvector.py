'''
In the postgres database, the `tsvector` type is used to represent a document for full text search.
This file uses the spacy library to convert input text into a `tsvector`.
The main user-facing function is `lemmatize`.
'''

import pkgutil
import importlib
import inspect
import spacy

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
from collections import defaultdict
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

    def __init__(self, lower_case=True, remove_special_chars=True, remove_stop_words=True):
        self.lower_case=lower_case
        self.remove_special_chars=remove_special_chars
        self.remove_stop_words=remove_stop_words


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
        if add_positions:
            if token.lemma_ == ' ':
                return ' '
            else:
                return token.lemma_ + ':' + str(i + 1)
        else:
            return token.lemma_

    def include_token(token):
        if config.remove_stop_words:
            return not token.is_stop
        else:
            return True

    # NOTE:
    # in the code below, we take only the first 500 characters of each token;
    # this is because the postgresql btree implementation throws an error when
    # used on an input field with more than 2000 bytes;
    # all unicode characters take at most 4 bytes per character,
    # so by truncating to length 500,
    # we are guaranteed that the number of bytes will be less than 2000.
    # From a practical perspective, no real words should ever be this long,
    # so this won't effect precision/recall.
    lemmas = [format_token(token, i)[:500] for i, token in enumerate(doc) if include_token(token)]
    lemmas_joined = ' '.join(lemmas)

    # NOTE:
    # Some lemmatizers capitalize proper nouns even if the input text is lowercase.
    # That's why we need to lower case here and before doing the lemmatization.
    if config.lower_case and lang in ['ja', 'hr']:
        lemmas_joined = lemmas_joined.lower()

    return lemmas_joined



def lemmatize_query(
        lang,
        text,
        lower_case=True,
        remove_special_chars=True,
        remove_stop_words=True,
        ):
    '''
    >>> lemmatize_query('xx', 'Abraham Lincoln was president of the United States')
    'abraham & lincoln & was & president & of & the & united & states'
    >>> lemmatize_query('en', 'Abraham Lincoln was president of the United States')
    'abraham & lincoln & president & unite & state'
    >>> lemmatize_query('en', '      Abraham Lincoln was president of   the     United     States   ')
    'abraham & lincoln & president & unite & state'


    FIXME:
    we should add quotations like in the following examples

    #>>> lemmatize_query('en', '"Abraham Lincoln" was president of the United States')
    #'abraham <1> lincoln & president & united & state'
    #>>> lemmatize_query('en', '"Abraham Lincoln" was "president of the United States"')
    #'abraham <1> lincoln & president <2> united <1> state'
    '''
    lemmas = lemmatize(
        lang,
        text,
        add_positions=False,
        config = Config(
            lower_case=lower_case,
            remove_special_chars=remove_special_chars,
            remove_stop_words=remove_stop_words,
            )
        )
    return ' & '.join(lemmas.split())


