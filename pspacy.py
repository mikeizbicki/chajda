import pkgutil
import importlib
import inspect
import spacy

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
import spacy.lang.ko
for stopword in ['이거', '이것', '는', '은', '가', '이', '을', '를', '기', '에']:
    spacy.lang.ko.stop_words.STOP_WORDS.add(stopword)


# this function loads a spacy language model
# it is used for lazily loading languages as they are needed
# FIXME:
# I'm pretty sure there should be an easier way to load a language model from the iso code,
# but I couldn't figure out a native way to do this.
def load_lang(lang_iso):

    logger.info('initializing ' + lang_iso)
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
    In typical usage, we want spacy's languages loaded lazily.
    When debugging and testing, however, it can be useful to force the immediate loading of all languages.
    '''
    if langs is None:
        langs = valid_langs

    for lang in langs:
        nlp[lang] = load_lang(lang)


# the nlp dictionary will hold the loaded spacy models,
# and entry of None indicates that the model still needs to be loaded
from collections import defaultdict
nlp = defaultdict(lambda: None)
nlp['xx'] = load_lang('xx')

# create a table of Unicode special characters for filtering
# this variable is used within the lemmatize function,
# but we precompute it outside the function for efficiency
import unicodedata
import sys
unicode_CPS = dict.fromkeys(i for i in range(0, sys.maxunicode + 1) if unicodedata.category(chr(i)).startswith(('P', 'S', 'C')))


def lemmatize_query(
        lang,
        text,
        lower_case=True,
        remove_special_chars=True,
        remove_stop_words=True,
        ):
    '''
    >>> lemmatize_query('xx', 'Abraham Lincoln was president of the United States')
    'abraham & lincoln & was & president & of & the & unite & states'
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
        lower_case=lower_case,
        remove_special_chars=remove_special_chars,
        remove_stop_words=remove_stop_words,
        add_positions=False
        )
    return ' & '.join(lemmas.split())


# this is the main function that gets called from postgresql
def lemmatize(
        lang,
        text,
        lower_case=True,
        remove_special_chars=True,
        remove_stop_words=True,
        add_positions=True,
        ):

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
            nlp[lang] = nlp['xx']

    # process the text according to input flags
    if lower_case:
        text = text.lower()

    if remove_special_chars:
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
        if remove_stop_words:
            return not token.is_stop
        else:
            return True

    lemmas = [format_token(token, i) for i, token in enumerate(doc) if include_token(token)]
    lemmas_joined = ' '.join(lemmas)

    # NOTE:
    # Some lemmatizers capitalize proper nouns even if the input text is lowercase.
    # That's why we need to lower case here and before doing the lemmatization.
    if lower_case and lang in ['ja', 'hr']:
        lemmas_joined = lemmas_joined.lower()

    return lemmas_joined
