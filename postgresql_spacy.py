import pkgutil
import importlib
import inspect
import pprint
import spacy

# initialize logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# calculate the valid languages for the installed spacy version
valid_langs = []
for _, lang_iso, is_lang in pkgutil.iter_modules(spacy.lang.__path__):
    if is_lang:
        valid_langs.append(lang_iso)
logger.info("valid_langs="+str(valid_langs))

# this function loads a spacy language model
# it is used for lazily loading languages as they are needed
def load_lang(lang_iso):

    module_name = 'spacy.lang.'+lang_iso
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
    shortest_name_length = min([ len(name) for name,obj in filtered_classes ])
    nlp = list(filter(lambda x: len(x[0])==shortest_name_length, filtered_classes))[0][1]()

    return nlp

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

# this is the main function that gets called from postgresql
def lemmatize(lang,text,lower_case=True,no_special_chars=True,add_positions=False):

    if lang is None or text is None:
        return None

    if nlp[lang] is None:
        if lang in valid_langs:
            logger.info('initializing '+lang)
            nlp[lang] = load_lang(lang)
        else:
            logger.warn('lang="'+lang+'" not in valid_langs, using lang="xx"')
            nlp[lang] = nlp['xx']

    if lower_case:
        text = text.lower()

    if no_special_chars:
        text = text.translate(unicode_CPS)

    try:
        doc = nlp[lang](text) 
    except ValueError as e:
        logger.error(str(e))
        return None

    if add_positions:
        lemmas = ' '.join([ token.lemma_+':'+str(i+1) for i,token in enumerate(doc) ])
    else:
        lemmas = ' '.join([ token.lemma_ for token in doc ])
    
    # NOTE:
    # The Japanese lemmatizer capitalizes proper nouns even if the input text is lowercase.
    # That's why we need to lower case here and before doing the lemmatization.
    if lower_case:
        lemmas = lemmas.lower()

    return lemmas
