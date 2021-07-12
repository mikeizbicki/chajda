'''
'''
import os
import json
import fasttext
import fasttext.util
from annoy import AnnoyIndex
from collections import namedtuple
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


################################################################################
# fasttext
################################################################################

# path used for storing/loading fasttext models and annoy indices
fasttext_annoy_data_path = os.path.dirname(__file__)

# dictionary used for storing languages as keys, and corresponding loaded fasttext models as values
fasttext_models = {}


def load_fasttext(lang):
    '''
    Downloads and loads a fasttext model for a particular language,
    storing the fasttext model object in the fasttext_models dictionary.
    Due to the size of fasttext models (>1gb), and the loading time,
    we do not load all models on program startup.
    Instead, this function is used to load them lazily as needed.
    lang: language passed from augments_fasttext

    NOTE:
    The doctest below has been commented out due to requiring more memory than github actions allows for.
    # >>> load_fasttext_model('ja') or fasttext_models['ja'] != None
    # True
    '''

    # suppress_stdout_stderr() is used for redirecting stdout and stderr when downloading and loading fasttext models.
    # Without this, stdout and stderr from fasttext causes doctests to fail.
    with suppress_stdout_stderr():

        # fasttext downloads models to the current directory,
        # but we want to store the model in chaja/tsquery/fasttext,
        # so we change the working directory before downloading
        working_dir = os.getcwd()
        os.makedirs(f"{fasttext_annoy_data_path}/fasttext", exist_ok=True)
        os.chdir(f"{fasttext_annoy_data_path}/fasttext")

        # download and load fasttext model
        fasttext.util.download_model(lang, if_exists='ignore')
        fasttext_models[lang] = fasttext.load_model(f"cc.{lang}.300.bin")

        # fasttext downloads cc.{lang}.300.bin.gz and then decompresses the .gz file into cc.{lang}.300.bin,
        # however, we only need cc.{lang}.300.bin,
        # so we remove unneeded .gz file if exists
        if os.path.isfile(f"{fasttext_annoy_data_path}/fasttext/cc.{lang}.300.bin.gz"):
            os.remove(f"{fasttext_annoy_data_path}/fasttext/cc.{lang}.300.bin.gz")

        # change working directory back
        os.chdir(working_dir)

    # Executing fasttext's get_nearest_neighbor as well as accessing the words in the fasttext model both run
    # significantly faster after they have been called once,
    # so we call them once upon loading the model to ensure most efficient runtime for subsequent uses.
    fasttext_models[lang].get_nearest_neighbors('google', k=5)
    fasttext_list = []
    for word in fasttext_models[lang].get_words(on_unicode_error='replace'):
        fasttext_list.append(word)


def destroy_fasttext(lang):
    '''
    removes fasttext model from disk and memory
    '''

    # remove fasttext download from disk
    if os.path.isfile(f"{fasttext_annoy_data_path}/fasttext/cc.{lang}.300.bin"):
        os.remove(f"{fasttext_annoy_data_path}/fasttext/cc.{lang}.300.bin")

    # remove fasttext model from memory
    try:
        del fasttext_models[lang]
    except KeyError:
        pass


################################################################################
# annoy
################################################################################

# dictionary used for storing language abbreviations as keys,
# and a namedtuple containing the corresponding annoy index, as well as two json files, as values
annoy_indices = {}

# dictionaries for storing fasttext model words, and their corresponding indices;
# fasttext_index_for_word stores fasttext words as keys, and corresponding indices as values;
# fasttext_word_at_index stores fasttext word indices as keys, and corresponding words as values;
# both dictionaries are populated by the load_annoy_index function,
# and subsequently written to json files
# NOTE:
# these dictionaries are necessary for using annoy for nearest neighbor query after removing fasttext download
fasttext_index_for_word = {}
fasttext_word_at_index = {}


def load_annoy(lang, word='google', n=5, seed=22, run_destroy_fasttext = True):
    '''
    Checks whether an annoy index has already been created for a specified language.
    Populates an annoy index  with vectors from the fasttext model that corresponds to language,
    builds the index, saves it to disk.
    The annoy index is stored in the annoy dictionary.
    lang: language passed from augments_fasttext

    NOTE:
    Populating an annoy index with vectors from a fasttext model requires more memory than github actions allows for.
    Therefore, the doctest has been commented out.
    # >>> create_annoy_index_if_needed('en') or annoy_indices['en'] != None
    # True
    '''
    # create annoy index if not already created
    try:
        annoy_indices[lang]
    except KeyError:
        annoy_indices[lang] = namedtuple("annoy", ['index', 'index_for_word', 'word_at_index'])
        annoy_indices[lang].index = AnnoyIndex(300, 'angular')

        # if annoy index has not been saved for this language yet,
        # populate annoy index with vectors from corresponding fasttext model
        try:
            annoy_indices[lang].index.load(f"{fasttext_annoy_data_path}/annoy/{lang}{seed}.ann")

        # OSError occurs if index is not already saved to disk
        except OSError:
            # download fasttext model if not already downloaded
            try:
                fasttext_models[lang]
            except KeyError:
                load_fasttext(lang)

            # The annoy library uses a random number genertor when building up the trees for an Annoy Index.
            # In order to ensure that the output is deterministic, we specify the seed value for the random number generator.
            annoy_indices[lang].index.set_seed(seed)

            # Populate annoy index with vectors from corresponding fasttext model
            i = 0
            for j in fasttext_models[lang].get_words(on_unicode_error='replace'):
                v = fasttext_models[lang][j]
                annoy_indices[lang].index.add_item(i,v)
                i += 1

            # build the trees for the index
            annoy_indices[lang].index.build(10)

            # save the index to annoy directory,
            # make the directory if needed
            try:
                annoy_indices[lang].index.save(f"{fasttext_annoy_data_path}/annoy/{lang}{seed}.ann")
            except OSError:
                os.mkdir(f"{fasttext_annoy_data_path}/annoy")
                annoy_indices[lang].index.save(f"{fasttext_annoy_data_path}/annoy/{lang}{seed}.ann")

            # populate fasttext_index_for_word and fasttext_word_at_index with words from fasttext model dictionary,
            # as well as indices for each word 
            i=0
            for word in fasttext_models[lang].get_words(on_unicode_error='replace'):
                fasttext_index_for_word[word] = i
                fasttext_word_at_index[i] = word
                i += 1

            # write fasttext_index_for_word to json file
            with open(f"{fasttext_annoy_data_path}/annoy/fasttext_index_for_word_{lang}_lookup.json", "w") as outfile:
                json.dump(fasttext_index_for_word, outfile)

            # write fasttext_word_at_index to json file
            with open(f"{fasttext_annoy_data_path}/annoy/fasttext_word_at_index_{lang}_lookup.json", "w") as outfile:
                json.dump(fasttext_word_at_index, outfile)

            json_index_for_word = open(f"{fasttext_annoy_data_path}/annoy/fasttext_index_for_word_{lang}_lookup.json", "r")
            annoy_indices[lang].index_for_word = json.load(json_index_for_word)

            json_word_at_index = open(f"{fasttext_annoy_data_path}/annoy/fasttext_word_at_index_{lang}_lookup.json", "r")
            annoy_indices[lang].word_at_index = json.load(json_word_at_index)

            # remove fasttext model from disk and memory
            if run_destroy_fasttext:
                destroy_fasttext(lang)


def destroy_annoy(lang, seed=22):
    '''
    removes annoy index from disk and memory
    '''

    # remove annoy index from disk
    if os.path.isfile(f"{fasttext_annoy_data_path}/annoy/{lang}{seed}.ann"):
        os.remove(f"{fasttext_annoy_data_path}/annoy/{lang}{seed}.ann")

    # remove annoy index from memory
    try:
        del annoy_indices[lang]
    except KeyError:
        pass

    # remove json files from disk and memory
    if os.path.isfile(f"{fasttext_annoy_data_path}/annoy/fasttext_index_for_word_{lang}_lookup.json"):
        os.remove(f"{fasttext_annoy_data_path}/annoy/fasttext_index_for_word_{lang}_lookup.json")

    try:
        del annoy_indices[lang].index_for_word
    except KeyError:
        pass

    if os.path.isfile(f"{fasttext_annoy_data_path}/annoy/fasttext_word_at_index_{lang}_lookup.json"):
        os.remove(f"{fasttext_annoy_data_path}/annoy/fasttext_word_at_index_{lang}_lookup.json")

    try:
        del annoy_indices[lang].word_at_index
    except KeyError:
        pass


def augments_fasttext(lang, word, config=Config(), n=5, annoy=True):
    ''' 
    Returns n words that are "similar" to the input word in the target language.
    These words can be used to augment a search with the Query class.
    This function will default to using the annoy library to get the nearest neighbors.

    >>> to_tsquery('en', 'baby boy', augment_with=lambda lang,word,config: augments_fasttext(lang,word,config,5,False))
    '(baby:A | newborn:B | infant:B) & (boy:A | girl:B | boyhe:B | boyit:B)'

    >>> to_tsquery('en', '"baby boy"', augment_with=lambda lang,word,config: augments_fasttext(lang,word,config,5,False))
    'baby:A <1> boy:A'

    >>> to_tsquery('en', '"baby boy" (school | home) !weapon', augment_with=lambda lang,word,config: augments_fasttext(lang,word,config,5,False))
    '(baby:A <1> boy:A) & ((school:A | schoo:B | schoolthe:B | schoool:B | kindergarten:B) | (home:A | house:B | homethe:B | homewhen:B | homethis:B)) & !(weapon:A | weaponthe:B | weopon:B)'

    >>> augments_fasttext('en','weapon', n=5, annoy=False)
    ['weaponthe', 'weopon']

    >>> augments_fasttext('en','king', n=5, annoy=False)
    ['queen', 'kingthe']
    
    NOTE:
    Populating an AnnoyIndex with vectors from a fasttext model requires more memory than github actions allows for.
    Therefore, the doctests involving the annoy library have been commented out below.

    # >>> augments_fasttext('en','weapon', n=5)
    # ['nonweapon', 'loadout', 'dualwield', 'autogun']

    # >>> augments_fasttext('en','king', n=5)
    # ['kingthe', 'kingly']

    NOTE:
    Due to the size of fasttext models (>1gb),
    testing multiple languages in the doctest requires more space than github actions allows for.
    For this reason, tests involving languages other than English have been commented out below.

    # >>> augments_fasttext('ja','さようなら', n=5)
    # ['さよなら', 'バイバイ', 'サヨウナラ', 'さらば', 'おしまい']

    # >>> augments_fasttext('es','escuela', n=5)
    # ['escuelala', 'academia', 'universidad', 'laescuela']
    '''

    # augments_fasttext defaults to using the annoy library to find nearest neighbors,
    # if annoy==False is passed into the function, then the fasttext library will be used.
    if annoy:

        # populate and load AnnoyIndex with vectors from fasttext model if not already populated
        load_annoy(lang)

        # find the most similar words using annoy library
        n_nearest_neighbor_indices = annoy_indices[lang].index.get_nns_by_item(annoy_indices[lang].index_for_word[word], n)
        n_nearest_neighbors = [ annoy_indices[lang].word_at_index[f"{i}"] for i in n_nearest_neighbor_indices ]

        words = ' '.join([ word for word in n_nearest_neighbors ])

    else:
        # download fasttext model if not already downloaded
        try:
            fasttext_models[lang]
        except KeyError:
            load_fasttext(lang)

        # find the most similar words using fasttext library
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
