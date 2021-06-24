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

import os
import fasttext
import fasttext.util
from annoy import AnnoyIndex

# path used for storing/loading fasttext models and annoy indices
fasttext_annoy_data_path = os.path.dirname(__file__)

fasttext_models = {}
annoy_indices = {}


def create_annoy_index_if_needed(lang, word='google', n=5, seed=22):
    '''
    Checks whether an annoy index has already been created for a specified language.
    Populates an annoy index  with vectors from the fasttext model that corresponds to language,
    builds the index, saves it to disk. 
    The annoy index is stored in the annoy_indices dictionary.

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
        annoy_indices[lang] = AnnoyIndex(300, 'angular')

        # if annoy index has not been saved for this language yet,
        # populate annoy index with vectors from corresponding fasttext model
        try:
            annoy_indices[lang].load(f"{fasttext_annoy_data_path}/annoy/{lang}{seed}.ann")

        # OSError occurs if index is not already saved to disk
        except OSError:
            # The annoy library uses a random number genertor when building up the trees for an Annoy Index.
            # In order to ensure that the output is deterministic, we specify the seed value for the random number generator.
            annoy_indices[lang].set_seed(seed)

            # Populate annoy index with vectors from corresponding fasttext model
            i = 0
            for j in fasttext_models[lang].words:
                v = fasttext_models[lang][j]
                annoy_indices[lang].add_item(i,v)
                i += 1

            # build the trees for the index    
            annoy_indices[lang].build(10)

            # save the index to annoy directory,
            # make the directory if needed
            try:
                annoy_indices[lang].save(f"{fasttext_annoy_data_path}/annoy/{lang}{seed}.ann")
            except OSError:
                os.mkdir(f"{fasttext_annoy_data_path}/annoy")
                annoy_indices[lang].save(f"{fasttext_annoy_data_path}/annoy/{lang}{seed}.ann")


def load_fasttext_model(lang):
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
    working_dir = os.getcwd()
    # suppress_stdout_stderr() is used for redirecting stdout and stderr when downloading and loading fasttext models.
    # Without this, stdout and stderr from fasttext causes doctests to fail. 
    with suppress_stdout_stderr():

        # fasttext downloads models to the current directory,
        # but we want to store the model in chaja/tsquery/fasttext,
        # so we change the working directory before downloading
        try:      
            os.chdir(f"{fasttext_annoy_data_path}/fasttext")
        except OSError:
            os.mkdir(f"{fasttext_annoy_data_path}/fasttext")
            os.chdir(f"{fasttext_annoy_data_path}/fasttext")

        # download and load fasttext model
        fasttext.util.download_model(lang, if_exists='ignore')
        fasttext_models[lang] = fasttext.load_model(f"cc.{lang}.300.bin")

        # change working directory back
        os.chdir(working_dir)

    # Executing fasttext's get_nearest_neighbor as well as accessing the words in the fasttext model both run
    # significantly faster after they has been called once,
    # so we call them once upon loading the model to ensure most efficient runtime for subsequent uses.
    fasttext_models[lang].get_nearest_neighbors('google', k=5)
    fasttext_list = []
    for word in fasttext_models[lang].words:
        fasttext_list.append(word)


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
    Populating an annoy index with vectors from a fasttext model requires more memory than github actions allows for.
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
    # download and load the fasttext model if it's not already loaded
    try:
        fasttext_models[lang]
    except KeyError:
        load_fasttext_model(lang)
    
    # augments_fasttext defaults to using the annoy library to find nearest neighbors,
    # if annoy==False is passed into the function, then the fasttext library will be used.
    if annoy:

        # populate and load annoy index with vectors from fasttext model if not already populated
        create_annoy_index_if_needed(lang)

        # find the most similar words using annoy library
        n_nearest_neighbor_indices = annoy_indices[lang].get_nns_by_vector(fasttext_models[lang][word], n)
        n_nearest_neighbors = [fasttext_models[lang].words[i] for i in n_nearest_neighbor_indices]
        words = ' '.join([ word for word in n_nearest_neighbors ])

    else:
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
