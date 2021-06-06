'''
'''

from chajda.tsvector import lemmatize, Config
from chajda.tsquery.__init__ import to_tsquery


from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """Context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)




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
from annoy import AnnoyIndex

fasttext_models = {}

def augments_fasttext(lang, word, config=Config(), n=5):
    ''' 
    Returns n words that are "similar" to the input word in the target language.
    These words can be used to augment a search with the Query class.

    Note: Due to the size of fasttext models, testing multiple languages in the doctest requires more space than github actions allows for. For this reason, only english is tested in the doctest.

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
    '''

    try:
        fasttext_models[lang]
    except:
        with suppress_stdout_stderr():
            fasttext.util.download_model(lang, if_exists='ignore')
        fasttext_models[lang] = fasttext.load_model('cc.{0}.300.bin'.format(lang))
    #print(fasttext_models[lang].words)
    index = AnnoyIndex(300, 'angular')
    #populating AnnoyIndex with vectors from fasttext model
    i = 0
    for j in fasttext_models[lang].words:
        v = fasttext_models[lang][j]
        index.add_item(i,v)
        i += 1

    index.build(10)
    index.save('test.ann')
    index.save('test.ann')
    index.load('test.ann')

    #find the most similar words using annoy index
    try:
        n_nearest_neighbor_indices = index.get_nns_by_vector(fasttext_models[lang][word], n)
        n_nearest_neighbors = []
        for i in range(n):
            n_nearest_neighbors.append(fasttext_models[lang].words[n_nearest_neighbor_indices[i]])
        words = ' '.join([ word for word in n_nearest_neighbors ])
        print("words = ", words)
    except KeyError:
        return []




    #find the most similar words
   # try:
   #     topn = fasttext_models[lang].get_nearest_neighbors(word, k=n)
   #     words = ' '.join([ word for (rank, word) in topn ])
   # except KeyError:
   #     return []

    # lemmatize the results so that they'll be in the search document's vocabulary
    words = lemmatize(lang, words, add_positions=False, config=config).split()
    words = list(filter(lambda w: len(w)>1 and w != word, words))[:n]


    return words
