import gzip
import numpy as np
import os
import shutil
import tempfile
import urllib
from gensim.models import KeyedVectors
from gensim.similarities.annoy import AnnoyIndexer
from chajda.cache import LRUCache

# initialize logging
import logging
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    )
logger = logging.getLogger(__name__)

# global variables that store information about the available embeddings 
# see:
# https://fasttext.cc/docs/en/aligned-vectors.html
# https://fasttext.cc/docs/en/crawl-vectors.html
_fasttext_langs_aligned = ['af','ar','bg','bn','bs','ca','cs','da','de','el','en','es','et','fa','fi','fr','he','hi','hr','hu','id','it','ko','lt','lv','mk','ms','nl','no','pl','pt','ro','ru','sk','sl','sq','sv','ta','th','tl','tr','uk','vi','zh',]
_embeddings_fasttext_aligned = [ 
    (f'wiki.{lang}.align.vec', lang, f'https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.{lang}.align.vec')
    for lang in _fasttext_langs_aligned
    ]

_fasttext_langs = ['af','als','am','an','ar','arz','as','ast','az','azb','ba','bar','bcl','be','bg','bh','bn','bo','bpy','br','bs','ca','ce','ceb','ckb','co','cs','cv','cy','da','de','diq','dv','el','eml','en','eo','es','et','eu','fa','fi','fr','frr','fy','ga','gd','gl','gom','gu','gv','he','hi','hif','hr','hsb','ht','hu','hy','ia','id','ilo','io','is','it','ja','jv','ka','kk','km','kn','ko','ku','ky','la','lb','li','lmo','lt','lv','mai','mg','mhr','min','mk','ml','mn','mr','mrj','ms','mt','mwl','my','myv','mzn','nah','nap','nds','ne','new','nl','nn','no','nso','oc','or','os','pa','pam','pfl','pl','pms','pnb','ps','pt','qu','rm','ro','ru','sa','sah','sc','scn','sco','sd','sh','si','sk','sl','so','sq','sr','su','sv','sw','ta','te','tg','th','tk','tl','tr','tt','ug','uk','ur','uz','vec','vi','vls','vo','wa','war','xmf','yi','yo','zea','zh']
_embeddings_fasttext = [
    (f'cc.{lang}.300.vec', lang, f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.vec.gz')
    for lang in _fasttext_langs
    ]

embeddings = _embeddings_fasttext_aligned + _embeddings_fasttext


# the keys represent the embedding names and the values the embedding;
# this dictionary shouldn't be accessed directly from user code
_embeddings_loaded = LRUCache(maxitems=10)


def get_test_embedding(lang):
    return get_embedding(lang=lang, max_n=50000, annoy=False)


def get_embedding(annoy=True, **kwargs):
    '''
    Return an Embedding class has been loaded into memory.

    The first time this function is run with a particular set of arguments it will be slow,
    but subsequent runs will be fast.
    This is because:
    if the embedding has not been downloaded, it will download it;
    if the download has not been loaded into memory, it will load it;
    if the model has already been loaded into memory, it will return it instantly.

    >>> get_embedding(lang='en', max_n=50000, max_d=50).word_frequency('hello')
    1.8072576301422368e-05
    >>> get_embedding(lang='en', max_n=50000).word_frequency('hello')
    1.8072576301422368e-05
    >>> get_embedding(lang='en', max_n=10000, max_d=50).word_frequency('hello')
    2.1044290373194325e-05
    '''
    embedding = Embedding(**kwargs)
    if embedding.internal_name not in _embeddings_loaded:
        embedding.load_kv()
        if annoy:
            embedding.load_annoy()
        _embeddings_loaded[embedding.internal_name] = embedding
    return _embeddings_loaded[embedding.internal_name]


class Embedding():
    '''
    This class provides a unified interface for working with word embeddings.
    It is not meant to be constructed directly, but instead created using the get_embedding function.

    >>> assert Embedding(lang='en')
    >>> assert Embedding(lang='ko')
    >>> assert Embedding(lang='xx')
    >>> assert Embedding(lang='us')
    >>> assert Embedding(lang='undefined')
    >>> assert Embedding(name='wiki.en.align.vec')
    >>> assert Embedding(name='wiki.ko.align.vec')
    >>> assert Embedding(name='undefined')

    >>> assert Embedding(lang='en', max_n=50000, max_d=50)
    >>> assert Embedding(lang='xx', max_n=50000, max_d=50)
    '''

    def __init__(this, name=None, lang=None, max_n=None, max_d=None, projection='svd_vh.npy', storage_dir=None):
        assert name is not None or lang is not None
        this.max_n = max_n
        this.max_d = max_d
        this.kv = None
        this.annoy_index = None
        this.projection = projection

        # remove locale information from language;
        # e.g. convert 'en-us' into 'en'
        if lang:
            lang = lang.split('-')[0]

        # if the name is not provided,
        # then use the first name in the embeddings list that matches the language
        if name is None:
            for (tmp_name, tmp_lang, tmp_url) in embeddings:
                if lang == tmp_lang:
                    name = tmp_name
                    break
        this.name = str(name)

        # find the matching name in the embeddings list to get the lang/url
        for (tmp_name, tmp_lang, tmp_url) in embeddings:
            if tmp_name == name:
                this.lang = tmp_lang
                this.url = tmp_url
                break

        # internal_name will be used for all file names
        this.internal_name = this.name
        if max_n:
            this.internal_name += f'-max_n={max_n}'
        if max_d:
            this.internal_name += f'-max_d={max_d}'
            if projection:
                this.internal_name += f'-projection={projection}'

        # FIXME:
        # we're currently using the python file's path as a temporary storage dir;
        # this works, but I'm pretty sure python has built-in functionality for a better location
        if storage_dir is None:
            this.storage_dir = os.path.join(os.path.dirname(__file__), 'embeddings')
            os.makedirs(this.storage_dir, exist_ok=True)
        else:
            this.storage_dir = storage_dir


    def load_kv(this, force_reload=False, allow_download=True):
        '''
        Loads an embedding into an instance of gensim's KeyedVector.
        If needed, the embedding is first downloaded.
        '''
        logger.info(f'loading {this.internal_name}')

        # create the internal_name file if it doesn't exist
        internal_name_path = os.path.join(this.storage_dir, this.internal_name)
        if not os.path.isfile(internal_name_path):
            logger.debug(f'{internal_name_path} does not exist')

            # download the unmodified embedding file if it doesn't exist
            name_path = os.path.join(this.storage_dir, this.name)
            if not os.path.isfile(name_path):
                if not allow_download:
                    raise ValueError('embedding {this.name} not downloaded, and allow_download=False')

                # download the url;
                # if the program gets interrupted during download, the resulting file will be corrupted;
                # therefore, we download to a file that ends in .tmp;
                # only once the download has completed do we move it to the correct location;
                logger.info(f'downloading {this.url}')
                logger.info('these urls are large (multiple gigabytes) and so download may take a while')
                urllib.request.urlretrieve(this.url, name_path+'.tmp')
                shutil.move(name_path + '.tmp', name_path)

            # the unmodified embedding file is guaranteed to exist on disk,
            # so create the internal_name file from it if needed
            if this.internal_name != this.name:
                logger.info(f'creating {internal_name_path}')

                # downloaded word vectors come in both gzipped and uncompressed form;
                # we use the url to detect which open function to use
                fopen = gzip.open if this.url[-3:] == '.gz' else open
                with fopen(name_path, 'rt') as fin, open(internal_name_path + '.tmp', 'wt') as fout:

                    # first line of the embedding files contain the number of data points and dimensions
                    header = fin.readline()
                    n_str, d_str = header.split()
                    if this.max_d is None:
                        d = int(d_str)
                    else:
                        d = min(this.max_d, int(d_str))
                        if this.projection:
                            projections_dir = os.path.join(os.path.dirname(__file__), 'embeddings')
                            vh = np.load(os.path.join(projections_dir, 'svd_vh.npy'))
                    if this.max_n is None:
                        n = int(n_str)
                    else:
                        n = min(this.max_n, int(n_str))
                    
                    # output to the file
                    fout.write(f'{n} {d}\n')
                    for i,line in enumerate(fin):
                        if this.max_d is None:
                            fout.write(line)
                        else:
                            if this.projection:
                                line_list = line.split(' ')
                                word = line_list[0]
                                x = np.array([float(d) for d in line_list[1:]])
                                x_proj = vh[:this.max_d] @ x
                                newline = ' '.join([ str(d) for d in x_proj ])
                                fout.write(word+' '+newline+'\n')
                            else:
                                fout.write(' '.join(line.split(' ')[:d+1])+'\n')
                        if i>n:
                            break

                # internal_name file successfully created,
                # so move it to correct location
                shutil.move(internal_name_path + '.tmp', internal_name_path)

        # the internal_name file is guaranteed to exist,
        # so load it into memory
        logger.info(f'loading model from {internal_name_path}')
        this.kv = KeyedVectors.load_word2vec_format(internal_name_path)

    def load_annoy(this, num_trees=10, force_regenerate=False):
        '''
        If an index already exists on disk, then num_trees is ignored.
        If force_regenerate==True, a new index is created and the one on disk is overwritten.
        '''
        logger.info(f'loading annoy index for {this.internal_name}')
        index_path = os.path.join(this.storage_dir, this.internal_name) + '.annoy'
        try:
            if force_regenerate:
                raise OSError
            this.annoy_index = AnnoyIndexer()
            this.annoy_index.load(index_path)
        except OSError:
            logger.info(f'creating new index for {this.internal_name}')
            this.annoy_index = AnnoyIndexer(this.kv, num_trees)
            this.annoy_index.save(index_path)

    def most_similar(this, target, topn, restrict_vocab=None):
        '''
        Returns k-nearest neighbors of target (can be str or numpy array).
        '''
        return this.kv.most_similar([target], topn=topn, restrict_vocab=restrict_vocab, indexer=this.annoy_index)

    def word_frequency(this, word):
        '''
        Computes the frequency of the word using zipf's law.
        '''
        try:
            return zipf_at(this.kv.key_to_index[word], len(this.kv))
        except KeyError:
            return 0


    def make_projectionvector(this, pos_words, neg_words):
        '''
        >>> all(get_test_embedding('en').make_projectionvector(['happy'],['sad'])[0] == -get_test_embedding('en').make_projectionvector(['sad'],['happy'])[0])
        True
        >>> get_test_embedding('en').make_projectionvector(['happy'],['sad'])[1]
        []
        >>> get_test_embedding('en').make_projectionvector(['happytypo'],['sad'])[1]
        ['happytypo']
        >>> get_test_embedding('en').make_projectionvector(['happy'],['sadtypo'])[1]
        ['sadtypo']
        '''
        pos_vectors = [this.kv[word] for word in pos_words if word in this.kv]
        neg_vectors = [this.kv[word] for word in neg_words if word in this.kv]
        unknown_words = [word for word in pos_words+neg_words if word not in this.kv]
        projectionvector = sum(pos_vectors) - sum(neg_vectors)
        projectionvector /= np.linalg.norm(projectionvector)
        return (projectionvector, unknown_words)


def get_top_dimensions(embedding_name, d):
    '''
    '''

    # if we're working with the aligned embeddings,
    # then the top dimensions will be taken from english;
    # this ensures that the embeddings remain aligned;
    # for embeddings not aligned with the English embedding,
    # we just use the specified embedding
    aligned_names = [ name for name, lang, url in _embeddings_fasttext_aligned ]
    if embedding_name in aligned_names:
        embedding_name = 'wiki.en.align.vec'
    embedding = get_embedding('embedding_name')

    # FIXME:
    # we compute all singular vectors here, but we only need the top-d;
    # the computation is already expensive, and this makes it much more expensive when d is small
    u, s, vh = np.linalg.svd(embedding.kv.vectors)

    top_dimensions = vh[:d]

# the code below is used for computing word frequencies;
# see: https://stackoverflow.com/questions/58735585/gensim-any-chance-to-get-word-frequency-in-word2vec-format
from numpy import euler_gamma
from scipy.special import digamma

def digamma_H(s):
    """ If s is complex the result becomes complex. """
    return digamma(s + 1) + euler_gamma

def zipf_at(k_rank, N_total):
    return 1.0 / (k_rank * digamma_H(N_total))
