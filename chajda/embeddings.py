import gzip
import math
import numpy as np
import os
import shutil
import tempfile
import urllib
from collections import namedtuple
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

################################################################################
# FancyContext
################################################################################

FancyContext = namedtuple('FancyContext', ['name', 'pos_words', 'neg_words', 'a', 'clip'])
fancycontexts = [
        FancyContext('distance-culture', ['music','film','books','literature','sports','entertainment','culture'], [], 1e-3, 0.6),
        FancyContext('distance-culture-books', ['books','literature'], [], 1e-3, 0.6),
        FancyContext('distance-culture-film', ['movie','film','cinema'], [], 1e-3, 0.6),
        FancyContext('distance-culture-sports', ['baseball','basketball','soccer','football','tennis','sports'], [], 1e-3, 0.6),
        FancyContext('distance-math', ['algebra','topology','geometry'], [], 1e-3, 0.6),
        FancyContext('distance-math-algebra', ['algebra'], [], 1e-3, 0.6),
        FancyContext('distance-math-geometry', ['geometry'], [], 1e-3, 0.6),
        FancyContext('distance-math-topology', ['topology'], [], 1e-3, 0.6),
        FancyContext('distance-military', ['military','army','navy','marines','veteran','soldier'], [], 1e-3, 0.6),
        FancyContext('distance-military-army', ['army','soldier'], [], 1e-3, 0.6),
        FancyContext('distance-military-gun', ['gun','firearm'], [], 1e-3, 0.6),
        FancyContext('distance-military-marines', ['marines'], [], 1e-3, 0.6),
        FancyContext('distance-military-navy', ['navy'], [], 1e-3, 0.6),
        FancyContext('distance-military-terrorism', ['terrorism'], [], 1e-3, 0.6),
        FancyContext('distance-military-wmd', ['wmd'], [], 1e-3, 0.6),
        FancyContext('distance-military-war', ['war','battle','conflict'], [], 1e-3, 0.6),
        FancyContext('distance-other-badwords', ['fuck','shit','crap','nigger','cock'], [], 1e-3, 0.6),
        FancyContext('distance-other-pacifism', ['pacifism','peaceful','diplomatic'], [], 1e-3, 0.6),
        FancyContext('distance-other-protest', ['protest','riot'], [], 1e-3, 0.6),
        FancyContext('distance-other-sexy', ['sexy','hot','nubile','foxy','bodacious','erotic','sensual'], [], 1e-3, 0.6),
        FancyContext('distance-other-sick', ['illness','sick','disease'], [], 1e-3, 0.6),
        FancyContext('distance-policy-crime', ['crime','criminal','jail','police'], [], 1e-3, 0.6),
        FancyContext('distance-policy-education', ['education','school','student','university'], [], 1e-3, 0.6),
        FancyContext('distance-policy-healthcare', ['healthcare','doctors','medicine','nurse'], [], 1e-3, 0.6),
        FancyContext('distance-policy-immigration', ['immigration','immigrant','foreigner'], [], 1e-3, 0.6),
        FancyContext('distance-policy-politics', ['politics','government','policy','government','mayor','governor','president','voting'], [], 1e-3, 0.6),
        FancyContext('distance-policy-voting', ['election','voting','suffrage'], [], 1e-3, 0.6),
        FancyContext('distance-religion', ['christianity','islam','budhism','atheism','god','faith','belief'], [], 1e-3, 0.6),
        FancyContext('distance-religion-atheism', ['atheism','agnosticism','atheist','agnostic'], [], 1e-3, 0.6),
        FancyContext('distance-religion-christianity', ['christianity','church','jesus','christ','catholic','protestant','bible'], [], 1e-3, 0.6),
        FancyContext('distance-religion-islam', ['islam','muslim','muhammad','imam','koran'], [], 1e-3, 0.6),
        FancyContext('distance-religion-judaism', ['jew','temple','hebrew','torah'], [], 1e-3, 0.6),
        FancyContext('distance-religion-pagan', ['pagan','spirituality','astrology','wicca','witch','witchcraft','goddess'], [], 1e-3, 0.6),
        FancyContext('distance-science', ['science','biology','engineering','sociology','psychology','chemistry','physics','medicine'], [], 1e-3, 0.6),
        FancyContext('distance-science-biology', ['biology'], [], 1e-3, 0.6),
        FancyContext('distance-science-chemistry', ['chemistry'], [], 1e-3, 0.6),
        FancyContext('distance-science-engineering', ['engineering'], [], 1e-3, 0.6),
        FancyContext('distance-science-medicine', ['medicine'], [], 1e-3, 0.6),
        FancyContext('distance-science-physics', ['physics'], [], 1e-3, 0.6),
        FancyContext('distance-science-psychology', ['psychology'], [], 1e-3, 0.6),
        FancyContext('distance-science-sociology', ['sociology'], [], 1e-3, 0.6),
        FancyContext('distance-technology-internet', ['email','computer','internet'], [], 1e-3, 0.6),
        FancyContext('distance-technology-nuclear', ['nuclear','fusion','fission'], [], 1e-3, 0.6),
        FancyContext('distance-technology-space', ['spaceship','satellite','rocket'], [], 1e-3, 0.6),
        FancyContext('distance-transportation-airplane', ['airplane','airport','helicopter','flight'], [], 1e-3, 0.6),
        FancyContext('distance-transportation-boat', ['boat','ship','port','harbor'], [], 1e-3, 0.6),
        FancyContext('distance-transportation-car', ['car','truck','road','van','highway'], [], 1e-3, 0.6),
        FancyContext('distance-transportation-pedestrian', ['bicycle','bike','walk','run'], [], 1e-3, 0.6),
        FancyContext('distance-transportation-train', ['train'], [], 1e-3, 0.6),
        FancyContext('projection-emotions-goodness',  ['good', 'better', 'best'], ['bad', 'worse', 'worst'], 1e-3, 0.6),
        FancyContext('projection-emotions-happiness1', ['happy','joyful','ecstatic'], ['sad','melancholy','tragic'], 1e-3, 0.6),
        FancyContext('projection-emotions-happiness2', ['happy'], ['sad'], 1e-3, 0.6),
        FancyContext('projection-emotions-happiness3', ['joyful'], ['sad'], 1e-3, 0.6),
        FancyContext('projection-emotions-happiness4', ['happy'], ['angry'], 1e-3, 0.6),
        FancyContext('projection-emotions-happiness5', ['happy'], ['distressed'], 1e-3, 0.6),
        FancyContext('projection-emotions-happiness6', ['happy'], ['scared'], 1e-3, 0.6),
        FancyContext('projection-emotions-happiness7', ['happy'], ['worried'], 1e-3, 0.6),
        FancyContext('projection-emotions-positive', ['positive'], ['negative'], 1e-3, 0.6),
        FancyContext('projection-emotions-scary', ['scary','alarming','creepy','terrifying','frightening'], ['calming','comforting','comfort','nonthreatening'], 1e-3, 0.6),
        FancyContext('projection-other-advanced', ['advanced'], ['primitive'], 1e-3, 0.6),
        FancyContext('projection-other-big', ['big'], ['small'], 1e-3, 0.6),
        FancyContext('projection-other-clean', ['clean'], ['dirty'], 1e-3, 0.6),
        FancyContext('projection-other-healthy', ['healthy','wellness'], ['unhealthy','sick','illness'], 1e-3, 0.6),
        FancyContext('projection-other-hot', ['hot'], ['cold'], 1e-3, 0.6),
        FancyContext('projection-other-intelligent', ['intelligent','smart'], ['stupid'], 1e-3, 0.6),
        FancyContext('projection-other-life', ['alive'], ['dead'], 1e-3, 0.6),
        FancyContext('projection-other-male', ['male','man','husband','father','king','he'], ['female','woman','wife','mother','queen','she'], 1e-3, 0.6),
        FancyContext('projection-other-military', ['military'], ['civilian'], 1e-3, 0.6),
        FancyContext('projection-other-real', ['real'], ['fake'], 1e-3, 0.6),
        FancyContext('projection-other-rural', ['rural'], ['urban'], 1e-3, 0.6),
        FancyContext('projection-other-science', ['science','biology','chemistry','physics','medicine'], ['religion','christianity','islam','budhism','church'], 1e-3, 0.6),
        FancyContext('projection-other-strong', ['strong'], ['weak'], 1e-3, 0.6),
        FancyContext('projection-threat-dangerous', ['dangerous','danger'], ['safe','safety'], 1e-3, 0.6),
        FancyContext('projection-threat-enemy', ['enemy','foe','adversary'], ['friend','ally'], 1e-3, 0.6),
        FancyContext('projection-threat-lie', ['lie','cheat','fib','trick'], ['truth','accurate','authentic','honesty'], 1e-3, 0.6),
        FancyContext('projection-threat-oppression1', ['oppression'], ['freedom'], 1e-3, 0.6),
        FancyContext('projection-threat-oppression2', ['oppression','tyranny','dictator'], ['freedom','independence','liberty'], 1e-3, 0.6),
        FancyContext('projection-threat-war1', ['war'], ['peace'], 1e-3, 0.6),
        FancyContext('projection-threat-war2', ['war','fight','battle'], ['peace','diplomacy','negotiate'], 1e-3, 0.6),
        FancyContext('projection-threat-war3', ['war','attack','invade'], ['peace','defend','defense'], 1e-3, 0.6),
        ]

################################################################################
# Embedding
################################################################################

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
        embedding.make_fancycontext_projector()
    return _embeddings_loaded[embedding.internal_name]


class Embedding():
    '''
    This class provides a unified interface for working with word embeddings.
    It is not meant to be constructed directly, but instead created using the get_embedding function.

    >>> assert Embedding(lang='en')
    >>> assert Embedding(lang='es').lang == 'es'
    >>> assert Embedding(lang='ko').lang == 'ko'
    >>> assert Embedding(lang='xx').lang == 'en'
    >>> assert Embedding(lang='us').lang == 'en'
    >>> assert Embedding(lang='undefined').lang == 'en'
    >>> assert Embedding(name='wiki.en.align.vec')
    >>> assert Embedding(name='wiki.ko.align.vec')
    >>> assert Embedding(name='undefined')

    >>> Embedding(lang='en', max_n=50000, max_d=50).load_kv()
    >>> Embedding(lang='xx', max_n=50000, max_d=50).load_kv()
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
        # then use the first name in the embeddings list that matches the language;
        # if no match is found, then use the english word embeddings
        this.name = 'wiki.en.align.vec'
        if name is None:
            for (tmp_name, tmp_lang, tmp_url) in embeddings:
                if lang == tmp_lang:
                    this.name = tmp_name
                    break

        # find the matching name in the embeddings list to get the lang/url
        for (tmp_name, tmp_lang, tmp_url) in embeddings:
            if tmp_name == this.name:
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


    def make_projectionvector(this, pos_words, neg_words, normalize=True):
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

        if not normalize:
            projectionvector = sum(pos_vectors) - sum(neg_vectors)
        else:
            pos_vector = sum(pos_vectors)
            pos_vector /= np.linalg.norm(pos_vector)
            neg_vector = sum(neg_vectors)
            neg_vector /= np.linalg.norm(neg_vector)
            projectionvector = pos_vector - neg_vector

        projectionvector /= np.linalg.norm(projectionvector)
        return (projectionvector, unknown_words)


    def make_projector(this, pos_words, neg_words, method='arclen', a=None, clip=None, alpha=3, beta=2):
        '''
        methods = ['projection_nonorm', 'projection_norm', 'arclen']

        These tests show basic usage works.

        >>> get_test_embedding('en').make_projector(['happy'], [], 'projection_nonorm')[0]('happy')
        1.0
        >>> get_test_embedding('en').make_projector(['happy'], [], 'projection_nonorm')[0]('sad')
        0.022573705006315753
        >>> get_test_embedding('en').make_projector(['happy'], ['sad'], 'projection_nonorm')[0]('happy')
        0.5329241
        >>> get_test_embedding('en').make_projector(['happy'], ['sad'], 'projection_norm')[0]('happy')
        0.53292537
        >>> get_test_embedding('en').make_projector(['happy'], ['sad'], 'arclen')[0]('happy')
        1.0
        >>> get_test_embedding('en').make_projector(['happy'], ['sad'], 'arclen')[0]('sad')
        -0.9999998010135039

        These tests cover the `a` and `clip` parameters.

        >>> get_test_embedding('en').make_projector(['happy'], ['sad'], 'arclen', a=1e-3, clip=0.6)[0]('sad')
        -0.9873143002469335
        >>> get_test_embedding('en').make_projector(['happy'], ['sad'], 'arclen', a=1e-3, clip=0.6)[0]('ball')
        0.0

        These tests ensure the unknown_words return value works.
    
        >>> get_test_embedding('en').make_projector(['happy'],['sad'], 'arclen')[1]
        []
        >>> get_test_embedding('en').make_projector(['happy','happytypo'],['sad'], 'arclen')[1]
        ['happytypo']
        >>> get_test_embedding('en').make_projector(['happy'],['sadtypo'], 'arclen')[1]
        ['sadtypo']
        '''

        unknown_words = [word for word in pos_words+neg_words if word not in this.kv]

        pos_vectors = [this.kv[word] for word in pos_words if word in this.kv]
        pos_vector = sum(pos_vectors)
        pos_vector /= np.linalg.norm(pos_vector)

        if len(neg_words) == 0:
            def projector(word):
                vector = this.kv[word]
                dist = arclen(pos_vector, vector)
                score = math.exp(-alpha*dist**beta)
                if score < 1e-2:
                    return 0
                else:
                    return score

        else:
            neg_vectors = [this.kv[word] for word in neg_words if word in this.kv]
            neg_vector = sum(neg_vectors)
            neg_vector /= np.linalg.norm(neg_vector)

            if clip:
                num_clips = 20
                clip_vectors = [ alpha*pos_vector + (1-alpha) * neg_vector for alpha in [i/num_clips for i in range(num_clips+1)] ]

            def mod_result(word, result):
                if clip:
                    vector = this.kv[word]
                    distances = [ np.linalg.norm(clip_vector - vector) for clip_vector in clip_vectors ]
                    if not any([distance < clip for distance in distances]):
                        result = 0.0
                if a:
                    result *= a/(a+this.word_frequency(word))
                return result

            if method == 'projection_nonorm':
                projectionvector = sum(pos_vectors) - sum(neg_vectors)
                projectionvector /= np.linalg.norm(projectionvector)

                def projector(word):
                    vector = this.kv[word]
                    return mod_result(word, np.dot(vector, projectionvector))

            elif method == 'projection_norm':
                projectionvector = pos_vector - neg_vector
                projectionvector /= np.linalg.norm(projectionvector)

                def projector(word):
                    vector = this.kv[word]
                    return mod_result(word, np.dot(vector, projectionvector))

            else:
                e1 = pos_vector/np.linalg.norm(pos_vector)
                v2 = neg_vector - np.dot(e1, neg_vector) * e1
                e2 = v2 / np.linalg.norm(v2)
                mat = np.array([e1,e2])
                zero = (pos_vector+neg_vector)/2
                zero /= np.linalg.norm(zero)
                zero = mat @ zero
                zero /= np.linalg.norm(zero)

                normalizer = arclen([1,0], zero)

                def projector(word):
                    vector = this.kv[word]
                    x = np.dot(mat,vector)
                    x = mat @ vector
                    pos_dist = np.linalg.norm(vector - pos_vector)
                    neg_dist = np.linalg.norm(vector - neg_vector)
                    if pos_dist > neg_dist:
                        projection = -arclen(x,zero)/normalizer
                    else:
                        projection = arclen(x,zero)/normalizer
                    return mod_result(word, projection)

        return (projector, unknown_words)


    def make_fancycontext_projector(embedding):
        '''
        These make good examples for visual inspection:

        get_test_embedding('en').make_fancycontext_projector('music')
        get_test_embedding('en').make_fancycontext_projector('books')
        get_test_embedding('en').make_fancycontext_projector('film')
        get_test_embedding('en').make_fancycontext_projector('happy')
        get_test_embedding('en').make_fancycontext_projector('war')
        get_test_embedding('en').make_fancycontext_projector('euclid')
        get_test_embedding('en').make_fancycontext_projector('topological')


        This test just ensures that no exceptions are thrown.

        len(get_test_embedding('en').fancycontext_projector('topological'))
        80
        '''
        projectors = []
        for fancycontext in fancycontexts:
            projector = embedding.make_projector(fancycontext.pos_words, fancycontext.neg_words, 'arclen', fancycontext.a, 1.0) #fancycontext.clip)
            projectors.append(projector)

        embedding.fancycontext = {}
        def fancycontext_projector(word):
            if word not in embedding.fancycontext:
                embedding.fancycontext[word] = np.array([ projector[0](word) for projector in projectors])
            return embedding.fancycontext[word]
        embedding.fancycontext_projector = fancycontext_projector


def arclen(x,y):
    '''

    All doctests equal pi/2.

    >>> arclen(np.array([1,0]), np.array([0,1]))
    1.5707963267948966
    >>> arclen(np.array([2,0]), np.array([0,1]))
    1.5707963267948966
    >>> arclen(np.array([1,0]), np.array([0,2]))
    1.5707963267948966
    '''
    # numerical instability can cause cos_sim to be slightly above 1,
    # and the min/max ensures that we don't exceed this range
    cos_sim = max(min(np.dot(x/np.linalg.norm(x),y/np.linalg.norm(y)), 1), -1)
    return math.acos(cos_sim)


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
