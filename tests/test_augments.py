import pytest
import os

# the sys import is needed so that we can import from the current project
import sys
sys.path.append('.')
from collections import defaultdict
from chajda.tsquery.augments import load_annoy, load_fasttext, augments_fasttext, destroy_annoy, destroy_fasttext

# NOTE:
# the benchmark tests in this file are structured such that an environment variable can be used to specify which languages to test


################################################################################
# global variables
################################################################################

# default to testing only English if environment variable is misspelled or omitted
params = defaultdict(lambda: ['en'])

# test_level 0: tests only English
params['0'] = ['en']

# test_level 1: tests English, Spanish, Korean, and Chinese
params['1'] = ['en', 'es', 'ko', 'zh']

# test_level 2: Assamese (as), Zazaki (diq), Piedmontese (pms), Walloon (wa)
# NOTE:
# populating an annoy index with vectors from even one  major language, e.g. English, Spanish, Chinese, Korean,
# requires more memory allocation than github actions allows for.
# For this reason, we use test_level 2, which consists of much smaller fasttext language models,
# for github actions testing
params['2'] = ['as', 'diq', 'pms', 'wa']

# test_level 3: tests all languages supported by the spaCy library
params['3'] = ['af', 'ar', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'ga', 'gu', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'kn', 'ko', 'lb', 'lij', 'lt', 'lv', 'ml', 'mr', 'nb', 'ne', 'nl', 'pl', 'pt', 'ro', 'ru', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'ta', 'te', 'th', 'tl', 'tr', 'tt', 'uk', 'ur', 'vi', 'xx', 'yo', 'zh']

# test_level 4: tests all languages supported by the fasttext library
params['4']  = ['af', 'als', 'am', 'an', 'ar', 'arz', 'as', 'ast', 'az', 'azb', 'ba', 'bar', 'bcl', 'be', 'bg', 'bh', 'bn', 'bo', 'bpy', 'br', 'bs', 'ca', 'ce', 'ceb', 'ckb', 'co', 'cs', 'cv', 'cy', 'da', 'de', 'diq', 'dv', 'el', 'eml', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'frr', 'fy', 'ga', 'gd', 'gl', 'gom', 'gu', 'gv', 'he', 'hi', 'hif', 'hr', 'hsb', 'ht', 'hu', 'hy', 'ia', 'id', 'ilo', 'io', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'li', 'lmo', 'lt', 'lv', 'mai', 'mg', 'mhr', 'min', 'mk', 'ml', 'mn', 'mr', 'mrj', 'ms', 'mt', 'mwl', 'my', 'myv', 'mzn', 'nah', 'nap', 'nds', 'ne', 'new', 'nl', 'nn', 'no', 'nso', 'oc', 'or', 'os', 'pa', 'pam', 'pfl', 'pl', 'pms', 'pnb', 'ps', 'pt', 'qu', 'rm', 'ro', 'ru', 'sa', 'sah', 'sc', 'scn', 'sco', 'sd', 'sh', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vec', 'vi', 'vls', 'vo', 'wa', 'war', 'xmf', 'yi', 'yo', 'zea', 'zh']

# n values to be parametrized
nval = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# NOTE:
# for the same reason mentioned above in the test_level 2 comment,
# we use Walloon as the language for tests parametrized by n-values instead of English when pushing to github,
# however, the default language is English
nval_lang = defaultdict(lambda: 'en')

nval_lang['github'] = 'wa'

# all languages will be tested using the word 'google'
test_word = 'google'


################################################################################
# setup/teardown for test_level parametrized tests
################################################################################

@pytest.fixture(scope='module', params=params[os.getenv('test_level')])
def req_param(request):
    return request.param

@pytest.fixture(scope='module')
def run_around_tests(req_param):
    param = req_param

    # setup: preloading fasttext model and annoy index for param language
    load_fasttext(param)

    # given that test_augments_fasttext requires the fasttext model to be downloaded,
    # we specify run_destroy_fasttext=False to prevent load_annoy from removing the fasttext download
    load_annoy(param, run_destroy_fasttext=False)

    # running test
    yield run_around_tests
    
    # teardown
    destroy_annoy(param)
    destroy_fasttext(param)


################################################################################
# setup for benchmark tests parametrized by 'n' value
################################################################################

@pytest.fixture(scope='session', autouse=True)
def setup_teardown_n(lang=nval_lang[os.getenv('nval_lang')]):
    # Will be executed before the first test
    load_fasttext(lang)
    load_annoy(lang, run_destroy_fasttext=False)

    yield setup_teardown_n

    # Will be executed after the last test
    destroy_annoy(lang)
    destroy_fasttext(lang)


################################################################################
# test cases
################################################################################

# test case using only English language, parametrized on 'n' value with annoy
@pytest.mark.parametrize('nval', nval)
def test_augments_n_values_annoy(setup_teardown_n,nval, benchmark):
    benchmark(augments_fasttext, nval_lang[os.getenv('nval_lang')], test_word, n=nval)

# test case using only English language, parametrized on 'n' value with fasttext
@pytest.mark.parametrize('nval', nval)
def test_augments_n_values_fasttext(setup_teardown_n, nval, benchmark):
    benchmark(augments_fasttext, nval_lang[os.getenv('nval_lang')], test_word, n=nval, annoy=False)

# test case using test_level parameters with annoy
def test_augments_annoy(run_around_tests, benchmark, req_param):
    benchmark(augments_fasttext, req_param, test_word)

# test case using test_level parameters with fasttext
def test_augments_fasttext(run_around_tests, benchmark, req_param):
    benchmark(augments_fasttext, req_param, test_word, annoy=False)
