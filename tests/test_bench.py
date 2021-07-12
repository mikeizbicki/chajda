import csv
import pytest

# the sys import is needed so that we can import from the current project
import sys
sys.path.append('.')
from chajda.tsvector import lemmatize, load_lang, destroy_lang

# load the input lang/text pairs
inputs = []
with open('tests/input.csv', 'rt', encoding='utf-8', newline='\n') as f:
    inputs = list(csv.DictReader(f, dialect='excel', strict=True))

# filter the input languages to
test_langs = None  # ['en','ko','ja','zh']
if test_langs is not None:
    inputs = [input for input in inputs if input['lang'] in test_langs]

langs = [input['lang'] for input in inputs]

@pytest.fixture(scope='function', autouse=True)
def setup_and_teardown(test):
    # setup
    load_lang(test['lang'])

    # run test
    yield setup_and_teardown

    # teardown
    destroy_lang(test['lang'])


################################################################################
# test cases
################################################################################


@pytest.mark.parametrize('test', inputs, ids=[input['lang'] for input in inputs])
def test__lemmatize(test, benchmark):
    benchmark(lemmatize, test['lang'], test['text'])
