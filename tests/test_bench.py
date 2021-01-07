import csv
import pytest

# the sys import is needed so that we can import from the current project
import sys
sys.path.append('.')
import pspacy

# load the input lang/text pairs
inputs = []
with open('tests/input.csv', 'rt', encoding='utf-8', newline='\n') as f:
    inputs = list(csv.DictReader(f, dialect='excel', strict=True))

# filter the input languages to
test_langs = None  # ['en','ko','ja','zh']
if test_langs is not None:
    inputs = [input for input in inputs if input['lang'] in test_langs]

# pre-loading all languages ensures that the benchmark times accurately reflect
# the performance of the model's execution time, and not load time
langs = [input['lang'] for input in inputs]
pspacy.load_all_langs(langs)

################################################################################
# test cases
################################################################################


@pytest.mark.parametrize('test', inputs, ids=[input['lang'] for input in inputs])
def test__lemmatize(test, benchmark):
    benchmark(pspacy.lemmatize, test['lang'], test['text'])
