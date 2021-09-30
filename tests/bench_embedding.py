import csv
import pytest

# the sys import is needed so that we can import from the current project
import sys
sys.path.append('.')
from chajda.embeddings import get_test_embedding
from chajda.tsvector import tsvector_to_contextvectors, lemmatize

tsv = lemmatize('en', 'war and peace')
embedding = get_test_embedding('en')

################################################################################
# test cases
################################################################################

@pytest.mark.parametrize('do_fancy', [True, False])
def test__contextvectors_fancy(do_fancy, benchmark):
    benchmark(tsvector_to_contextvectors, embedding, tsv, do_fancy=do_fancy)

