import pytest
import csv

# the sys import is needed so that we can import from the current project
import sys
sys.path.append('.')
from postgresql_spacy import lemmatize


################################################################################
# this section defines the test cases and is used when running pytest


def get_golden_tests():
    '''
    Returns the list of test cases
    '''
    tests = []
    golden_test_file = 'tests/golden.csv'
    with open(golden_test_file, 'rt', encoding='utf-8', newline='\n') as f:
        tests = list(csv.DictReader(f, dialect='excel', strict=True))
    return tests


@pytest.mark.parametrize('test', get_golden_tests())
def test__lemmatize(test):
    import ast
    kwargs = ast.literal_eval(test['kwargs'])
    print("kwargs=",kwargs)
    assert lemmatize(test['lang'],test['text'],**kwargs) == test['result']


################################################################################
# this section generates the test cases when the file is called directly


def make_golden_tests(input_file='tests/input.csv', golden_file='tests/golden.csv'):
    '''
    For each lang/text pair in input_file, generate a series of test cases.
    This series of test cases will test every possible combination 
    of keyword arguments to the lemmatize function.
    '''

    # load the input lang/text pairs
    inputs = []
    with open(input_file, 'rt', encoding='utf-8', newline='\n') as f:
        inputs = list(csv.DictReader(f, dialect='excel', strict=True))

    # the kwargss list will contain for each keyword all combinations of True/False
    keywords = [
        'lower_case',
        'remove_special_chars',
        'remove_stopwords',
        'add_positions',
        ]
    kwargss = [{}]
    for keyword in keywords:
        kwargss_new = []
        for kwargs in kwargss:
            kwargss_new.append( {**kwargs, keyword:True})
            kwargss_new.append( {**kwargs, keyword:False})
        kwargss = kwargss_new

    # generate a test for each combination of the lang/text input pairs and entry in kwargss
    tests = []
    for input in inputs:
        for kwargs in kwargss:
            result = lemmatize(input['lang'],input['text'],**kwargs)
            tests.append({
                'lang':input['lang'],
                'text':input['text'],
                'kwargs':str(kwargs),
                'result':result,
                })

    # write the tests to the golden_file
    with open(golden_file, 'wt', encoding='utf-8', newline='\n') as f:
        writer = csv.DictWriter(f, fieldnames=['lang','text','kwargs','result'])
        writer.writeheader()
        for test in tests:
            writer.writerow(test)


# running this file directly will generate new golden tests
if __name__=='__main__':
    make_golden_tests()
