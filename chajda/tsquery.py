'''
'''

import copy
from lark import Lark, Transformer, Token, Tree, Discard
from chajda.tsvector import lemmatize, Config

# defines the grammar of our queries
grammar = Lark(r"""
    ?exp: term (sym_or term)*
    ?term: factor (sym_and? factor)*
    ?factor: sym_not factor | filter | str | "(" exp ")"
    
    sym_not: /NOT\b/i | "!"
    sym_and: /AND\b/i | "&"+
    sym_or: /OR\b/i | "|"+

    str: ESCAPED_STRING | STRING

    filter: str_raw ":" str_raw
    str_raw: ESCAPED_STRING | STRING

    STRING: /(?!(NOT|AND|OR)\b)[^!&|\s:"()]+/i

    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS
    """, start='exp')


def to_tsquery(lang, query, augment_with=None, config=Config()):
    '''
    A simple wrapper around the parse function that returns only the tsquery.
    '''
    return parse(lang, query, augment_with, config)['tsquery']


def parse(lang, query, augment_with=None, config=Config()):
    '''
    IMPLEMENTATION DETAILS:
    Parsing is done using the Lark library in a two step process.
    First, a grammar defines how to convert an input string into an AST.
    Then, a series of Transformer classes are used to further process the AST.
    If you're trying to understand the details of how these parsing steps work,
    it's probably a good idea to first go through Lark's JSON parsing tutorial:
    https://lark-parser.readthedocs.io/en/latest/json_tutorial.html

    >>> to_tsquery('xx', 'word_or word_and word_not or_word and_word not_word _and_ _or_ _not_')
    'wordor & wordand & wordnot & orword & andword & notword & and & or & not'

    >>> to_tsquery('xx', 'wordor or wordand and not wordnot')
    'wordor | (wordand & !wordnot)'

    >>> to_tsquery('xx', 'keyword1 key1:value1 keyword2 key2:value2')
    'keyword1 & keyword2'

    >>> to_tsquery('xx', 'not not not ((((t1))) t2 t3) (t4 t5 (t6 t7))')
    '!(t1 & t2 & t3) & t4 & t5 & t6 & t7'

    >>> to_tsquery('en', 'baby')
    'baby'

    >>> to_tsquery('en', '"baby"')
    'baby'

    >>> to_tsquery('en', '"baby boy"')
    'baby <1> boy'

    >>> to_tsquery('en', 'not "george washington" "abraham lincoln" and ("washington d.c.")')
    '!(george <1> washington) & (abraham <1> lincoln) & (washington <1> dc)'

    >>> to_tsquery('en', '"the United States of America" and "the democratic people\\\'s republic of korea"')
    '(unite <1> state <2> america) & (democratic <1> people <1> republic <2> korea)'
    '''

    # parse the query into an AST
    tree = grammar.parse(query)

    # apply the transformation stages
    tree = _stage_simplify(tree)
    tree = _stage_lemmatize(tree, lang, config)
    tree, augments = _stage_augment(tree, lang, augment_with, config)
    tsquery = _stage_tsquery(tree, bool(augment_with))
    filtertree = _stage_filtertree(tree)

    # return the dictionary of results
    return {
        'tsquery': tsquery,
        'augments': augments,
        'filtertree': filtertree,
        }


################################################################################
# compilation stages
################################################################################


def _stage_simplify(tree):
    '''
    Simplify the AST by removing:
    1) redundant not expressions (done in the factor method)
    2) redundant and expressions (done in the term method)

    FIXME:
    There are lots of other useful simplifications that could be added.

    ---

    Each of these features are tested in a two-stage doctest below.
    The first test provides an "integration test" with the `to_tsquery` function, and is more human readable.
    The second test operates directly on the AST.
    It is more properly a unit test of the function,
    but also much harder for a human to read and understand what it's doing.

    The test for redundant not expressions:

    >>> to_tsquery('xx','not not not not term')
    'term'
    >>> _stage_simplify(Tree('factor', [Tree('sym_not', [Token('__ANON_0', 'not')]), Tree('factor', [Tree('sym_not', [Token('__ANON_0', 'not')]), Tree('factor', [Tree('sym_not', [Token('__ANON_0', 'not')]), Tree('factor', [Tree('sym_not', [Token('__ANON_0', 'not')]), Tree('str', [Token('STRING', 'term')])])])])]))
    Tree('str', [Token('STRING', 'term')])

    The tests for redundant and parens:

    >>> to_tsquery('xx','a & (b & c)')
    'a & b & c'
    >>> _stage_simplify(Tree('term', [Tree('str', [Token('STRING', 'a')]), Tree('sym_and', []), Tree('term', [Tree('str', [Token('STRING', 'b')]), Tree('sym_and', []), Tree('str', [Token('STRING', 'c')])])]))
    Tree('term', [Tree('str', [Token('STRING', 'a')]), Tree('sym_and', []), Tree('str', [Token('STRING', 'b')]), Tree('sym_and', []), Tree('str', [Token('STRING', 'c')])])
    '''
    class Transformer_simplify(Transformer):

        def factor(self, t):
            if type(t[0]) == Tree and t[0].data == 'sym_not' and type(t[1])==Tree:
                t2 = t[1].children
                if type(t2[0]) == Tree and t2[0].data == 'sym_not':
                    return t2[1]
            return Tree('factor', t)

        def term(self, t):
            t2 = []
            has_subterm = False
            for child in t:
                if type(child) == Tree and child.data == 'term':
                    has_subterm = True
                    t2.extend(copy.deepcopy(child.children))
                else:
                    t2.append(copy.deepcopy(child))
            if not has_subterm:
                return Tree('term',t)
            return Tree('term',t2)

    return Transformer_simplify().transform(tree)


def _stage_augment(tree, lang, augment_with, config):
    '''

    NOTE:
    The functions in the chajda.tsvector.augments module have integration tests.
    '''
    class Transformer_lemmatize(Transformer):

        def __init__(self):
            self.augments = {}

        def str(self, t):
            term = str(t[0])

            # do nothing if no augment_with parameter
            if augment_with is None:
                return Tree('str', [term])

            # else compute the augments
            else:
                augments = augment_with(lang, term, config)
                augments_lemmatized = []
                for augment in augments:
                    try:
                        augments_lemmatized.append(_lemmatize_rawterm(lang, augment, config))
                    except Discard:
                        pass
                return Tree('augment', [term]+augments_lemmatized)


        def _rawterm_to_tsquery(self, rawterm, weight=None):
            term = lemmatize(
                lang,
                rawterm,
                add_positions=True,
                config=config,
                )
            terms = term.split()
            if len(terms) < 1:
                return ''
            if weight:
                weight_term = ':'+weight
            else:
                weight_term = ''
            ret = terms[0].split(':')[0]+weight_term
            for i in range(1,len(terms)):
                t0,p0 = terms[i-1].split(':')
                t1,p1 = terms[i  ].split(':')
                diff = int(p1)-int(p0)
                ret += f' <{diff}> {t1}{weight_term}'
            if len(terms) > 1:
                ret = f'({ret})'
            return ret

    lemmatizer = Transformer_lemmatize()
    lemmatized_tree = lemmatizer.transform(tree)
    if type(lemmatized_tree) == str:
        lemmatized_tree = Tree('exp', [lemmatized_tree])
    augments = lemmatizer.augments

    return (lemmatized_tree, augments)


def _stage_lemmatize(tree, lang, config):
    '''
    Apply the chajda.lemmatize function to all terminal nodes in the AST.
    The str nodes represent query terms, and they are lemmatized using spacy.
    Terms in quotes are processed to maintain the exact positioning when converted into a tsvector.
    The str_raw nodes are used inside of the filter terms,
    and these are maintained as-is because we do not want the filter terms to have language-sensitive transformations.
    '''
    class Transformer_lemmatize(Transformer):

        def str_raw(self, t):
            rawterm = str(t[0].value)
            if rawterm[0] == '"' and rawterm[-1] == '"':
                rawterm = rawterm[1:-1]
            return rawterm

        def str(self, t):
            rawterm = str(t[0].value)
            return _lemmatize_rawterm(lang, rawterm, config)

    return Transformer_lemmatize(config).transform(tree)


def _stage_tsquery(tree, weighted):
    '''
    Flattens the AST into a string that can be cast into postgresql's `tsvector` type.
    For example, we replace all `sym_and` nodes with the `&` character and ensure that parentheses get added appropriately.

    >>> parse('en', 'this is a test united states')['tsquery']
    'test & unite & state'

    >>> parse('en', 'this is a test "united states"')['tsquery']
    'test & (unite <1> state)'

    >>> parse('en', '"the a test":a')['tsquery']
    ''
    '''
    class Transformer_tsvector(Transformer):

        def factor(self, t):
            if len(t)==2 and t[0]=='!':
                return '!' + t[1]
            else:
                return self.exp(t)

        def exp(self, t):
            return '(' + ' '.join(t) + ')'

        def term(self, t):
            t2 = filter(lambda x: len(x)>0, t)
            if len(t) > 1:
                return '(' + ' & '.join(t2) + ')'
            else:
                return t[0]

        def filter(self, t):
            return ''

        def str(self, t):
            return t[0]

        def phrase(self, t):
            if weighted:
                weight_term = ':A'
            else:
                weight_term = ''
            ret = t[0] + weight_term
            num_underscores = 0
            for child in t[1:]:
                if child == '_':
                    num_underscores += 1
                else:
                    ret += f' <{num_underscores+1}> {child}{weight_term}'
                    num_underscores = 0
            if len(t) > 1:
                ret = '(' + ret + ')'
            return ret

        def augment(self, t):
            rawterm = t[0] + ':A'
            augments = [ child + ':B' for child in t[1:] ]

            if len(rawterm) == 0:
                return rawterm

            else:
                return '(' + rawterm + ' | ' + ' | '.join(augments) + ')'

        def sym_or(self, t):
            return '|'

        def sym_and(self, t):
            return ''

        def sym_not(self, t):
            return '!'

    tsvector = Transformer_tsvector().transform(tree)
    
    # if the result is wrapped in unnecessary parentheses,
    # remove them before returning
    if len(tsvector) > 0 and tsvector[0] == '(' and tsvector[-1] == ')':
        tsvector = tsvector[1:-1]
    return tsvector


def _stage_filtertree(tree):
    '''
    This function extracts all of the filter objects from the AST and returns a tree representing these filters.

    >>> parse('en', '"the a test":a b c')['filtertree']
    Tree('and', [Tree('filter', ['the a test', 'a'])])

    >>> parse('en', '"the a test":a')['filtertree']
    Tree('filter', ['the a test', 'a'])

    >>> parse('en', 'k1:v1 k2:v2 (k3:v3 or k4:v4) !k5:v5')['filtertree']
    Tree('and', [Tree('filter', ['k1', 'v1']), Tree('filter', ['k2', 'v2']), Tree('or', [Tree('filter', ['k3', 'v3']), Tree('filter', ['k4', 'v4'])]), Tree('not', Tree('filter', ['k5', 'v5']))])

    >>> parse('en', 't1 t2 & t3 t4 and t5')['filtertree']
    Tree('and', [])
    '''
    class Transformer_filtertree(Transformer):

        def exp(self, t):
            return Tree('or', t)

        def term(self, t):
            return Tree('and', t)

        def factor(self, t):
            if len(t) == 2:
                return Tree('not', t[1])
            else:
                return Tree('factor', t)

        def sym_and(self, t):
            raise Discard

        def sym_or(self, t):
            raise Discard

        def str(self, t):
            raise Discard

    try:
        return Transformer_filtertree().transform(tree)
    
    # the filtertree transformation discards lots of nodes in the tree;
    # if there are no filter commands in the input, this will result in an empty tree;
    # Lark does not catch the Discard error in this event, so we must explicitly do so;
    # we return None in this event to indicate there are no filters
    except Discard:
        return None


################################################################################
# helper functions
################################################################################


def _lemmatize_rawterm(lang, rawterm, config):
    '''
    A helper function for calling spacy to lemmatize a given rawterm.
    This function is called internally in several different stages

    >>> _lemmatize_rawterm('en', 'United', Config())
    Tree('str', ['unite'])

    >>> _lemmatize_rawterm('en', 'the United States of America', Config())
    Tree('phrase', ['unite', 'state', '_', 'america'])
    '''
    # if there were quotations around the original input rawterm, remove them
    if rawterm[0] == '"' and rawterm[-1] == '"':
        rawterm = rawterm[1:-1]

    # convert the rawterm into a tsquery
    terms_str = lemmatize(lang, rawterm, add_positions=True, config=config)
    terms = terms_str.split()

    # if rawterm only contain stop words,
    # then they may be removed by lemmatization,
    # resulting in an empty set of terms;
    # we return an empty list,
    # and subsequent parsing steps will need to handle this case specially
    if len(terms) < 1:
        raise Discard

    # if there's exactly one term,
    # then remove the position information and return the term
    elif len(terms) == 1:
        ret = terms[0].split(':')[0]
        return Tree('str', [ret])

    # if there's more than one term,
    # we create a new node in the tree
    else:
        children = []
        children.append(terms[0].split(':')[0])
        for i in range(1,len(terms)):
            t0,p0 = terms[i-1].split(':')
            t1,p1 = terms[i  ].split(':')
            diff = int(p1)-int(p0)
            children.extend(['_']*(diff-1))
            children.append(t1)
        return Tree('phrase', children)
