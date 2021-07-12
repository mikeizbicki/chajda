\echo Use "CREATE EXTENSION chajda" to load this file. \quit

CREATE FUNCTION chajda_load()
RETURNS VOID AS $$
# We load the chajda library into the GD dictionary if it is not already loaded.
# This dictionary is shared by all invocations of this function within a session,
# and so all invocations of the function will have access to the library without reloading it.
if 'chajda' not in GD:
    # The sys.argv variable normally holds the command line arguments used to invoke python,
    # but when python is lauchned through plpython3u, 
    # there are no command line arguments,
    # and so sys.argv is set to None.
    # The spacy library and its dependencies, however, 
    # assume that the sys.argv variable exists,
    # and throw an exception when it does not.
    # (Specifically, the korean language parser mecab-ko requires this variable set.)
    # Setting this variable as we do tricks these libraries into thinking that 
    # they were launched from a command called 'chajda' 
    import sys
    sys.argv=['chajda']

    GD['chajda'] = True
$$ 
LANGUAGE plpython3u STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION chajda_lemmatize(
    lang TEXT,
    text TEXT,
    lower_case BOOLEAN DEFAULT TRUE,
    remove_special_chars BOOLEAN DEFAULT TRUE,
    remove_stop_words BOOLEAN DEFAULT TRUE,
    add_positions BOOLEAN DEFAULT TRUE
    )
RETURNS text AS $$
plpy.execute('SELECT chajda_load();');
import chajda.tsvector
return chajda.tsvector.lemmatize(
    lang,
    text,
    add_positions = add_positions,
    config = chajda.tsvector.Config(
        lower_case = lower_case,
        remove_special_chars = remove_special_chars,
        remove_stop_words = remove_stop_words,
        )
    )
$$ 
LANGUAGE plpython3u STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION chajda_tsvector(lang TEXT, text TEXT)
RETURNS tsvector AS $$
    SELECT chajda_lemmatize(lang,text) :: tsvector
$$
LANGUAGE SQL STRICT IMMUTABLE PARALLEL SAFE;


CREATE OR REPLACE FUNCTION tsvector_to_ngrams(tsv tsvector, n INTEGER DEFAULT 3, uniq BOOLEAN DEFAULT TRUE)
RETURNS TEXT[] LANGUAGE plpython3u IMMUTABLE STRICT PARALLEL SAFE 
AS $$
import chajda.tsvector
return list(chajda.tsvector.tsvector_to_ngrams(tsv, n, uniq))
$$;


CREATE OR REPLACE FUNCTION tsvector_to_wordcontext(tsv tsvector, n INTEGER DEFAULT 3, windowsize INTEGER DEFAULT 5)
RETURNS TABLE(a TEXT,b TEXT,c INTEGER) LANGUAGE plpython3u IMMUTABLE STRICT PARALLEL SAFE 
AS $$
import chajda.tsvector
return chajda.tsvector.tsvector_to_wordcontext(tsv, n, windowsize)
$$;
 

CREATE FUNCTION chajda_tsquery(
    lang TEXT,
    text TEXT,
    lower_case BOOLEAN DEFAULT TRUE,
    remove_special_chars BOOLEAN DEFAULT TRUE,
    remove_stop_words BOOLEAN DEFAULT TRUE
    )
RETURNS text AS $$
plpy.execute('SELECT chajda_load();');
import chajda.tsquery
return chajda.tsquery.to_tsquery(
    lang,
    text,
    config = chajda.tsvector.Config(
        lower_case = lower_case,
        remove_special_chars = remove_special_chars,
        remove_stop_words = remove_stop_words,
        )
    )
$$ 
LANGUAGE plpython3u STRICT IMMUTABLE PARALLEL SAFE;

--------------------------------------------------------------------------------
-- simple test cases
--------------------------------------------------------------------------------

do $$
BEGIN
    -- NOTE:
    -- the tests/ folder contains pytest unit tests for the python functions
    -- that are significantly more detailed than the test here;
    -- those tests are designed to catch regressions in the lemmatization;
    -- the purpose of this test is merely to ensure that postgres can connect to the python library
    assert (chajda_tsquery('en', 'united states') = 'unite & state');
    assert (chajda_lemmatize('en', 'this is a test') = 'test:4');
    assert (chajda_tsvector('en', 'this is a test') = 'test:4');
    assert (tsvector_to_ngrams(chajda_tsvector('en', 'united states'), uniq=>False) = ARRAY['unite', 'state', 'unite state']);
    assert (select count(*) from tsvector_to_wordcontext(chajda_tsvector('en', 'the united states is a country'), 2, 1) ) = 6;
END
$$ LANGUAGE plpgsql;
