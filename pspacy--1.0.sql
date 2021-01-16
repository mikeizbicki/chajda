\echo Use "CREATE EXTENSION nvlfunc" to load this file. \quit

CREATE OR REPLACE LANGUAGE plpython3u;


CREATE FUNCTION spacy_load()
RETURNS VOID AS $$
# We load the pspacy library into the GD dictionary if it is not already loaded.
# This dictionary is shared by all invocations of this function within a session,
# and so all invocations of the function will have access to the library without reloading it.
if 'pspacy' not in GD:
    # The sys.argv variable normally holds the command line arguments used to invoke python,
    # but when python is lauchned through plpython3u, 
    # there are no command line arguments,
    # and so sys.argv is set to None.
    # The spacy library and its dependencies, however, 
    # assume that the sys.argv variable exists,
    # and throw an exception when it does not.
    # Setting this variable as we do tricks these libraries into thinking that 
    # they were launched from a command called 'pspacy' 
    import sys
    sys.argv=['pspacy']

    # we must add postgres's extension folder into the path in order to import the module
    # FIXME:
    # using the subprocess module to get the $SHAREDIR variable is a hack
    # that won't work on some systems (esp. windows) where pg_config isn't in the path,
    # but I can't find a way to get this folder directly via sql/plpython
    import subprocess
    import os
    sharedir = subprocess.check_output(['pg_config','--sharedir']).decode('utf-8').strip()
    path = os.path.join(sharedir,'extension')
    sys.path.append(path)

    # load the library into the GD dictionary
    import pspacy
    GD['pspacy']=pspacy
$$ 
LANGUAGE plpython3u STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION spacy_lemmatize(
    lang TEXT,
    text TEXT,
    lower_case BOOLEAN DEFAULT TRUE,
    remove_special_chars BOOLEAN DEFAULT TRUE,
    remove_stop_words BOOLEAN DEFAULT TRUE,
    add_positions BOOLEAN DEFAULT TRUE
    )
RETURNS text AS $$
plpy.execute('SELECT spacy_load();');
return GD['pspacy'].lemmatize(
    lang,
    text,
    lower_case,
    remove_special_chars,
    remove_stop_words,
    add_positions
    )
$$ 
LANGUAGE plpython3u STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION spacy_lemmatize_query(
    lang TEXT,
    text TEXT,
    lower_case BOOLEAN DEFAULT TRUE,
    remove_special_chars BOOLEAN DEFAULT TRUE,
    remove_stop_words BOOLEAN DEFAULT TRUE
    )
RETURNS text AS $$
plpy.execute('SELECT spacy_load();');
return GD['pspacy'].lemmatize_query(
    lang,
    text,
    lower_case,
    remove_special_chars,
    remove_stop_words
    )
$$ 
LANGUAGE plpython3u STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION spacy_tsvector(lang TEXT, text TEXT)
RETURNS tsvector AS $$
    SELECT spacy_lemmatize(lang,text) :: tsvector
$$
LANGUAGE SQL STRICT IMMUTABLE PARALLEL SAFE;


CREATE FUNCTION spacy_tsquery(lang TEXT, text TEXT)
RETURNS tsquery AS $$
    SELECT to_tsquery('simple', spacy_lemmatize_query(lang,text))
$$
LANGUAGE SQL STRICT IMMUTABLE PARALLEL SAFE;
