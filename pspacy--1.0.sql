\echo Use "CREATE EXTENSION nvlfunc" to load this file. \quit

CREATE LANGUAGE plpython3u;


CREATE OR REPLACE FUNCTION spacy_lemmatize(
    lang TEXT,
    text TEXT,
    lower_case BOOLEAN DEFAULT TRUE,
    remove_special_chars BOOLEAN DEFAULT TRUE,
    remove_stop_words BOOLEAN DEFAULT TRUE,
    add_positions BOOLEAN DEFAULT TRUE,
    tsquery BOOLEAN DEFAULT FALSE
    )
RETURNS text AS $$
# We load the pspacy library into the SD dictionary if it is not already loaded.
# This dictionary is shared by all invocations of this function within a session,
# and so all invocations of the function will have access to the library without reloading it.
if 'pspacy' not in SD:
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

    # load the library into the SD dictionary
    import pspacy
    SD['pspacy']=pspacy

return SD['pspacy'].lemmatize(
    lang,
    text,
    lower_case,
    remove_special_chars,
    remove_stop_words,
    add_positions
    )
$$ 
LANGUAGE plpython3u
IMMUTABLE
RETURNS NULL ON NULL INPUT;


CREATE OR REPLACE FUNCTION spacy_tsvector(lang TEXT, text TEXT)
RETURNS tsvector AS $$
    SELECT spacy_lemmatize(lang,text) :: tsvector
$$
LANGUAGE SQL 
IMMUTABLE
RETURNS NULL ON NULL INPUT;


CREATE OR REPLACE FUNCTION spacy_tsvector2(lang TEXT, text TEXT)
RETURNS tsvector AS $$
    SELECT spacy_lemmatize(lang,'') :: tsvector
    --SELECT to_tsvector('simple', spacy_lemmatize(lang,text))
$$
LANGUAGE SQL 
IMMUTABLE
RETURNS NULL ON NULL INPUT;


CREATE OR REPLACE FUNCTION spacy_tsquery(lang TEXT, text TEXT)
RETURNS tsquery AS $$
    SELECT to_tsquery('simple', spacy_lemmatize(lang,text, tsquery => TRUE))
$$
LANGUAGE SQL 
IMMUTABLE
RETURNS NULL ON NULL INPUT;
