CREATE SEQUENCE querytest_testid_seq;  -- Crea la sequenza per l'incremento automatico

CREATE TABLE querytest (
    testid INTEGER PRIMARY KEY NOT NULL DEFAULT nextval('querytest_testid_seq'),
    question TEXT NOT NULL,
    querySQL TEXT, -- può essere NULL nel caso in cui non venga generata la querySQL da parte del modello
    response_execute_query TEXT NOT NULL,
    model TEXT NOT NULL
);
