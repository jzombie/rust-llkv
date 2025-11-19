Copying tests from: https://github.com/duckdb/duckdb/tree/main/test/sql

For more information regarding DuckDB's sqllogictest integration: https://duckdb.org/docs/stable/dev/sqllogictest/intro.html

Note: Generally most tests are run unmodified, but occasionally tiny modifications have been made to enforce sort ordering, etc. (such as in constraints/foreignkey/test_fk_temporary.slt) This is highly discouraged in general and should only be used if there is no underlying SQL standard which enforces a particular default.
