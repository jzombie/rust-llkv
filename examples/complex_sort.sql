-- Complex sort example for llkv
CREATE TABLE people (
    id INT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    age INT,
    city TEXT
);

-- Insert many rows including NULLs and duplicates for sorting edge cases
INSERT INTO people (id, first_name, last_name, age, city) VALUES (1, 'Ada', 'Lovelace', 36, 'London');
INSERT INTO people (id, first_name, last_name, age, city) VALUES (2, 'Alan', 'Turing', 41, 'Cambridge');
INSERT INTO people (id, first_name, last_name, age, city) VALUES (3, 'Grace', 'Hopper', 85, 'New York');
INSERT INTO people (id, first_name, last_name, age, city) VALUES (4, 'Edsger', 'Dijkstra', 72, NULL);
INSERT INTO people (id, first_name, last_name, age, city) VALUES (5, 'Barbara', 'Liskov', NULL, 'Boston');
INSERT INTO people (id, first_name, last_name, age, city) VALUES (6, 'Donald', 'Knuth', 83, 'Stanford');
INSERT INTO people (id, first_name, last_name, age, city) VALUES (7, NULL, 'Unknown', 28, 'Unknown');
INSERT INTO people (id, first_name, last_name, age, city) VALUES (8, 'Ada', 'Byron', 36, 'London');
INSERT INTO people (id, first_name, last_name, age, city) VALUES (9, 'Alan', 'Turing', 41, 'Princeton');
INSERT INTO people (id, first_name, last_name, age, city) VALUES (10, 'Zoe', 'Zeta', 29, 'Zurich');
INSERT INTO people (id, first_name, last_name, age, city) VALUES (11, 'Ada', 'Lovelace', 36, 'London');
INSERT INTO people (id, first_name, last_name, age, city) VALUES (12, 'Bob', 'Alice', 36, 'Cambridge');
INSERT INTO people (id, first_name, last_name, age, city) VALUES (13, 'Charlie', 'Brown', 36, 'Cambridge');

-- Complex sort: ORDER BY last_name ASC, first_name DESC, age ASC
SELECT id, first_name, last_name, age, city
FROM people
ORDER BY last_name ASC, first_name DESC, age ASC;

-- Aggregate with GROUP BY and ORDER BY on computed key
-- Aggregates over the entire table (no GROUP BY). Compute average client-side if needed.
SELECT
    COUNT(*) AS cnt,
    SUM(age) AS sum_age,
    COUNT(age) AS nonnull_age_count
FROM people;
