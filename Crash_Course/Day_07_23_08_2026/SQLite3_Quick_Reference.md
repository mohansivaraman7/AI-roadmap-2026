# SQLite3 Quick Reference (same concepts, SQLite syntax)

> Run via terminal: `sqlite3 company.db` — SQLite is file-based, so there's no `CREATE DATABASE`/`USE`. Opening a file (or `:memory:`) IS your database.

---

## 1. Starting SQLite3
```bash
sqlite3 company.db          # opens/creates a db file
.tables                     # list tables
.schema employees           # show table structure
.headers on                 # show column headers in output
.mode column                # pretty print
.exit                       # quit
```

---

## 2. DDL — Create / Alter / Drop

```sql
CREATE TABLE departments (
    dept_id INTEGER PRIMARY KEY,
    dept_name TEXT
);

CREATE TABLE employees (
    emp_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER CHECK (age >= 18),
    salary REAL DEFAULT 0,
    dept_id INTEGER,
    email TEXT UNIQUE,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);

-- Alter (SQLite supports limited ALTER)
ALTER TABLE employees ADD COLUMN phone TEXT;
ALTER TABLE employees RENAME COLUMN name TO full_name;
ALTER TABLE employees RENAME TO staff;
-- NOTE: no MODIFY COLUMN or DROP COLUMN before SQLite 3.35;
-- newer versions support: ALTER TABLE employees DROP COLUMN phone;

DROP TABLE employees;
DELETE FROM employees;      -- SQLite has no TRUNCATE, use DELETE (or VACUUM after)
```

**Key SQLite type notes:**
- Uses **dynamic typing** (type affinity): `INTEGER`, `TEXT`, `REAL`, `BLOB`, `NUMERIC`
- No strict `VARCHAR(50)` enforcement — `TEXT` works for all strings
- `AUTOINCREMENT` only valid on `INTEGER PRIMARY KEY`
- Foreign keys are **off by default** — enable with:
```sql
PRAGMA foreign_keys = ON;
```

---

## 3. DML — Insert / Update / Delete
*(Identical to standard SQL)*
```sql
INSERT INTO departments (dept_id, dept_name) VALUES (1, 'IT'), (2, 'HR');

INSERT INTO employees (name, age, salary, dept_id, email)
VALUES ('Ravi', 25, 45000, 1, 'ravi@mail.com');

UPDATE employees SET salary = 65000 WHERE emp_id = 1;

DELETE FROM employees WHERE emp_id = 2;
```

---

## 4. DQL — SELECT, WHERE, ORDER BY
*(Same syntax as MySQL/standard SQL)*
```sql
SELECT * FROM employees;
SELECT name, salary FROM employees WHERE salary > 50000;
SELECT * FROM employees WHERE name LIKE 'A%';
SELECT * FROM employees ORDER BY salary DESC LIMIT 5;
```

---

## 5. Aggregates & GROUP BY
*(Same as standard SQL)*
```sql
SELECT dept_id, COUNT(*), AVG(salary)
FROM employees
GROUP BY dept_id
HAVING AVG(salary) > 50000;
```

---

## 6. Joins

```sql
SELECT e.name, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id;      -- INNER JOIN (default)

SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;

-- RIGHT JOIN and FULL JOIN: supported only in SQLite 3.39+ (2022 onward)
SELECT e.name, d.dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id;

-- If your SQLite version is older, simulate RIGHT JOIN by swapping table order with LEFT JOIN
```
Check your version: `SELECT sqlite_version();`

---

## 7. Subqueries & Set Ops
*(Same as standard SQL — fully supported)*
```sql
SELECT name FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

SELECT name FROM employees
UNION
SELECT dept_name FROM departments;
-- INTERSECT and EXCEPT also supported (not MINUS)
```

---

## 8. Date & String Functions (SQLite-specific!)

```sql
-- Dates: SQLite has NO native DATE type — stored as TEXT/REAL/INTEGER
SELECT date('now');                       -- current date
SELECT datetime('now');                   -- current date+time
SELECT strftime('%Y', hire_date) FROM employees;   -- extract year
SELECT date(hire_date, '+1 year') FROM employees;  -- add interval

-- Strings
SELECT upper(name), lower(name), length(name) FROM employees;
SELECT name || ' - ' || dept_id FROM employees;    -- concatenation uses ||, not CONCAT()
SELECT substr(name, 1, 3) FROM employees;
SELECT trim('  hello  ');
SELECT replace(name, 'a', 'A') FROM employees;
```

---

## 9. Conditional Logic
*(Same as standard SQL)*
```sql
SELECT name,
    CASE
        WHEN salary > 60000 THEN 'High'
        ELSE 'Low'
    END AS grade
FROM employees;

SELECT name, IFNULL(email, 'N/A') FROM employees;
SELECT name, COALESCE(email, phone, 'No Contact') FROM employees;
```

---

## 10. Views & Indexes
*(Same as standard SQL)*
```sql
CREATE VIEW high_earners AS
SELECT name, salary FROM employees WHERE salary > 60000;

CREATE INDEX idx_name ON employees(name);
```

---

## 11. Transactions
```sql
BEGIN TRANSACTION;
UPDATE employees SET salary = salary + 1000 WHERE dept_id = 1;
SAVEPOINT sp1;
UPDATE employees SET salary = salary - 500 WHERE dept_id = 2;
ROLLBACK TO sp1;
COMMIT;
```

---

## 12. What SQLite Does NOT Have
| Feature | Status |
|---|---|
| `CREATE DATABASE` / `USE` | ❌ N/A — file = database |
| `TRUNCATE` | ❌ use `DELETE FROM table` |
| `GRANT` / `REVOKE` (DCL) | ❌ no user permission system |
| Stored procedures | ❌ not supported |
| `MODIFY COLUMN` | ❌ limited ALTER support |
| Strict data types | ❌ uses type affinity instead |
| `RIGHT/FULL JOIN` | ⚠️ only in 3.39+ |

---

## Quick Test Flow in Terminal
```bash
sqlite3 test.db
.headers on
.mode column
CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT);
INSERT INTO t(name) VALUES ('Alice'), ('Bob');
SELECT * FROM t;
.exit
```
