# Complete SQL Guide — Everything to Cover

## 1. SQL Basics — What is SQL?
SQL (Structured Query Language) is used to communicate with relational databases. Commands are grouped into categories:

| Category | Full Form | Purpose | Commands |
|---|---|---|---|
| DDL | Data Definition Language | Define/modify structure | CREATE, ALTER, DROP, TRUNCATE |
| DML | Data Manipulation Language | Modify data | INSERT, UPDATE, DELETE |
| DQL | Data Query Language | Fetch data | SELECT |
| DCL | Data Control Language | Permissions | GRANT, REVOKE |
| TCL | Transaction Control Language | Manage transactions | COMMIT, ROLLBACK, SAVEPOINT |

---

## 2. DDL — Data Definition Language

```sql
-- Create a database
CREATE DATABASE company;
USE company;

-- Create a table
CREATE TABLE employees (
    emp_id INT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    age INT CHECK (age >= 18),
    salary DECIMAL(10,2) DEFAULT 0,
    dept_id INT,
    email VARCHAR(100) UNIQUE
);

-- Alter table
ALTER TABLE employees ADD COLUMN phone VARCHAR(15);
ALTER TABLE employees DROP COLUMN phone;
ALTER TABLE employees MODIFY COLUMN name VARCHAR(100);
ALTER TABLE employees RENAME COLUMN name TO full_name;

-- Drop / Truncate / Rename
DROP TABLE employees;          -- deletes table + structure
TRUNCATE TABLE employees;      -- deletes all rows, keeps structure
RENAME TABLE employees TO staff;
```

---

## 3. Constraints
```sql
NOT NULL       -- value must be provided
UNIQUE         -- no duplicate values
PRIMARY KEY    -- NOT NULL + UNIQUE, identifies row
FOREIGN KEY    -- links to another table's primary key
CHECK          -- validates condition
DEFAULT        -- default value if none given
AUTO_INCREMENT -- auto-generates numeric values (MySQL)
```
Example with a foreign key:
```sql
CREATE TABLE departments (
    dept_id INT PRIMARY KEY,
    dept_name VARCHAR(50)
);

CREATE TABLE employees (
    emp_id INT PRIMARY KEY,
    name VARCHAR(50),
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);
```

---

## 4. DML — Data Manipulation Language

```sql
-- Insert
INSERT INTO employees (emp_id, name, age, salary, dept_id)
VALUES (1, 'Ravi', 25, 45000, 1);

INSERT INTO employees VALUES
(2, 'Anita', 30, 55000, 2),
(3, 'John', 28, 60000, 1);

-- Update
UPDATE employees
SET salary = 65000
WHERE emp_id = 3;

-- Delete
DELETE FROM employees WHERE emp_id = 2;
```

---

## 5. DQL — SELECT and Filtering

```sql
SELECT * FROM employees;
SELECT name, salary FROM employees;
SELECT DISTINCT dept_id FROM employees;

-- WHERE clause
SELECT * FROM employees WHERE salary > 50000;
SELECT * FROM employees WHERE age BETWEEN 25 AND 30;
SELECT * FROM employees WHERE dept_id IN (1, 2);
SELECT * FROM employees WHERE name LIKE 'A%';   -- starts with A
SELECT * FROM employees WHERE name LIKE '%i';   -- ends with i
SELECT * FROM employees WHERE email IS NULL;
SELECT * FROM employees WHERE age > 25 AND salary > 40000;
SELECT * FROM employees WHERE age < 25 OR salary > 60000;
SELECT * FROM employees WHERE NOT dept_id = 1;

-- Sorting & limiting
SELECT * FROM employees ORDER BY salary DESC;
SELECT * FROM employees ORDER BY dept_id ASC, salary DESC;
SELECT * FROM employees LIMIT 5;
SELECT * FROM employees LIMIT 5 OFFSET 10;
```

---

## 6. Aggregate Functions & Grouping

```sql
SELECT COUNT(*) FROM employees;
SELECT SUM(salary) FROM employees;
SELECT AVG(salary) FROM employees;
SELECT MIN(salary), MAX(salary) FROM employees;

-- GROUP BY
SELECT dept_id, COUNT(*) AS total_emp, AVG(salary) AS avg_sal
FROM employees
GROUP BY dept_id;

-- HAVING (filter groups, WHERE filters rows before grouping)
SELECT dept_id, AVG(salary) AS avg_sal
FROM employees
GROUP BY dept_id
HAVING AVG(salary) > 50000;
```
**Order of execution to remember:**
`FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT`

---

## 7. Joins

```sql
-- INNER JOIN: only matching rows
SELECT e.name, d.dept_name
FROM employees e
INNER JOIN departments d ON e.dept_id = d.dept_id;

-- LEFT JOIN: all from left table + matches from right
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;

-- RIGHT JOIN: all from right table + matches from left
SELECT e.name, d.dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id;

-- FULL OUTER JOIN: all rows from both (not in MySQL directly — use UNION)
SELECT e.name, d.dept_name FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id
UNION
SELECT e.name, d.dept_name FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id;

-- SELF JOIN
SELECT a.name AS emp, b.name AS manager
FROM employees a
JOIN employees b ON a.manager_id = b.emp_id;

-- CROSS JOIN: cartesian product
SELECT * FROM employees CROSS JOIN departments;
```

---

## 8. Subqueries

```sql
-- Subquery in WHERE
SELECT name FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Subquery with IN
SELECT name FROM employees
WHERE dept_id IN (SELECT dept_id FROM departments WHERE dept_name = 'IT');

-- Correlated subquery
SELECT name FROM employees e
WHERE salary > (SELECT AVG(salary) FROM employees WHERE dept_id = e.dept_id);

-- Subquery in FROM (derived table)
SELECT dept_id, avg_sal FROM
(SELECT dept_id, AVG(salary) AS avg_sal FROM employees GROUP BY dept_id) AS temp;
```

---

## 9. Set Operations

```sql
SELECT name FROM employees
UNION
SELECT name FROM managers;        -- combines, removes duplicates

SELECT name FROM employees
UNION ALL
SELECT name FROM managers;        -- keeps duplicates

SELECT name FROM employees
INTERSECT
SELECT name FROM managers;        -- common rows

SELECT name FROM employees
EXCEPT
SELECT name FROM managers;        -- in first, not in second
```

---

## 10. String, Date, and Math Functions

```sql
-- String functions
SELECT UPPER(name), LOWER(name), LENGTH(name) FROM employees;
SELECT CONCAT(name, ' - ', dept_id) FROM employees;
SELECT SUBSTRING(name, 1, 3) FROM employees;
SELECT TRIM('  hello  ');
SELECT REPLACE(name, 'a', 'A') FROM employees;

-- Date functions
SELECT NOW();
SELECT CURDATE();
SELECT DATEDIFF('2026-01-01', '2025-01-01');
SELECT YEAR(hire_date), MONTH(hire_date) FROM employees;
SELECT DATE_ADD(hire_date, INTERVAL 1 YEAR) FROM employees;

-- Math functions
SELECT ROUND(salary, 2), CEIL(salary), FLOOR(salary), ABS(-5) FROM employees;
```

---

## 11. Conditional Logic

```sql
SELECT name,
    CASE
        WHEN salary > 60000 THEN 'High'
        WHEN salary > 40000 THEN 'Medium'
        ELSE 'Low'
    END AS salary_grade
FROM employees;

SELECT name, IFNULL(email, 'Not Provided') FROM employees;
SELECT name, COALESCE(email, phone, 'No Contact') FROM employees;
```

---

## 12. Views

```sql
CREATE VIEW high_earners AS
SELECT name, salary FROM employees WHERE salary > 60000;

SELECT * FROM high_earners;

DROP VIEW high_earners;
```

---

## 13. Indexes

```sql
CREATE INDEX idx_name ON employees(name);
DROP INDEX idx_name ON employees;
```

---

## 14. TCL — Transaction Control

```sql
START TRANSACTION;
UPDATE employees SET salary = salary + 1000 WHERE dept_id = 1;
SAVEPOINT sp1;
UPDATE employees SET salary = salary - 500 WHERE dept_id = 2;
ROLLBACK TO sp1;
COMMIT;
```

---

## 15. DCL — Permissions

```sql
GRANT SELECT, INSERT ON company.employees TO 'user1'@'localhost';
REVOKE INSERT ON company.employees FROM 'user1'@'localhost';
```

---

## 16. Keys & Relationships (Concepts to Know)
- **Primary Key** – uniquely identifies each row
- **Foreign Key** – links two tables
- **Composite Key** – primary key made of multiple columns
- **Candidate Key** – column(s) that could be a primary key
- **Normalization** – organizing tables to reduce redundancy (1NF, 2NF, 3NF)

---

## 17. Common Interview/Exam Traps
- `WHERE` filters rows **before** grouping; `HAVING` filters **after** grouping.
- `NULL` isn't equal to anything, even `NULL` — always use `IS NULL` / `IS NOT NULL`.
- `DELETE` can be rolled back (with transactions); `TRUNCATE` and `DROP` usually can't.
- `INNER JOIN` drops unmatched rows; `LEFT/RIGHT JOIN` keeps them.
- Always alias tables in multi-table joins for readability (`e.name`, `d.dept_name`).

---

## Quick Practice Checklist
- [ ] Create a database and at least 2 related tables
- [ ] Insert sample data
- [ ] Practice SELECT with WHERE, ORDER BY, LIMIT
- [ ] Try all 4 join types
- [ ] Write one query using GROUP BY + HAVING
- [ ] Write one subquery
- [ ] Try a transaction with ROLLBACK
