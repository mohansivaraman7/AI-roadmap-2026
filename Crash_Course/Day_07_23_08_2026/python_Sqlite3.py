import sqlite3

connection = sqlite3.connect("uptor.db")
cursor = connection.cursor()
cursor.execute("""CREATE TABLE employees (
    emp_id INT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    age INT CHECK (age >= 18),
    salary DECIMAL(10,2) DEFAULT 0,
    dept_id INT,
    email VARCHAR(100) UNIQUE
);""")
connection.close()