# Stored Procedures — Explained Using Chocolate Sales Data

> Note: SQLite does **not** support stored procedures (no `CREATE PROCEDURE`). This is a MySQL / PostgreSQL / SQL Server feature. The examples below assume the Shipments and Dimension_Data tables from your Excel file are imported into MySQL.

## Dataset Schema (from your file)

**Shipments table**
| Column | Description |
|---|---|
| ShipmentID | Unique shipment id (e.g. S00000012) |
| SPID | Sales person id |
| PID | Product id |
| GID | Geography id |
| Shipdate | Date of shipment |
| Amount | Sale amount |
| Boxes | Number of boxes shipped |
| Order_Status | e.g. Delivered |

**Dimension_Data table**
| Column | Description |
|---|---|
| Product, Category, Cost_per_box, PID | Product info |
| Geo, Region, GID | Location info |
| Sales_person, Team, Picture, SPID | Sales rep info |

---

## What is a Stored Procedure?

A stored procedure is a **saved block of SQL code** (with logic — variables, loops, conditions) that lives inside the database itself. Instead of writing the same query repeatedly from your application, you save it once and *call it by name*, like a function.

---

## Example 1: Simple stored procedure (no parameters)

Total sales amount per sales person:

```sql
DELIMITER //

CREATE PROCEDURE GetTotalSalesByPerson()
BEGIN
    SELECT d.Sales_person, SUM(s.Amount) AS Total_Sales
    FROM Shipments s
    JOIN Dimension_Data d ON s.SPID = d.SPID
    GROUP BY d.Sales_person
    ORDER BY Total_Sales DESC;
END //

DELIMITER ;
```

Call it:
```sql
CALL GetTotalSalesByPerson();
```

*(`DELIMITER //` is needed because the procedure body has semicolons inside it — MySQL needs a different end-marker so it doesn't stop early.)*

---

## Example 2: Stored procedure WITH a parameter (IN)

Sales for one specific product:

```sql
DELIMITER //

CREATE PROCEDURE GetSalesByProduct(IN productID VARCHAR(10))
BEGIN
    SELECT d.Product, s.Shipdate, s.Amount, s.Boxes, s.Order_Status
    FROM Shipments s
    JOIN Dimension_Data d ON s.PID = d.PID
    WHERE s.PID = productID
    ORDER BY s.Shipdate;
END //

DELIMITER ;
```

Call it for any product:
```sql
CALL GetSalesByProduct('P01');   -- Milk Bars
CALL GetSalesByProduct('P02');   -- 50% Dark Bites
```

---

## Example 3: Stored procedure that RETURNS a value (OUT parameter)

Total revenue for a region, returned as a single value:

```sql
DELIMITER //

CREATE PROCEDURE GetRegionRevenue(
    IN regionName VARCHAR(50),
    OUT totalRevenue DECIMAL(12,2)
)
BEGIN
    SELECT SUM(s.Amount) INTO totalRevenue
    FROM Shipments s
    JOIN Dimension_Data d ON s.GID = d.GID
    WHERE d.Region = regionName;
END //

DELIMITER ;
```

Call it and read the result:
```sql
CALL GetRegionRevenue('APAC', @rev);
SELECT @rev;
```

---

## Example 4: Adding logic (IF / ELSEIF conditions)

Flag whether a shipment is "high value":

```sql
DELIMITER //

CREATE PROCEDURE CheckShipmentValue(IN shipID VARCHAR(15))
BEGIN
    DECLARE amt DECIMAL(10,2);
    SELECT Amount INTO amt FROM Shipments WHERE ShipmentID = shipID;

    IF amt > 10000 THEN
        SELECT 'High Value Shipment' AS Result;
    ELSEIF amt > 5000 THEN
        SELECT 'Medium Value Shipment' AS Result;
    ELSE
        SELECT 'Standard Shipment' AS Result;
    END IF;
END //

DELIMITER ;
```

Call it:
```sql
CALL CheckShipmentValue('S00000058');   -- Amount was 13279.5 -> "High Value Shipment"
```

---

## Managing Stored Procedures

```sql
SHOW PROCEDURE STATUS WHERE Db = 'your_database';   -- list all procedures
SHOW CREATE PROCEDURE GetTotalSalesByPerson;         -- view a procedure's definition
DROP PROCEDURE GetTotalSalesByPerson;                -- delete one
DROP PROCEDURE IF EXISTS GetTotalSalesByPerson;      -- safe delete
```

---

## Why Use Stored Procedures?

| Benefit | Why it matters |
|---|---|
| **Reusability** | Write the sales-by-region query once, call it forever |
| **Performance** | Precompiled — often faster than sending raw SQL each time |
| **Security** | App can `CALL` a procedure without direct table access |
| **Consistency** | Everyone on the team gets the same logic, no copy-paste drift |
| **Reduced network traffic** | One `CALL` vs sending a full query string each time |

## Stored Procedures vs Functions (quick note)
| Stored Procedure | Function |
|---|---|
| Can return 0, 1, or many values (via OUT params or result sets) | Must return exactly one value |
| Called with `CALL procedure_name()` | Called inside a query, e.g. `SELECT function_name()` |
| Can perform INSERT/UPDATE/DELETE | Typically used only for calculations |
