DROP TABLE IF EXISTS {table_name}_timestamps_0;

CREATE TABLE {table_name}_timestamps_0 
AS
(SELECT DISTINCT download_timestamp AS t_
FROM {table_name});

DROP TABLE IF EXISTS {table_name}_timestamps_1;

CREATE TABLE {table_name}_timestamps_1 
AS
(SELECT t_0,
       MAX(t_1) AS t_1
FROM (SELECT t0.t_ AS t_0,
             t1.t_ AS t_1
      FROM {table_name}_timestamps_0 t0
        LEFT JOIN {table_name}_timestamps_0 t1 ON t0.t_ > t1.t_)
GROUP BY t_0);

DROP TABLE IF EXISTS buffer_{table_name}_2;

CREATE buffer_{table_name}_2 AS (
SELECT *
FROM (SELECT --t0.download_timestamp,
             --t0.update_timestamp,
             *,
             t1.t_1,
             CASE
               WHEN t1.t_1 IS NULL THEN 1
               WHEN t0.update_timestamp >= t1.t_1 THEN 1
               ELSE 0
             END AS new_
      FROM {buffer_table_name} t0
        JOIN {table_name}_timestamps_1 t1 ON t0.download_timestamp = t1.t_0)
WHERE new_ = 1);


SELECT id,
       station_id,
       name,
       download_timestamp,
       update_timestamp,
       EXTRACT(YEAR FROM update_timestamp) AS year_,
       EXTRACT(MONTH FROM update_timestamp) AS month_,
       EXTRACT(DAY FROM update_timestamp) AS dom,
       EXTRACT(DOW FROM update_timestamp) AS dow,
       EXTRACT(HOUR FROM update_timestamp) AS hour_,
       EXTRACT(MINUTE FROM update_timestamp) AS minute_,    
       longitude,
       latitude,
       capacity,
       numdocksavailable / CAST(capacity AS NUMERIC),
       mechanical / CAST(capacity AS NUMERIC),
       ebike / CAST(capacity AS NUMERIC)
FROM {table_name} WHERE capacity>0;