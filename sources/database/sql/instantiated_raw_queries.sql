-- Drop main tables

DROP TABLE IF EXISTS velib_0; --basic data

DROP TABLE IF EXISTS velib_1; --basic data and coordinate

DROP TABLE IF EXISTS velib_download_timestamps_0; -- distinct download times

DROP TABLE IF EXISTS velib_update_timestamps_0; -- latest update timestamp

DROP TABLE IF EXISTS velib_stations_0; -- distinct stations

DROP TABLE IF EXISTS velib_neighbors_0;  -- neighbors

DROP TABLE IF EXISTS velib_pairs_0; --pairs time of data for each station

DROP TABLE IF EXISTS velib_2; -- closest time 

DROP TABLE IF EXISTS velib_spells_0; -- spells as closest time pair 

--Create main tables

CREATE TABLE velib_0 (
id VARCHAR(255) PRIMARY KEY ,
station_id VARCHAR(255),
download_timestamp TIMESTAMP,
station_code VARCHAR(255),
station_name VARCHAR(255),
is_installed INT,
capacity INT,
available_docks NUMERIC,
mechanicals NUMERIC,
ebikes NUMERIC,
update_timestamp TIMESTAMP,
latitude NUMERIC,
longitude NUMERIC,
new_ BOOLEAN
);

CREATE TABLE velib_1 (
id VARCHAR(255) PRIMARY KEY ,
station_id VARCHAR(255),
download_timestamp TIMESTAMP,
station_code VARCHAR(255),
station_name VARCHAR(255),
is_installed INT,
capacity INT,
available_docks NUMERIC,
mechanicals NUMERIC,
ebikes NUMERIC,
update_timestamp TIMESTAMP,
latitude NUMERIC,
longitude NUMERIC,
pos_0 NUMERIC,
pos_1 NUMERIC,
new_ BOOLEAN
);

CREATE TABLE velib_download_timestamps_0 (
download_timestamp TIMESTAMP PRIMARY KEY,
new_ BOOLEAN
);

CREATE TABLE velib_update_timestamps_0 (
station_id VARCHAR(255) PRIMARY KEY ,
update_timestamp TIMESTAMP,
new_ BOOLEAN
);

CREATE TABLE velib_stations_0 (
station_id VARCHAR(255) PRIMARY KEY ,
station_code VARCHAR(255),
station_name VARCHAR(255),
latitude NUMERIC,
longitude NUMERIC,
pos_0 NUMERIC,
pos_1 NUMERIC,
new_ BOOLEAN
);

CREATE TABLE velib_neighbors_0 (
neighborhood_id VARCHAR(255) PRIMARY KEY, 
station_id VARCHAR(255),
neighbor_id VARCHAR(255),
one NUMERIC,
distance NUMERIC,
new_ BOOLEAN
);

CREATE TABLE velib_pairs_0 (
pair_id VARCHAR(255) PRIMARY KEY,
id VARCHAR(255),
station_id VARCHAR(255),
download_timestamp TIMESTAMP,
station_code VARCHAR(255),
station_name VARCHAR(255),
is_installed INT,
current_capacity INT,
current_available_docks NUMERIC,
current_mechanicals NUMERIC,
current_ebikes NUMERIC,
current_update_timestamp TIMESTAMP,
diff_capacity INT,
diff_available_docks NUMERIC,
diff_mechanicals NUMERIC,
diff_ebikes NUMERIC,
diff_update_timestamp INTERVAL,
latitude NUMERIC,
longitude NUMERIC,
pos_0 NUMERIC,
pos_1 NUMERIC,
new_ BOOLEAN
);

CREATE TABLE velib_2 (
id VARCHAR(255) PRIMARY KEY ,
station_id VARCHAR(255),
diff_update_timestamp INTERVAL,
new_ BOOLEAN
);

CREATE TABLE velib_spells_0 (
id VARCHAR(255) PRIMARY KEY ,
station_id VARCHAR(255),
download_timestamp TIMESTAMP,
station_code VARCHAR(255),
station_name VARCHAR(255),
is_installed INT,
current_capacity INT,
current_available_docks NUMERIC,
current_mechanicals NUMERIC,
current_ebikes NUMERIC,
current_update_timestamp TIMESTAMP,
diff_capacity INT,
diff_available_docks NUMERIC,
diff_mechanicals NUMERIC,
diff_ebikes NUMERIC,
diff_update_timestamp INTERVAL,
latitude NUMERIC,
longitude NUMERIC,
pos_0 NUMERIC,
pos_1 NUMERIC,
new_ BOOLEAN
);


--Drop create buffer tables
DROP TABLE IF EXISTS velib_buffer_from_csv;

DROP TABLE IF EXISTS velib_buffer_0;

DROP TABLE IF EXISTS velib_buffer_1;

DROP TABLE IF EXISTS velib_download_timestamps_buffer_0;

DROP TABLE IF EXISTS velib_update_timestamps_buffer_0;

DROP TABLE IF EXISTS velib_stations_buffer_0; 

DROP TABLE IF EXISTS velib_neighbors_buffer_0; 

DROP TABLE IF EXISTS velib_pairs_buffer_0; 

DROP TABLE IF EXISTS velib_buffer_2; 

DROP TABLE IF EXISTS velib_spells_buffer_0;

CREATE TABLE velib_buffer_from_csv (
station_code VARCHAR(255),
station_name VARCHAR(255),
is_installed VARCHAR(255),
capacity VARCHAR(255),
available_docks VARCHAR(255),
available_bikes VARCHAR(255),
mechanicals VARCHAR(255),
ebikes VARCHAR(255),
is_renting VARCHAR(255),
is_returning VARCHAR(255),
update_timestamp VARCHAR(255),
coordinates VARCHAR(255),
city_name VARCHAR(255),
insee_city_code VARCHAR(255)
);

CREATE TABLE velib_buffer_0 (
id VARCHAR(255) PRIMARY KEY ,
station_id VARCHAR(255),
download_timestamp TIMESTAMP,
station_code VARCHAR(255),
station_name VARCHAR(255),
is_installed INT,
capacity NUMERIC,
available_docks NUMERIC,
mechanicals NUMERIC,
ebikes NUMERIC,
update_timestamp TIMESTAMP,
latitude NUMERIC,
longitude NUMERIC,
new_ BOOLEAN
);

CREATE TABLE velib_buffer_1 (
id VARCHAR(255) PRIMARY KEY ,
station_id VARCHAR(255),
download_timestamp TIMESTAMP,
station_code VARCHAR(255),
station_name VARCHAR(255),
is_installed INT,
capacity INT,
available_docks NUMERIC,
mechanicals NUMERIC,
ebikes NUMERIC,
update_timestamp TIMESTAMP,
latitude NUMERIC,
longitude NUMERIC,
pos_0 NUMERIC,
pos_1 NUMERIC,
new_ BOOLEAN
);

CREATE TABLE velib_download_timestamps_buffer_0 (
download_timestamp TIMESTAMP PRIMARY KEY,
new_ BOOLEAN
);

CREATE TABLE velib_update_timestamps_buffer_0 (
station_id VARCHAR(255) PRIMARY KEY ,
update_timestamp TIMESTAMP,
new_ BOOLEAN
);

CREATE TABLE velib_stations_buffer_0 (
station_id VARCHAR(255) PRIMARY KEY ,
station_code VARCHAR(255),
station_name VARCHAR(255),
latitude NUMERIC,
longitude NUMERIC,
pos_0 NUMERIC,
pos_1 NUMERIC,
new_ BOOLEAN
);

CREATE TABLE velib_neighbors_buffer_0 (
neighborhood_id VARCHAR(255) PRIMARY KEY, 
station_id VARCHAR(255),
neighbor_id VARCHAR(255),
one NUMERIC,
distance NUMERIC,
new_ BOOLEAN
);

CREATE TABLE velib_pairs_buffer_0 (
pair_id VARCHAR(255) PRIMARY KEY,
id VARCHAR(255),
station_id VARCHAR(255),
download_timestamp TIMESTAMP,
station_code VARCHAR(255),
station_name VARCHAR(255),
is_installed INT,
current_capacity INT,
current_available_docks NUMERIC,
current_mechanicals NUMERIC,
current_ebikes NUMERIC,
current_update_timestamp TIMESTAMP,
diff_capacity INT,
diff_available_docks NUMERIC,
diff_mechanicals NUMERIC,
diff_ebikes NUMERIC,
diff_update_timestamp INTERVAL,
latitude NUMERIC,
longitude NUMERIC,
pos_0 NUMERIC,
pos_1 NUMERIC,
new_ BOOLEAN
);

CREATE TABLE velib_buffer_2 (
id VARCHAR(255) PRIMARY KEY ,
station_id VARCHAR(255),
diff_update_timestamp INTERVAL,
new_ BOOLEAN
);

CREATE TABLE velib_spells_buffer_0 (
id VARCHAR(255) PRIMARY KEY ,
station_id VARCHAR(255),
download_timestamp TIMESTAMP,
station_code VARCHAR(255),
station_name VARCHAR(255),
is_installed INT,
current_capacity INT,
current_available_docks NUMERIC,
current_mechanicals NUMERIC,
current_ebikes NUMERIC,
current_update_timestamp TIMESTAMP,
diff_capacity INT,
diff_available_docks NUMERIC,
diff_mechanicals NUMERIC,
diff_ebikes NUMERIC,
diff_update_timestamp INTERVAL,
latitude NUMERIC,
longitude NUMERIC,
pos_0 NUMERIC,
pos_1 NUMERIC,
new_ BOOLEAN
);
 
-- Insert or merge in buffer table
COPY velib_buffer_from_csv
FROM '/Users/julienguitard/local_data/velib/velib_buffer_1703258920.csv' CSV DELIMITER ';' HEADER;

INSERT INTO velib_buffer_0 (SELECT MD5(CONCAT(station_code,CAST(CAST(update_timestamp AS TIMESTAMP) AS VARCHAR(255)))) AS id,
       MD5(station_code) AS station_id,
       TO_TIMESTAMP(CAST(1703258920 AS BIGINT)) AS download_timestamp,
       CAST(station_code AS VARCHAR(255)) AS station_code,
       CAST(station_name AS VARCHAR(255)) AS station_name,
       CAST(is_installed = 'OUI' AS INT) AS is_installed,
       CAST(capacity AS NUMERIC) AS capacity,
       CAST(available_docks AS NUMERIC) / CAST(capacity AS NUMERIC) AS available_docks,
       CAST(mechanicals AS NUMERIC) / CAST(capacity AS NUMERIC) AS mechanical,
       CAST(ebikes AS NUMERIC) AS ebikes,
       CAST(update_timestamp AS TIMESTAMP) AS update_timestamp,
       CAST((STRING_TO_ARRAY(coordinates,','))[1] AS NUMERIC) AS longitude,
       CAST((STRING_TO_ARRAY(coordinates,','))[2] AS NUMERIC) AS latitude,
       TRUE AS new_
       FROM velib_buffer_from_csv 
       WHERE CAST(capacity AS NUMERIC)>0);

DELETE FROM velib_buffer_from_csv
WHERE TRUE;

MERGE INTO velib_0 t0
USING velib_buffer_0 t1 ON t0.id = t1.id AND t0.station_id = t1.station_id
WHEN MATCHED AND t0.download_timestamp < t1.download_timestamp THEN
  UPDATE SET download_timestamp=t1.download_timestamp, new_=t1.new_
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (
    t1.id,
    t1.station_id,
    t1.download_timestamp,
    t1.station_code,
    t1.station_name,
    t1.is_installed,
    t1.capacity,
    t1.available_docks,
    t1.mechanicals,
    t1.ebikes,
    t1.update_timestamp,
    t1.latitude,
    t1.longitude,
    t1.new_
  );

DELETE FROM velib_buffer_0
WHERE TRUE;

INSERT INTO velib_buffer_0 (SELECT *
FROM velib_0
WHERE new_);

INSERT INTO velib_buffer_1 (SELECT id,
       station_id,
       download_timestamp,
       station_code,
       station_name,
       is_installed,
       capacity,
       available_docks,
       mechanicals,
       ebikes,
       update_timestamp,
       latitude,
       longitude,
       2 *PI()*COS(2*PI()*latitude / 360)*longitude / 360 * 13500 AS pos_0,
       2 *PI()*latitude / 360 * 13500 AS pos_1 ,
       TRUE AS new_
FROM velib_buffer_0);

MERGE INTO velib_1 t0
USING velib_buffer_1 t1 ON t0.station_id = t1.station_id AND t0.update_timestamp=t1.update_timestamp
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (
    t1.id,
    t1.station_id,
    t1.download_timestamp,
    t1.station_code,
    t1.station_name,
    t1.is_installed,
    t1.capacity,
    t1.available_docks,
    t1.mechanicals,
    t1.ebikes,
    t1.update_timestamp,
    t1.latitude,
    t1.longitude,
    t1.pos_0,
    t1.pos_1,
    t1.new_
  );

DELETE FROM velib_buffer_1
WHERE TRUE;

INSERT INTO velib_buffer_1 (SELECT *
FROM velib_1
WHERE new_);

INSERT INTO velib_download_timestamps_buffer_0 (SELECT DISTINCT download_timestamp,
       TRUE AS new_
FROM velib_buffer_0
);

MERGE INTO velib_download_timestamps_0 t0
USING velib_download_timestamps_buffer_0 t1 ON t0.download_timestamp = t1.download_timestamp
WHEN NOT MATCHED THEN
  INSERT VALUES
(
  t1.download_timestamp,t1.new_
);

DELETE FROM velib_download_timestamps_buffer_0
WHERE TRUE;

INSERT INTO velib_download_timestamps_buffer_0 (SELECT *
FROM velib_download_timestamps_0
WHERE new_);

INSERT INTO velib_update_timestamps_buffer_0 (SELECT station_id,
       MAX(update_timestamp) AS update_timestamp,
       TRUE AS new_
FROM velib_buffer_0
GROUP BY station_id
);

MERGE INTO velib_update_timestamps_0 t0
USING velib_update_timestamps_buffer_0 t1 ON t0.station_id = t1.station_id
WHEN MATCHED AND t1.update_timestamp > t0.update_timestamp THEN UPDATE
  SET update_timestamp = t1.update_timestamp,
      new_ = t1.new_
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (
    t1.station_id,
    t1.update_timestamp,
    t1.new_
  );

DELETE FROM velib_update_timestamps_buffer_0
WHERE TRUE;

INSERT INTO velib_update_timestamps_buffer_0 (SELECT *
FROM velib_update_timestamps_0
WHERE new_);

INSERT INTO velib_stations_buffer_0 (SELECT station_id,
       MIN(station_code) AS station_code,
       MIN(station_name) AS station_name,
       MIN(latitude) AS latitude,
       MIN(longitude) AS longitude,
       MIN(pos_0) AS pos_0,
       MIN(pos_1) AS pos_1,
       TRUE AS new_
FROM velib_buffer_1
GROUP BY station_id);

MERGE INTO velib_stations_0 t0
USING velib_stations_buffer_0 t1 ON t0.station_id = t1.station_id
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (
    t1.station_id,
    t1.station_code,
    t1.station_name,
    t1.latitude,
    t1.longitude,
    t1.pos_0,
    t1.pos_1,
    t1.new_
  );

DELETE FROM velib_stations_buffer_0
WHERE TRUE;

INSERT INTO velib_stations_buffer_0 (SELECT *
FROM velib_stations_0
WHERE new_);

INSERT INTO velib_neighbors_buffer_0 (SELECT MD5(CONCAT(t0.station_id,t1.station_id)) AS neighborhood_id,
       t0.station_id,
       t1.station_id AS neighbor_id,
       1 AS one,
       POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5) AS distance,
       TRUE AS new_
FROM velib_stations_buffer_0 t0
  JOIN velib_stations_buffer_0 t1
    ON (POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5)) < 1
   AND t0.station_id != t1.station_id
UNION ALL
SELECT MD5(CONCAT(t0.station_id,t1.station_id)) AS neighborhood_id,
       t0.station_id,
       t1.station_id AS neighbor_id,
       1 AS one,
       POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5) AS distance,
       TRUE AS new_
FROM velib_stations_buffer_0 t0
  JOIN velib_stations_0 t1
    ON (POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5)) < 1
   AND t0.station_id != t1.station_id
   AND NOT(t1.new_)
UNION ALL
SELECT MD5(CONCAT(t0.station_id,t1.station_id)) AS neighborhood_id,
       t0.station_id,
       t1.station_id AS neighbor_id,
       1 AS one,
       POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5) AS distance,
       TRUE AS new_
FROM velib_stations_0 t0
  JOIN velib_stations_buffer_0 t1
    ON (POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5)) < 1
   AND t0.station_id != t1.station_id
   AND NOT(t0.new_));   
   
MERGE INTO velib_neighbors_0 t0
USING velib_neighbors_buffer_0 t1 ON t0.neighborhood_id = t1.neighborhood_id AND t0.station_id = t1.station_id AND t0.neighbor_id = t1.neighbor_id
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (
    t1.neighborhood_id,
    t1.station_id,
    t1.neighbor_id,
    t1.one,
    t1.distance,
    t1.new_
  );

DELETE FROM velib_neighbors_buffer_0
WHERE TRUE;

INSERT INTO velib_neighbors_buffer_0 (SELECT *
FROM velib_neighbors_0
WHERE new_);

INSERT INTO velib_pairs_buffer_0 (
SELECT MD5(CONCAT(t0.id,t1.id)) AS pair_id,
       t0.id,
       t0.station_id,
       t0.download_timestamp,
       t0.station_code,
       t0.station_name,
       t0.is_installed,
       t0.capacity AS current_capacity,
       t0.available_docks AS current_available_docks,
       t0.mechanicals AS current_mechanicals,
       t0.ebikes AS ebikes ,
       t0.update_timestamp AS current_update_timestamp,
       t0.capacity - t1.capacity AS diff_capacity,
       t0.available_docks - t1.available_docks AS diff_available_docks,
       t0.mechanicals - t1.mechanicals AS diff_mechanicals,
       t0.ebikes - t1.ebikes AS diff_ebikes,
       t0.update_timestamp - t1.update_timestamp AS diff_update_timestamp,
       t0.latitude,
       t0.longitude,
       t0.pos_0,
       t0.pos_1,
       TRUE AS new_
FROM velib_buffer_1 t0
  JOIN velib_buffer_1 t1
    ON t0.station_id = t1.station_id
    AND t0.update_timestamp != t1.update_timestamp
   AND EXTRACT (epoch FROM t0.update_timestamp - t1.update_timestamp) < 900
UNION ALL
SELECT MD5(CONCAT(t0.id,t1.id)) AS pair_id,
       t0.id,
       t0.station_id,
       t0.download_timestamp,
       t0.station_code,
       t0.station_name,
       t0.is_installed,
       t0.capacity AS current_capacity,
       t0.available_docks AS current_available_docks,
       t0.mechanicals AS current_mechanicals,
       t0.ebikes AS ebikes ,
       t0.update_timestamp AS current_update_timestamp,
       t0.capacity - t1.capacity AS diff_capacity,
       t0.available_docks - t1.available_docks AS diff_available_docks,
       t0.mechanicals - t1.mechanicals AS diff_mechanicals,
       t0.ebikes - t1.ebikes AS diff_ebikes,
       t0.update_timestamp - t1.update_timestamp AS diff_update_timestamp,
       t0.latitude,
       t0.longitude,
       t0.pos_0,
       t0.pos_1,
       TRUE AS new_
FROM velib_buffer_1 t0
  JOIN velib_1 t1
    ON t0.station_id = t1.station_id
    AND t0.update_timestamp != t1.update_timestamp
   AND EXTRACT (epoch FROM t0.update_timestamp - t1.update_timestamp) < 900
   AND NOT(t1.new_)
UNION ALL
SELECT MD5(CONCAT(t0.id,t1.id)) AS pair_id,
       t0.id,
       t0.station_id,
       t0.download_timestamp,
       t0.station_code,
       t0.station_name,
       t0.is_installed,
       t0.capacity AS current_capacity,
       t0.available_docks AS current_available_docks,
       t0.mechanicals AS current_mechanicals,
       t0.ebikes AS ebikes ,
       t0.update_timestamp AS current_update_timestamp,
       t0.capacity - t1.capacity AS diff_capacity,
       t0.available_docks - t1.available_docks AS diff_available_docks,
       t0.mechanicals - t1.mechanicals AS diff_mechanicals,
       t0.ebikes - t1.ebikes AS diff_ebikes,
       t0.update_timestamp - t1.update_timestamp AS diff_update_timestamp,
       t0.latitude,
       t0.longitude,
       t0.pos_0,
       t0.pos_1,
       TRUE AS new_
FROM velib_1 t0
  JOIN velib_buffer_1 t1
    ON t0.station_id = t1.station_id
    AND t0.update_timestamp != t1.update_timestamp
   AND EXTRACT (epoch FROM t0.update_timestamp - t1.update_timestamp) < 900
   AND NOT(t0.new_)
);

   
MERGE INTO velib_pairs_0 t0
USING velib_pairs_buffer_0 t1 ON t0.pair_id = t1.pair_id AND t0.id = t1.id AND t0.station_id = t1.station_id
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (t1.pair_id,
       t1.id,
       t1.station_id,
       t1.download_timestamp,
       t1.station_code,
       t1.station_name,
       t1.is_installed,
       t1.current_capacity,
       t1.current_available_docks,
       t1.current_mechanicals,
       t1.current_ebikes ,
       t1.current_update_timestamp,
       t1.diff_capacity,
       t1.diff_available_docks,
       t1.diff_mechanicals,
       t1.diff_ebikes,
       t1.diff_update_timestamp,
       t1.latitude,
       t1.longitude,
       t1.pos_0,
       t1.pos_1,
       t1.new_
  );

DELETE FROM velib_pairs_buffer_0
WHERE TRUE;

INSERT INTO velib_pairs_buffer_0 (SELECT *
FROM velib_pairs_0
WHERE new_);

INSERT INTO velib_buffer_2 (
SELECT id,
       station_id,
       MIN(diff_update_timestamp) AS diff_update_timestamp,
       TRUE AS new_
FROM velib_pairs_buffer_0
GROUP BY id,
         station_id
);

MERGE INTO velib_2 t0
USING velib_buffer_2 t1 ON t0.id = t1.id AND t0.station_id = t1.station_id
WHEN MATCHED AND t0.diff_update_timestamp > t1.diff_update_timestamp THEN UPDATE
  SET diff_update_timestamp = t1.diff_update_timestamp,
      new_ = t1.new_
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (
    t1.id,
    t1.station_id,
    t1.diff_update_timestamp,
    t1.new_
  );

DELETE FROM velib_buffer_2
WHERE TRUE;

INSERT INTO velib_buffer_2 (SELECT *
FROM velib_2
WHERE new_);

INSERT INTO velib_spells_buffer_0 (
SELECT t0.id,
       t0.station_id,
       t0.download_timestamp,
       t0.station_code,
       t0.station_name,
       t0.is_installed,
       t0.current_capacity,
       t0.current_available_docks,
       t0.current_mechanicals,
       t0.current_ebikes,
       t0.current_update_timestamp,
       t0.diff_capacity,
       t0.diff_available_docks,
       t0.diff_mechanicals,
       t0.diff_ebikes,
       t0.diff_update_timestamp,
       t0.latitude,
       t0.longitude,
       t0.pos_0,
       t0.pos_1,
       TRUE AS new_
FROM velib_pairs_buffer_0 t0
  JOIN velib_buffer_2 t1
    ON t0.id = t1.id
   AND t0.station_id = t1.station_id
   AND t0.diff_update_timestamp = t1.diff_update_timestamp
UNION ALL
SELECT t0.id,
       t0.station_id,
       t0.download_timestamp,
       t0.station_code,
       t0.station_name,
       t0.is_installed,
       t0.current_capacity,
       t0.current_available_docks,
       t0.current_mechanicals,
       t0.current_ebikes,
       t0.current_update_timestamp,
       t0.diff_capacity,
       t0.diff_available_docks,
       t0.diff_mechanicals,
       t0.diff_ebikes,
       t0.diff_update_timestamp,
       t0.latitude,
       t0.longitude,
       t0.pos_0,
       t0.pos_1,
       TRUE AS new_
FROM velib_pairs_buffer_0 t0
  JOIN velib_2 t1
    ON t0.id = t1.id
   AND t0.station_id = t1.station_id
   AND t0.diff_update_timestamp = t1.diff_update_timestamp
   AND NOT(t1.new_)
UNION ALL
SELECT t0.id,
       t0.station_id,
       t0.download_timestamp,
       t0.station_code,
       t0.station_name,
       t0.is_installed,
       t0.current_capacity,
       t0.current_available_docks,
       t0.current_mechanicals,
       t0.current_ebikes,
       t0.current_update_timestamp,
       t0.diff_capacity,
       t0.diff_available_docks,
       t0.diff_mechanicals,
       t0.diff_ebikes,
       t0.diff_update_timestamp,
       t0.latitude,
       t0.longitude,
       t0.pos_0,
       t0.pos_1,
       TRUE AS new_
FROM velib_pairs_0 t0
  JOIN velib_buffer_2 t1
    ON t0.id = t1.id
   AND t0.station_id = t1.station_id
   AND t0.diff_update_timestamp = t1.diff_update_timestamp
   AND NOT(t0.new_));

MERGE INTO velib_spells_0 t0
USING velib_spells_buffer_0 t1 ON t0.id = t1.id AND t0.station_id = t1.station_id
WHEN MATCHED AND t0.diff_update_timestamp > t1.diff_update_timestamp THEN UPDATE
  SET diff_update_timestamp = t1.diff_update_timestamp,
      diff_capacity = t1.diff_capacity,
      diff_available_docks = t1.diff_available_docks,
      diff_mechanicals = t1.diff_mechanicals,
      diff_ebikes = t1.diff_ebikes,
      new_ = t1.new_
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (
    t1.id,
    t1.station_id,
    t1.download_timestamp,
    t1.station_code,
    t1.station_name,
    t1.is_installed,
    t1.current_capacity,
    t1.current_available_docks,
    t1.current_mechanicals,
    t1.current_ebikes,
    t1.current_update_timestamp,
    t1.diff_capacity,
    t1.diff_available_docks,
    t1.diff_mechanicals,
    t1.diff_ebikes,
    t1.diff_update_timestamp,
    t1.latitude,
    t1.longitude,
    t1.pos_0,
    t1.pos_1,
    t1.new_
  );

DELETE
FROM velib_spells_buffer_0
WHERE TRUE;

INSERT INTO velib_spells_buffer_0
(SELECT*FROM velib_spells_0 WHERE new_);


--Update status

DELETE FROM velib_buffer_0
WHERE TRUE;

DELETE FROM velib_buffer_1
WHERE TRUE;

DELETE FROM velib_download_timestamps_buffer_0
WHERE TRUE;

DELETE FROM velib_update_timestamps_buffer_0
WHERE TRUE;

DELETE FROM velib_stations_buffer_0
WHERE TRUE;

DELETE FROM velib_neighbors_buffer_0
WHERE TRUE;
   
DELETE FROM velib_pairs_buffer_0
WHERE TRUE;
   
DELETE FROM velib_buffer_2
WHERE TRUE;
   
DELETE FROM velib_spells_buffer_0
WHERE TRUE;

UPDATE velib_0
   SET new_ = FALSE;

UPDATE velib_1
   SET new_ = FALSE;

UPDATE velib_download_timestamps_0
   SET new_ = FALSE;
   
UPDATE velib_update_timestamps_0
   SET new_ = FALSE;

UPDATE velib_stations_0
   SET new_ = FALSE;

UPDATE velib_neighbors_0
   SET new_ = FALSE;
   
UPDATE velib_pairs_0
   SET new_ = FALSE;
   
UPDATE velib_2
   SET new_ = FALSE;
   
UPDATE velib_spells_0
   SET new_ = FALSE;

--REPEAT
-- Insert or merge in buffer table
COPY velib_buffer_from_csv
FROM '/Users/julienguitard/local_data/velib/velib_buffer_1703259220.csv' CSV DELIMITER ';' HEADER;

INSERT INTO velib_buffer_0 (SELECT MD5(CONCAT(station_code,CAST(CAST(update_timestamp AS TIMESTAMP) AS VARCHAR(255)))) AS id,
       MD5(station_code) AS station_id,
       TO_TIMESTAMP(CAST(1703259220 AS BIGINT)) AS download_timestamp,
       CAST(station_code AS VARCHAR(255)) AS station_code,
       CAST(station_name AS VARCHAR(255)) AS station_name,
       CAST(is_installed = 'OUI' AS INT) AS is_installed,
       CAST(capacity AS NUMERIC) AS capacity,
       CAST(available_docks AS NUMERIC) / CAST(capacity AS NUMERIC) AS available_docks,
       CAST(mechanicals AS NUMERIC) / CAST(capacity AS NUMERIC) AS mechanical,
       CAST(ebikes AS NUMERIC) AS ebikes,
       CAST(update_timestamp AS TIMESTAMP) AS update_timestamp,
       CAST((STRING_TO_ARRAY(coordinates,','))[1] AS NUMERIC) AS longitude,
       CAST((STRING_TO_ARRAY(coordinates,','))[2] AS NUMERIC) AS latitude,
       TRUE AS new_
       FROM velib_buffer_from_csv 
       WHERE CAST(capacity AS NUMERIC)>0);

DELETE FROM velib_buffer_from_csv
WHERE TRUE;

MERGE INTO velib_0 t0
USING velib_buffer_0 t1 ON t0.id = t1.id AND t0.station_id = t1.station_id
WHEN MATCHED AND t0.download_timestamp < t1.download_timestamp THEN
  UPDATE SET download_timestamp=t1.download_timestamp, new_=t1.new_
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (
    t1.id,
    t1.station_id,
    t1.download_timestamp,
    t1.station_code,
    t1.station_name,
    t1.is_installed,
    t1.capacity,
    t1.available_docks,
    t1.mechanicals,
    t1.ebikes,
    t1.update_timestamp,
    t1.latitude,
    t1.longitude,
    t1.new_
  );

DELETE FROM velib_buffer_0
WHERE TRUE;

INSERT INTO velib_buffer_0 (SELECT *
FROM velib_0
WHERE new_);

INSERT INTO velib_buffer_1 (SELECT id,
       station_id,
       download_timestamp,
       station_code,
       station_name,
       is_installed,
       capacity,
       available_docks,
       mechanicals,
       ebikes,
       update_timestamp,
       latitude,
       longitude,
       2 *PI()*COS(2*PI()*latitude / 360)*longitude / 360 * 13500 AS pos_0,
       2 *PI()*latitude / 360 * 13500 AS pos_1 ,
       TRUE AS new_
FROM velib_buffer_0);

MERGE INTO velib_1 t0
USING velib_buffer_1 t1 ON t0.station_id = t1.station_id AND t0.update_timestamp=t1.update_timestamp
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (
    t1.id,
    t1.station_id,
    t1.download_timestamp,
    t1.station_code,
    t1.station_name,
    t1.is_installed,
    t1.capacity,
    t1.available_docks,
    t1.mechanicals,
    t1.ebikes,
    t1.update_timestamp,
    t1.latitude,
    t1.longitude,
    t1.pos_0,
    t1.pos_1,
    t1.new_
  );

DELETE FROM velib_buffer_1
WHERE TRUE;

INSERT INTO velib_buffer_1 (SELECT *
FROM velib_1
WHERE new_);

INSERT INTO velib_download_timestamps_buffer_0 (SELECT DISTINCT download_timestamp,
       TRUE AS new_
FROM velib_buffer_0
);

MERGE INTO velib_download_timestamps_0 t0
USING velib_download_timestamps_buffer_0 t1 ON t0.download_timestamp = t1.download_timestamp
WHEN NOT MATCHED THEN
  INSERT VALUES
(
  t1.download_timestamp,t1.new_
);

DELETE FROM velib_download_timestamps_buffer_0
WHERE TRUE;

INSERT INTO velib_download_timestamps_buffer_0 (SELECT *
FROM velib_download_timestamps_0
WHERE new_);

INSERT INTO velib_update_timestamps_buffer_0 (SELECT station_id,
       MAX(update_timestamp) AS update_timestamp,
       TRUE AS new_
FROM velib_buffer_0
GROUP BY station_id
);

MERGE INTO velib_update_timestamps_0 t0
USING velib_update_timestamps_buffer_0 t1 ON t0.station_id = t1.station_id
WHEN MATCHED AND t1.update_timestamp > t0.update_timestamp THEN UPDATE
  SET update_timestamp = t1.update_timestamp,
      new_ = t1.new_
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (
    t1.station_id,
    t1.update_timestamp,
    t1.new_
  );

DELETE FROM velib_update_timestamps_buffer_0
WHERE TRUE;

INSERT INTO velib_update_timestamps_buffer_0 (SELECT *
FROM velib_update_timestamps_0
WHERE new_);

INSERT INTO velib_stations_buffer_0 (SELECT station_id,
       MIN(station_code) AS station_code,
       MIN(station_name) AS station_name,
       MIN(latitude) AS latitude,
       MIN(longitude) AS longitude,
       MIN(pos_0) AS pos_0,
       MIN(pos_1) AS pos_1,
       TRUE AS new_
FROM velib_buffer_1
GROUP BY station_id);

MERGE INTO velib_stations_0 t0
USING velib_stations_buffer_0 t1 ON t0.station_id = t1.station_id
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (
    t1.station_id,
    t1.station_code,
    t1.station_name,
    t1.latitude,
    t1.longitude,
    t1.pos_0,
    t1.pos_1,
    t1.new_
  );

DELETE FROM velib_stations_buffer_0
WHERE TRUE;

INSERT INTO velib_stations_buffer_0 (SELECT *
FROM velib_stations_0
WHERE new_);

INSERT INTO velib_neighbors_buffer_0 (SELECT MD5(CONCAT(t0.station_id,t1.station_id)) AS neighborhood_id,
       t0.station_id,
       t1.station_id AS neighbor_id,
       1 AS one,
       POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5) AS distance,
       TRUE AS new_
FROM velib_stations_buffer_0 t0
  JOIN velib_stations_buffer_0 t1
    ON (POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5)) < 1
   AND t0.station_id != t1.station_id
UNION ALL
SELECT MD5(CONCAT(t0.station_id,t1.station_id)) AS neighborhood_id,
       t0.station_id,
       t1.station_id AS neighbor_id,
       1 AS one,
       POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5) AS distance,
       TRUE AS new_
FROM velib_stations_buffer_0 t0
  JOIN velib_stations_0 t1
    ON (POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5)) < 1
   AND t0.station_id != t1.station_id
   AND NOT(t1.new_)
UNION ALL
SELECT MD5(CONCAT(t0.station_id,t1.station_id)) AS neighborhood_id,
       t0.station_id,
       t1.station_id AS neighbor_id,
       1 AS one,
       POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5) AS distance,
       TRUE AS new_
FROM velib_stations_0 t0
  JOIN velib_stations_buffer_0 t1
    ON (POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5)) < 1
   AND t0.station_id != t1.station_id
   AND NOT(t0.new_));   
   
MERGE INTO velib_neighbors_0 t0
USING velib_neighbors_buffer_0 t1 ON t0.neighborhood_id = t1.neighborhood_id AND t0.station_id = t1.station_id AND t0.neighbor_id = t1.neighbor_id
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (
    t1.neighborhood_id,
    t1.station_id,
    t1.neighbor_id,
    t1.one,
    t1.distance,
    t1.new_
  );

DELETE FROM velib_neighbors_buffer_0
WHERE TRUE;

INSERT INTO velib_neighbors_buffer_0 (SELECT *
FROM velib_neighbors_0
WHERE new_);

INSERT INTO velib_pairs_buffer_0 (
SELECT MD5(CONCAT(t0.id,t1.id)) AS pair_id,
       t0.id,
       t0.station_id,
       t0.download_timestamp,
       t0.station_code,
       t0.station_name,
       t0.is_installed,
       t0.capacity AS current_capacity,
       t0.available_docks AS current_available_docks,
       t0.mechanicals AS current_mechanicals,
       t0.ebikes AS ebikes ,
       t0.update_timestamp AS current_update_timestamp,
       t0.capacity - t1.capacity AS diff_capacity,
       t0.available_docks - t1.available_docks AS diff_available_docks,
       t0.mechanicals - t1.mechanicals AS diff_mechanicals,
       t0.ebikes - t1.ebikes AS diff_ebikes,
       t0.update_timestamp - t1.update_timestamp AS diff_update_timestamp,
       t0.latitude,
       t0.longitude,
       t0.pos_0,
       t0.pos_1,
       TRUE AS new_
FROM velib_buffer_1 t0
  JOIN velib_buffer_1 t1
    ON t0.station_id = t1.station_id
    AND t0.update_timestamp != t1.update_timestamp
   AND EXTRACT (epoch FROM t0.update_timestamp - t1.update_timestamp) < 900
UNION ALL
SELECT MD5(CONCAT(t0.id,t1.id)) AS pair_id,
       t0.id,
       t0.station_id,
       t0.download_timestamp,
       t0.station_code,
       t0.station_name,
       t0.is_installed,
       t0.capacity AS current_capacity,
       t0.available_docks AS current_available_docks,
       t0.mechanicals AS current_mechanicals,
       t0.ebikes AS ebikes ,
       t0.update_timestamp AS current_update_timestamp,
       t0.capacity - t1.capacity AS diff_capacity,
       t0.available_docks - t1.available_docks AS diff_available_docks,
       t0.mechanicals - t1.mechanicals AS diff_mechanicals,
       t0.ebikes - t1.ebikes AS diff_ebikes,
       t0.update_timestamp - t1.update_timestamp AS diff_update_timestamp,
       t0.latitude,
       t0.longitude,
       t0.pos_0,
       t0.pos_1,
       TRUE AS new_
FROM velib_buffer_1 t0
  JOIN velib_1 t1
    ON t0.station_id = t1.station_id
    AND t0.update_timestamp != t1.update_timestamp
   AND EXTRACT (epoch FROM t0.update_timestamp - t1.update_timestamp) < 900
   AND NOT(t1.new_)
UNION ALL
SELECT MD5(CONCAT(t0.id,t1.id)) AS pair_id,
       t0.id,
       t0.station_id,
       t0.download_timestamp,
       t0.station_code,
       t0.station_name,
       t0.is_installed,
       t0.capacity AS current_capacity,
       t0.available_docks AS current_available_docks,
       t0.mechanicals AS current_mechanicals,
       t0.ebikes AS ebikes ,
       t0.update_timestamp AS current_update_timestamp,
       t0.capacity - t1.capacity AS diff_capacity,
       t0.available_docks - t1.available_docks AS diff_available_docks,
       t0.mechanicals - t1.mechanicals AS diff_mechanicals,
       t0.ebikes - t1.ebikes AS diff_ebikes,
       t0.update_timestamp - t1.update_timestamp AS diff_update_timestamp,
       t0.latitude,
       t0.longitude,
       t0.pos_0,
       t0.pos_1,
       TRUE AS new_
FROM velib_1 t0
  JOIN velib_buffer_1 t1
    ON t0.station_id = t1.station_id
    AND t0.update_timestamp != t1.update_timestamp
   AND EXTRACT (epoch FROM t0.update_timestamp - t1.update_timestamp) < 900
   AND NOT(t0.new_)
);

   
MERGE INTO velib_pairs_0 t0
USING velib_pairs_buffer_0 t1 ON t0.pair_id = t1.pair_id AND t0.id = t1.id AND t0.station_id = t1.station_id
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (t1.pair_id,
       t1.id,
       t1.station_id,
       t1.download_timestamp,
       t1.station_code,
       t1.station_name,
       t1.is_installed,
       t1.current_capacity,
       t1.current_available_docks,
       t1.current_mechanicals,
       t1.current_ebikes ,
       t1.current_update_timestamp,
       t1.diff_capacity,
       t1.diff_available_docks,
       t1.diff_mechanicals,
       t1.diff_ebikes,
       t1.diff_update_timestamp,
       t1.latitude,
       t1.longitude,
       t1.pos_0,
       t1.pos_1,
       t1.new_
  );

DELETE FROM velib_pairs_buffer_0
WHERE TRUE;

INSERT INTO velib_pairs_buffer_0 (SELECT *
FROM velib_pairs_0
WHERE new_);

INSERT INTO velib_buffer_2 (
SELECT id,
       station_id,
       MIN(diff_update_timestamp) AS diff_update_timestamp,
       TRUE AS new_
FROM velib_pairs_buffer_0
GROUP BY id,
         station_id
);

MERGE INTO velib_2 t0
USING velib_buffer_2 t1 ON t0.id = t1.id AND t0.station_id = t1.station_id
WHEN MATCHED AND t0.diff_update_timestamp > t1.diff_update_timestamp THEN UPDATE
  SET diff_update_timestamp = t1.diff_update_timestamp,
      new_ = t1.new_
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (
    t1.id,
    t1.station_id,
    t1.diff_update_timestamp,
    t1.new_
  );

DELETE FROM velib_buffer_2
WHERE TRUE;

INSERT INTO velib_buffer_2 (SELECT *
FROM velib_2
WHERE new_);

INSERT INTO velib_spells_buffer_0 (
SELECT t0.id,
       t0.station_id,
       t0.download_timestamp,
       t0.station_code,
       t0.station_name,
       t0.is_installed,
       t0.current_capacity,
       t0.current_available_docks,
       t0.current_mechanicals,
       t0.current_ebikes,
       t0.current_update_timestamp,
       t0.diff_capacity,
       t0.diff_available_docks,
       t0.diff_mechanicals,
       t0.diff_ebikes,
       t0.diff_update_timestamp,
       t0.latitude,
       t0.longitude,
       t0.pos_0,
       t0.pos_1,
       TRUE AS new_
FROM velib_pairs_buffer_0 t0
  JOIN velib_buffer_2 t1
    ON t0.id = t1.id
   AND t0.station_id = t1.station_id
   AND t0.diff_update_timestamp = t1.diff_update_timestamp
UNION ALL
SELECT t0.id,
       t0.station_id,
       t0.download_timestamp,
       t0.station_code,
       t0.station_name,
       t0.is_installed,
       t0.current_capacity,
       t0.current_available_docks,
       t0.current_mechanicals,
       t0.current_ebikes,
       t0.current_update_timestamp,
       t0.diff_capacity,
       t0.diff_available_docks,
       t0.diff_mechanicals,
       t0.diff_ebikes,
       t0.diff_update_timestamp,
       t0.latitude,
       t0.longitude,
       t0.pos_0,
       t0.pos_1,
       TRUE AS new_
FROM velib_pairs_buffer_0 t0
  JOIN velib_2 t1
    ON t0.id = t1.id
   AND t0.station_id = t1.station_id
   AND t0.diff_update_timestamp = t1.diff_update_timestamp
   AND NOT(t1.new_)
UNION ALL
SELECT t0.id,
       t0.station_id,
       t0.download_timestamp,
       t0.station_code,
       t0.station_name,
       t0.is_installed,
       t0.current_capacity,
       t0.current_available_docks,
       t0.current_mechanicals,
       t0.current_ebikes,
       t0.current_update_timestamp,
       t0.diff_capacity,
       t0.diff_available_docks,
       t0.diff_mechanicals,
       t0.diff_ebikes,
       t0.diff_update_timestamp,
       t0.latitude,
       t0.longitude,
       t0.pos_0,
       t0.pos_1,
       TRUE AS new_
FROM velib_pairs_0 t0
  JOIN velib_buffer_2 t1
    ON t0.id = t1.id
   AND t0.station_id = t1.station_id
   AND t0.diff_update_timestamp = t1.diff_update_timestamp
   AND NOT(t0.new_));

MERGE INTO velib_spells_0 t0
USING velib_spells_buffer_0 t1 ON t0.id = t1.id AND t0.station_id = t1.station_id
WHEN MATCHED AND t0.diff_update_timestamp > t1.diff_update_timestamp THEN UPDATE
  SET diff_update_timestamp = t1.diff_update_timestamp,
      diff_capacity = t1.diff_capacity,
      diff_available_docks = t1.diff_available_docks,
      diff_mechanicals = t1.diff_mechanicals,
      diff_ebikes = t1.diff_ebikes,
      new_ = t1.new_
WHEN NOT MATCHED THEN
  INSERT
  VALUES
  (
    t1.id,
    t1.station_id,
    t1.download_timestamp,
    t1.station_code,
    t1.station_name,
    t1.is_installed,
    t1.current_capacity,
    t1.current_available_docks,
    t1.current_mechanicals,
    t1.current_ebikes,
    t1.current_update_timestamp,
    t1.diff_capacity,
    t1.diff_available_docks,
    t1.diff_mechanicals,
    t1.diff_ebikes,
    t1.diff_update_timestamp,
    t1.latitude,
    t1.longitude,
    t1.pos_0,
    t1.pos_1,
    t1.new_
  );

DELETE
FROM velib_spells_buffer_0
WHERE TRUE;

INSERT INTO velib_spells_buffer_0
(SELECT*FROM velib_spells_0 WHERE new_);


--Update status

DELETE FROM velib_buffer_0
WHERE TRUE;

DELETE FROM velib_buffer_1
WHERE TRUE;

DELETE FROM velib_download_timestamps_buffer_0
WHERE TRUE;

DELETE FROM velib_update_timestamps_buffer_0
WHERE TRUE;

DELETE FROM velib_stations_buffer_0
WHERE TRUE;

DELETE FROM velib_neighbors_buffer_0
WHERE TRUE;
   
DELETE FROM velib_pairs_buffer_0
WHERE TRUE;
   
DELETE FROM velib_buffer_2
WHERE TRUE;
   
DELETE FROM velib_spells_buffer_0
WHERE TRUE;

UPDATE velib_0
   SET new_ = FALSE;

UPDATE velib_1
   SET new_ = FALSE;

UPDATE velib_download_timestamps_0
   SET new_ = FALSE;
   
UPDATE velib_update_timestamps_0
   SET new_ = FALSE;

UPDATE velib_stations_0
   SET new_ = FALSE;

UPDATE velib_neighbors_0
   SET new_ = FALSE;
   
UPDATE velib_pairs_0
   SET new_ = FALSE;
   
UPDATE velib_2
   SET new_ = FALSE;
   
UPDATE velib_spells_0
   SET new_ = FALSE;

SELECT AVG(pos_0), AVG(pos_1) FROM velib_stations_0;

SELECT cnt, COUNT(station_id) FROM (
SELECT station_id, COUNT(*) AS cnt FROM velib_neighbors_0 GROUP BY station_id) GROUP BY cnt; 
   


   

