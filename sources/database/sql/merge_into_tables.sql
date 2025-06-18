-- Insert or merge in buffer table
COPY {affix}_buffer_from_csv
FROM '{source}' CSV DELIMITER ';' HEADER;

INSERT INTO {affix}_buffer_0 (SELECT MD5(CONCAT(station_code,CAST(CAST(update_timestamp AS TIMESTAMP) AS VARCHAR(255)))) AS id,
       MD5(station_code) AS station_id,
       TO_TIMESTAMP(CAST({epoch} AS BIGINT)) AS download_timestamp,
       CAST(station_code AS VARCHAR(255)) AS station_code,
       CAST(station_name AS VARCHAR(255)) AS station_name,
       CAST(is_installed = 'OUI' AS INT) AS is_installed,
       CAST(capacity AS NUMERIC) AS capacity,
       CAST(available_docks AS NUMERIC) / CAST(capacity AS NUMERIC) AS available_docks,
       CAST(mechanicals AS NUMERIC) / CAST(capacity AS NUMERIC) AS mechanical,
       CAST(ebikes AS NUMERIC) / CAST(capacity AS NUMERIC) AS ebikes,
       CAST(update_timestamp AS TIMESTAMP) AS update_timestamp,
       CAST((STRING_TO_ARRAY(coordinates,','))[1] AS NUMERIC) AS longitude,
       CAST((STRING_TO_ARRAY(coordinates,','))[2] AS NUMERIC) AS latitude,
       TRUE AS new_
       FROM {affix}_buffer_from_csv 
       WHERE CAST(capacity AS NUMERIC)>0);

DELETE FROM {affix}_buffer_from_csv
WHERE TRUE;

MERGE INTO {affix}_0 t0
USING {affix}_buffer_0 t1 ON t0.id = t1.id AND t0.station_id = t1.station_id
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

DELETE FROM {affix}_buffer_0
WHERE TRUE;

INSERT INTO {affix}_buffer_0 (SELECT *
FROM {affix}_0
WHERE new_);

INSERT INTO {affix}_buffer_1 (SELECT id,
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
       2*PI()*COS(2*PI()*latitude/360)*longitude/360*13500 AS pos_0,
       2*PI()*latitude/360*13500 AS pos_1 ,
       TRUE AS new_
FROM {affix}_buffer_0);

MERGE INTO {affix}_1 t0
USING {affix}_buffer_1 t1 ON t0.station_id = t1.station_id AND t0.update_timestamp=t1.update_timestamp
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

DELETE FROM {affix}_buffer_1
WHERE TRUE;

INSERT INTO {affix}_buffer_1 (SELECT *
FROM {affix}_1
WHERE new_);

INSERT INTO {affix}_download_timestamps_buffer_0 (SELECT DISTINCT download_timestamp,
       TRUE AS new_
FROM {affix}_buffer_0
);

MERGE INTO {affix}_download_timestamps_0 t0
USING {affix}_download_timestamps_buffer_0 t1 ON t0.download_timestamp = t1.download_timestamp
WHEN NOT MATCHED THEN
  INSERT VALUES
(
  t1.download_timestamp,t1.new_
);

DELETE FROM {affix}_download_timestamps_buffer_0
WHERE TRUE;

INSERT INTO {affix}_download_timestamps_buffer_0 (SELECT *
FROM {affix}_download_timestamps_0
WHERE new_);

INSERT INTO {affix}_update_timestamps_buffer_0 (SELECT station_id,
       MAX(update_timestamp) AS update_timestamp,
       TRUE AS new_
FROM {affix}_buffer_0
GROUP BY station_id
);

MERGE INTO {affix}_update_timestamps_0 t0
USING {affix}_update_timestamps_buffer_0 t1 ON t0.station_id = t1.station_id
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

DELETE FROM {affix}_update_timestamps_buffer_0
WHERE TRUE;

INSERT INTO {affix}_update_timestamps_buffer_0 (SELECT *
FROM {affix}_update_timestamps_0
WHERE new_);

INSERT INTO {affix}_stations_buffer_0 (SELECT station_id,
       MIN(station_code) AS station_code,
       MIN(station_name) AS station_name,
       MIN(latitude) AS latitude,
       MIN(longitude) AS longitude,
       MIN(pos_0) AS pos_0,
       MIN(pos_1) AS pos_1,
       TRUE AS new_
FROM {affix}_buffer_1
GROUP BY station_id);

MERGE INTO {affix}_stations_0 t0
USING {affix}_stations_buffer_0 t1 ON t0.station_id = t1.station_id
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

DELETE FROM {affix}_stations_buffer_0
WHERE TRUE;

INSERT INTO {affix}_stations_buffer_0 (SELECT *
FROM {affix}_stations_0
WHERE new_);

INSERT INTO {affix}_neighbors_buffer_0 (SELECT MD5(CONCAT(t0.station_id,t1.station_id)) AS neighborhood_id,
       t0.station_id,
       t1.station_id AS neighbor_id,
       1 AS one,
       POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5) AS distance,
       TRUE AS new_
FROM {affix}_stations_buffer_0 t0
  JOIN {affix}_stations_buffer_0 t1
    ON (POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5)) < 1
   AND t0.station_id != t1.station_id
UNION ALL
SELECT MD5(CONCAT(t0.station_id,t1.station_id)) AS neighborhood_id,
       t0.station_id,
       t1.station_id AS neighbor_id,
       1 AS one,
       POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5) AS distance,
       TRUE AS new_
FROM {affix}_stations_buffer_0 t0
  JOIN {affix}_stations_0 t1
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
FROM {affix}_stations_0 t0
  JOIN {affix}_stations_buffer_0 t1
    ON (POWER(POWER(t0.pos_0 - t1.pos_0,2) + POWER(t0.pos_1 - t1.pos_1,2),0.5)) < 1
   AND t0.station_id != t1.station_id
   AND NOT(t0.new_));   
   
MERGE INTO {affix}_neighbors_0 t0
USING {affix}_neighbors_buffer_0 t1 ON t0.neighborhood_id = t1.neighborhood_id AND t0.station_id = t1.station_id AND t0.neighbor_id = t1.neighbor_id
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

DELETE FROM {affix}_neighbors_buffer_0
WHERE TRUE;

INSERT INTO {affix}_neighbors_buffer_0 (SELECT *
FROM {affix}_neighbors_0
WHERE new_);

INSERT INTO {affix}_pairs_buffer_0 (
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
       t0.ebikes AS current_ebikes ,
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
FROM {affix}_buffer_1 t0
  JOIN {affix}_buffer_1 t1
    ON t0.station_id = t1.station_id
    AND t0.update_timestamp > t1.update_timestamp
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
       t0.ebikes AS current_ebikes ,
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
FROM {affix}_buffer_1 t0
  JOIN {affix}_1 t1
    ON t0.station_id = t1.station_id
    AND t0.update_timestamp > t1.update_timestamp
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
       t0.ebikes AS current_ebikes ,
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
FROM {affix}_1 t0
  JOIN {affix}_buffer_1 t1
    ON t0.station_id = t1.station_id
    AND t0.update_timestamp > t1.update_timestamp
   AND EXTRACT (epoch FROM t0.update_timestamp - t1.update_timestamp) < 900
   AND NOT(t0.new_)
);

   
MERGE INTO {affix}_pairs_0 t0
USING {affix}_pairs_buffer_0 t1 ON t0.pair_id = t1.pair_id AND t0.id = t1.id AND t0.station_id = t1.station_id
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

DELETE FROM {affix}_pairs_buffer_0
WHERE TRUE;

INSERT INTO {affix}_pairs_buffer_0 (SELECT *
FROM {affix}_pairs_0
WHERE new_);

INSERT INTO {affix}_buffer_2 (
SELECT id,
       station_id,
       MIN(diff_update_timestamp) AS diff_update_timestamp,
       TRUE AS new_
FROM {affix}_pairs_buffer_0
GROUP BY id,
         station_id
);

MERGE INTO {affix}_2 t0
USING {affix}_buffer_2 t1 ON t0.id = t1.id AND t0.station_id = t1.station_id
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

DELETE FROM {affix}_buffer_2
WHERE TRUE;

INSERT INTO {affix}_buffer_2 (SELECT *
FROM {affix}_2
WHERE new_);

INSERT INTO {affix}_spells_buffer_0 (
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
FROM {affix}_pairs_buffer_0 t0
  JOIN {affix}_buffer_2 t1
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
FROM {affix}_pairs_buffer_0 t0
  JOIN {affix}_2 t1
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
FROM {affix}_pairs_0 t0
  JOIN {affix}_buffer_2 t1
    ON t0.id = t1.id
   AND t0.station_id = t1.station_id
   AND t0.diff_update_timestamp = t1.diff_update_timestamp
   AND NOT(t0.new_));

MERGE INTO {affix}_spells_0 t0
USING {affix}_spells_buffer_0 t1 ON t0.id = t1.id AND t0.station_id = t1.station_id
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
FROM {affix}_spells_buffer_0
WHERE TRUE;

INSERT INTO {affix}_spells_buffer_0
(SELECT*FROM {affix}_spells_0 WHERE new_);

