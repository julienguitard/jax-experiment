--Drop then create buffer tables
DROP TABLE IF EXISTS {affix}_buffer_from_csv;

DROP TABLE IF EXISTS {affix}_buffer_0;

DROP TABLE IF EXISTS {affix}_buffer_1;

DROP TABLE IF EXISTS {affix}_download_timestamps_buffer_0;

DROP TABLE IF EXISTS {affix}_update_timestamps_buffer_0;

DROP TABLE IF EXISTS {affix}_stations_buffer_0; 

DROP TABLE IF EXISTS {affix}_neighbors_buffer_0; 

DROP TABLE IF EXISTS {affix}_pairs_buffer_0; 

DROP TABLE IF EXISTS {affix}_buffer_2; 

DROP TABLE IF EXISTS {affix}_spells_buffer_0;

CREATE TABLE {affix}_buffer_from_csv (
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

CREATE TABLE {affix}_buffer_0 (
id VARCHAR(255) PRIMARY KEY,
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

CREATE TABLE {affix}_buffer_1 (
id VARCHAR(255) PRIMARY KEY,
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

CREATE TABLE {affix}_download_timestamps_buffer_0 (
download_timestamp TIMESTAMP PRIMARY KEY,
new_ BOOLEAN
);

CREATE TABLE {affix}_update_timestamps_buffer_0 (
station_id VARCHAR(255) PRIMARY KEY,
update_timestamp TIMESTAMP,
new_ BOOLEAN
);

CREATE TABLE {affix}_stations_buffer_0 (
station_id VARCHAR(255) PRIMARY KEY,
station_code VARCHAR(255),
station_name VARCHAR(255),
latitude NUMERIC,
longitude NUMERIC,
pos_0 NUMERIC,
pos_1 NUMERIC,
new_ BOOLEAN
);

CREATE TABLE {affix}_neighbors_buffer_0 (
neighborhood_id VARCHAR(255) PRIMARY KEY, 
station_id VARCHAR(255),
neighbor_id VARCHAR(255),
one NUMERIC,
distance NUMERIC,
new_ BOOLEAN
);

CREATE TABLE {affix}_pairs_buffer_0 (
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

CREATE TABLE {affix}_buffer_2 (
id VARCHAR(255) PRIMARY KEY,
station_id VARCHAR(255),
diff_update_timestamp INTERVAL,
new_ BOOLEAN
);

CREATE TABLE {affix}_spells_buffer_0 (
id VARCHAR(255) PRIMARY KEY,
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