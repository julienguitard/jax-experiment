--Create main tables

CREATE TABLE {affix}_0 (
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

CREATE TABLE {affix}_1 (
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

CREATE TABLE {affix}_download_timestamps_0 (
download_timestamp TIMESTAMP PRIMARY KEY,
new_ BOOLEAN
);

CREATE TABLE {affix}_update_timestamps_0 (
station_id VARCHAR(255) PRIMARY KEY ,
update_timestamp TIMESTAMP,
new_ BOOLEAN
);

CREATE TABLE {affix}_stations_0 (
station_id VARCHAR(255) PRIMARY KEY ,
station_code VARCHAR(255),
station_name VARCHAR(255),
latitude NUMERIC,
longitude NUMERIC,
pos_0 NUMERIC,
pos_1 NUMERIC,
new_ BOOLEAN
);

CREATE TABLE {affix}_stations_0 (
station_id VARCHAR(255) PRIMARY KEY ,
station_code VARCHAR(255),
station_name VARCHAR(255),
latitude NUMERIC,
longitude NUMERIC,
pos_0 NUMERIC,
pos_1 NUMERIC,
new_ BOOLEAN
);


CREATE TABLE {affix}_neighbors_0 (
neighborhood_id VARCHAR(255) PRIMARY KEY, 
station_id VARCHAR(255),
neighbor_id VARCHAR(255),
one NUMERIC,
distance NUMERIC,
new_ BOOLEAN
);

CREATE TABLE {affix}_pairs_0 (
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

CREATE TABLE {affix}_2 (
id VARCHAR(255) PRIMARY KEY ,
station_id VARCHAR(255),
diff_update_timestamp INTERVAL,
new_ BOOLEAN
);

CREATE TABLE {affix}_spells_0 (
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