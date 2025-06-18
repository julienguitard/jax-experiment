-- Drop main tables

DROP TABLE IF EXISTS {affix}_0; --basic data

DROP TABLE IF EXISTS {affix}_1; --basic data and coordinate

DROP TABLE IF EXISTS {affix}_download_timestamps_0; -- distinct download times

DROP TABLE IF EXISTS {affix}_update_timestamps_0; -- latest update timestamp

DROP TABLE IF EXISTS {affix}_stations_0; -- distinct stations

DROP TABLE IF EXISTS {affix}_neighbors_0;  -- neighbors

DROP TABLE IF EXISTS {affix}_pairs_0; --pairs time of data for each station

DROP TABLE IF EXISTS {affix}_2; -- closest time 

DROP TABLE IF EXISTS {affix}_spells_0; -- spells as closest time pair 