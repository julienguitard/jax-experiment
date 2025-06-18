--Update new_ status

DELETE FROM {affix}_buffer_0
WHERE TRUE;

DELETE FROM {affix}_buffer_1
WHERE TRUE;

DELETE FROM {affix}_dowload_timestamps_buffer_0
WHERE TRUE;

DELETE FROM {affix}_update_timestamps_buffer_0
WHERE TRUE;

DELETE FROM {affix}_stations_buffer_0
WHERE TRUE;

DELETE FROM {affix}_neighbors_buffer_0
WHERE TRUE;
   
DELETE FROM {affix}_pairs_buffer_0
WHERE TRUE;
   
DELETE FROM {affix}_buffer_2
WHERE TRUE;
   
DELETE FROM {affix}_spells_buffer_0
WHERE TRUE;

UPDATE {affix}_0
   SET new_ = FALSE;

UPDATE {affix}_1
   SET new_ = FALSE;

UPDATE {affix}_dowload_timestamps_0
   SET new_ = FALSE;
   
UPDATE {affix}_update_timestamps_0
   SET new_ = FALSE;

UPDATE {affix}_stations_0
   SET new_ = FALSE;

UPDATE {affix}_neighbors_0
   SET new_ = FALSE;
   
UPDATE {affix}_pairs_0
   SET new_ = FALSE;
   
UPDATE {affix}_2
   SET new_ = FALSE;
   
UPDATE {affix}_spells_0
   SET new_ = FALSE;
