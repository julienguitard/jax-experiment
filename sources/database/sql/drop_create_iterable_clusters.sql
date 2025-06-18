DROP TABLE IF EXISTS iterable_{affix}_spells_0;

CREATE TABLE iterable_{affix}_spells_0 
AS
(SELECT *, FLOOR((ROW_NUMBER() OVER(ORDER BY RANDOM()))/{batch_size}) AS iterand
FROM (SELECT *
      FROM {affix}_spells_0
        CROSS JOIN (SELECT COUNT(*) AS size FROM {affix}_spells_0)));