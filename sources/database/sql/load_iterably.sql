SELECT normalized_p_0,
       normalized_p_1,
       normalized_dow,
       normalized_hour,
       normalized_minute,
       (duf -(MIN(duf) OVER ())) /((MAX(duf) OVER ()) -(MIN(duf) OVER ())) AS normalized_duf,
       -- TO DO, EXTEND NORMALIZATION BEYOND BATCH
       (dad -(MIN(dad) OVER ())) /((MAX(dad) OVER ()) -(MIN(dad) OVER ())) AS normalized_dad,
       (dm -(MIN(dm) OVER ())) /((MAX(dm) OVER ()) -(MIN(dm) OVER ())) AS normalized_dm,
       (de -(MIN(de) OVER ())) /((MAX(de) OVER ()) -(MIN(de) OVER ())) AS normalized_de
FROM (SELECT (CAST(pos_0 AS FLOAT) - 335.0) / 58.0 AS normalized_p_0,
             (CAST(pos_1 AS FLOAT) -11489.0) / 44.0 AS normalized_p_1,
             CAST(EXTRACT(dow FROM current_update_timestamp) AS FLOAT) / 7.0 AS normalized_dow,
             CAST(EXTRACT(HOUR FROM current_update_timestamp) AS FLOAT) / 24.0 AS normalized_hour,
             CAST(EXTRACT(MINUTE FROM current_update_timestamp) AS FLOAT) / 60.0 AS normalized_minute,
             CAST(EXTRACT(EPOCH FROM diff_update_timestamp) AS FLOAT) AS duf,
             CAST(diff_available_docks AS FLOAT) / CAST(EXTRACT(EPOCH FROM diff_update_timestamp) AS FLOAT) AS dad,
             CAST(diff_mechanicals AS FLOAT) / CAST(EXTRACT(EPOCH FROM diff_update_timestamp) AS FLOAT) AS dm,
             CAST(diff_ebikes AS FLOAT) / CAST(EXTRACT(EPOCH FROM diff_update_timestamp) AS FLOAT) AS de
      FROM iterable_{affix}_spells_0
      WHERE iterand = {iterand});

