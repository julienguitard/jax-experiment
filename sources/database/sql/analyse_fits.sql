DROP TABLE IF EXISTS  {affix}_temp_json;

CREATE TABLE {affix}_temp_json 
(
  state   JSONB
);

COPY {affix}_temp_json
FROM '/Users/julienguitard/local_data/velib/fits/logs.json';

DROP TABLE IF EXISTS  {affix}_fits_0;

CREATE TABLE {affix}_fits_0 
AS
(SELECT CAST(state['epoch'] AS INT) AS epoch,
       CAST(state['load_0'] AS INT) AS load_0,
       CAST(state['load_1'] AS INT) AS load_1,
       CAST(state['batch'] AS INT) AS batch,
       state['max_weights'] AS max_weights,
       state['max_gradients'] AS max_gradients,
       state['max_moments'] AS max_moments,
       state['weights'] AS weights,
       state['moments'] AS moments,
       state['x'] AS x,
       state['y'] AS y,
       state['model'] AS model,
       state['loss'] AS loss,
       state['penalized_loss'] AS penalized_loss,
       state['gradients'] AS gradients
FROM (SELECT state FROM {affix}_temp_json));

DROP TABLE IF EXISTS  {affix}_weights_0;

CREATE TABLE {affix}_weights_0 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       weight_row,
       ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,weight_row) AS column_row,
       sub_weight
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             weight_row,
             JSONB_ARRAY_ELEMENTS(weight) AS sub_weight
      FROM (SELECT epoch,
                   load_0,
                   load_1,
                   batch,
                   layer_row,
                   ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row) AS weight_row,
                   weights AS weight
            FROM (SELECT epoch,
                         load_0,
                         load_1,
                         batch,
                         ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch) AS layer_row,
                         JSONB_ARRAY_ELEMENTS(weights) AS weights
                  FROM (SELECT epoch,
                               load_0,
                               load_1,
                               batch,
                               JSONB_ARRAY_ELEMENTS(weights) AS weights
                        FROM {affix}_fits_0)))));

DROP TABLE IF EXISTS  {affix}_weights_1;

CREATE TABLE {affix}_weights_1 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       weight_row,
       column_row,
       ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,weight_row,column_row) AS next_column_row,
       CAST(coef AS NUMERIC) AS coef
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             weight_row,
             column_row,
             JSONB_ARRAY_ELEMENTS(sub_weight) AS coef
      FROM {affix}_weights_0
      WHERE JSONB_TYPEOF(sub_weight) = 'array')
UNION ALL
SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       weight_row,
       column_row,
       0 AS next_column_row,
       CAST(coef AS NUMERIC) AS coef
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             weight_row,
             column_row,
             sub_weight AS coef
      FROM {affix}_weights_0
      WHERE JSONB_TYPEOF(sub_weight) = 'number'));

DROP TABLE IF EXISTS  {affix}_gradients_0;

CREATE TABLE {affix}_gradients_0 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       weight_row,
       ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,weight_row) AS column_row,
       sub_gradient
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             weight_row,
             JSONB_ARRAY_ELEMENTS(gradient) AS sub_gradient
      FROM (SELECT epoch,
                   load_0,
                   load_1,
                   batch,
                   layer_row,
                   ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row) AS weight_row,
                   gradients AS gradient
            FROM (SELECT epoch,
                         load_0,
                         load_1,
                         batch,
                         ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch) AS layer_row,
                         JSONB_ARRAY_ELEMENTS(gradients) AS gradients
                  FROM (SELECT epoch,
                               load_0,
                               load_1,
                               batch,
                               JSONB_ARRAY_ELEMENTS(gradients) AS gradients
                        FROM {affix}_fits_0)))));

DROP TABLE IF EXISTS  {affix}_gradients_1;

CREATE TABLE {affix}_gradients_1 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       weight_row,
       column_row,
       ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,weight_row,column_row) AS next_column_row,
       CAST(coef AS NUMERIC) AS coef
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             weight_row,
             column_row,
             JSONB_ARRAY_ELEMENTS(sub_gradient) AS coef
      FROM {affix}_gradients_0
      WHERE JSONB_TYPEOF(sub_gradient) = 'array')
UNION ALL
SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       weight_row,
       column_row,
       0 AS next_column_row,
       CAST(coef AS NUMERIC) AS coef
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             weight_row,
             column_row,
             sub_gradient AS coef
      FROM {affix}_gradients_0
      WHERE JSONB_TYPEOF(sub_gradient) = 'number'));

DROP TABLE IF EXISTS  {affix}_moments_0;

CREATE TABLE {affix}_moments_0 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       weight_row,
       sub_moment_row,
       ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,weight_row,sub_moment_row) AS column_row,
       sub_moment
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             weight_row,
             sub_moment_row,
             JSONB_ARRAY_ELEMENTS(sub_moment) AS sub_moment
      FROM (SELECT epoch,
                   load_0,
                   load_1,
                   batch,
                   layer_row,
                   weight_row,
                   ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,weight_row) AS sub_moment_row,
                   sub_moment
            FROM (SELECT epoch,
                         load_0,
                         load_1,
                         batch,
                         layer_row,
                         weight_row,
                         JSONB_ARRAY_ELEMENTS(moment) AS sub_moment
                  FROM (SELECT epoch,
                               load_0,
                               load_1,
                               batch,
                               layer_row,
                               ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row) AS weight_row,
                               moments AS moment
                        FROM (SELECT epoch,
                                     load_0,
                                     load_1,
                                     batch,
                                     ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch) AS layer_row,
                                     JSONB_ARRAY_ELEMENTS(moments) AS moments
                              FROM (SELECT epoch,
                                           load_0,
                                           load_1,
                                           batch,
                                           JSONB_ARRAY_ELEMENTS(moments) AS moments
                                    FROM {affix}_fits_0)))))));

DROP TABLE IF EXISTS  {affix}_moments_1;

CREATE TABLE {affix}_moments_1 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       weight_row,
       sub_moment_row,
       column_row,
       ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,weight_row,sub_moment_row,column_row) AS next_column_row,
       CAST(coef AS NUMERIC) AS coef
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             weight_row,
             sub_moment_row,
             column_row,
             JSONB_ARRAY_ELEMENTS(sub_moment) AS coef
      FROM {affix}_moments_0
      WHERE JSONB_TYPEOF(sub_moment) = 'array')
UNION ALL
SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       weight_row,
       sub_moment_row,
       column_row,
       0 AS next_column_row,
       CAST(coef AS NUMERIC) AS coef
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             weight_row,
             sub_moment_row,
             column_row,
             sub_moment AS coef
      FROM {affix}_moments_0
      WHERE JSONB_TYPEOF(sub_moment) = 'number'));

DROP TABLE IF EXISTS  {affix}_x_0;

CREATE TABLE {affix}_x_0 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       sample_row,
       ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,sample_row) AS column_row,
       CAST(x AS NUMERIC) AS x
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch) AS sample_row,
             JSONB_ARRAY_ELEMENTS(x) AS x
      FROM (SELECT epoch,
                   load_0,
                   load_1,
                   batch,
                   JSONB_ARRAY_ELEMENTS(x) AS x
            FROM {affix}_fits_0)));

DROP TABLE IF EXISTS  {affix}_y_0;

CREATE TABLE {affix}_y_0 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       sample_row,
       ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,sample_row) AS column_row,
       CAST(y AS NUMERIC) AS y
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch) AS sample_row,
             JSONB_ARRAY_ELEMENTS(y) AS y
      FROM (SELECT epoch,
                   load_0,
                   load_1,
                   batch,
                   JSONB_ARRAY_ELEMENTS(y) AS y
            FROM {affix}_fits_0)));

DROP TABLE IF EXISTS  {affix}_models_0;

CREATE TABLE {affix}_models_0 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row) AS node_row,
       node AS node
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch) - 1 AS layer_row,
             JSONB_ARRAY_ELEMENTS(layer_node) AS node
      FROM (SELECT epoch,
                   load_0,
                   load_1,
                   batch,
                   JSONB_ARRAY_ELEMENTS(model) AS layer_node
            FROM {affix}_fits_0)));

DROP TABLE IF EXISTS  {affix}_models_outputs_0;

CREATE TABLE {affix}_models_outputs_0 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       node_row,
       sub_node_row,
       ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,node_row,sub_node_row) AS mixed_row,
       mixed_
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             node_row,
             ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,node_row) AS sub_node_row,
             JSONB_ARRAY_ELEMENTS(sub_node) AS mixed_
      FROM (SELECT epoch,
                   load_0,
                   load_1,
                   batch,
                   layer_row,
                   node_row,
                   JSONB_ARRAY_ELEMENTS(node) AS sub_node
            FROM {affix}_models_0
            WHERE node_row = 1)));

DROP TABLE IF EXISTS  {affix}_models_outputs_1;

CREATE TABLE {affix}_models_outputs_1 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       node_row,
       sub_node_row,
       sample_row,
       ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,node_row,sub_node_row,sample_row) AS column_row,
       coef
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             node_row,
             sub_node_row,
             mixed_row AS sample_row,
             JSONB_ARRAY_ELEMENTS(mixed_) AS coef
      FROM {affix}_models_outputs_0
      WHERE JSONB_TYPEOF(mixed_) = 'array')
UNION ALL
SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       node_row,
       sub_node_row,
       0 AS sample_row,
       mixed_row AS column_row,
       mixed_ AS coef
FROM {affix}_models_outputs_0
WHERE JSONB_TYPEOF(mixed_) = 'number');

DROP TABLE IF EXISTS  {affix}_models_penalizations_0;

CREATE TABLE {affix}_models_penalizations_0 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       node_row,
       0 AS sample_row,
       0 AS column_row,
       node AS coef
FROM {affix}_models_0
WHERE node_row = 2);

DROP TABLE IF EXISTS  {affix}_models_grad_like_args_0;

CREATE TABLE {affix}_models_grad_like_args_0 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       node_row,
       sub_node_row,
       ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,node_row,sub_node_row) AS mixed_row,
       mixed_
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             node_row,
             ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,node_row) AS sub_node_row,
             JSONB_ARRAY_ELEMENTS(sub_node) AS mixed_
      FROM (SELECT epoch,
                   load_0,
                   load_1,
                   batch,
                   layer_row,
                   node_row,
                   JSONB_ARRAY_ELEMENTS(node) AS sub_node
            FROM {affix}_models_0
            WHERE node_row = 3)));

DROP TABLE IF EXISTS  {affix}_models_grad_like_args_1;

CREATE TABLE {affix}_models_grad_like_args_1 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       node_row,
       sub_node_row,
       sample_row,
       ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,node_row,sub_node_row,sample_row) AS column_row,
       coef
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             node_row,
             sub_node_row,
             mixed_row AS sample_row,
             JSONB_ARRAY_ELEMENTS(mixed_) AS coef
      FROM {affix}_models_grad_like_args_0
      WHERE JSONB_TYPEOF(mixed_) = 'array'));

DROP TABLE IF EXISTS  {affix}_models_grad_likes_0;

CREATE TABLE {affix}_models_grad_likes_0 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       node_row,
       weight_row,
       ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,node_row,weight_row) AS column_row,
       grad_like
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             node_row,
             ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,node_row) AS weight_row,
             JSONB_ARRAY_ELEMENTS(grad_like) AS grad_like
      FROM (SELECT epoch,
                   load_0,
                   load_1,
                   batch,
                   layer_row,
                   node_row,
                   JSONB_ARRAY_ELEMENTS(node) AS grad_like
            FROM {affix}_models_0
            WHERE node_row = 4)));
            
DROP TABLE IF EXISTS  {affix}_models_grad_likes_1;

CREATE TABLE {affix}_models_grad_likes_1 
AS
(SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       weight_row,
       column_row,
       ROW_NUMBER() OVER (PARTITION BY epoch,load_0,load_1,batch,layer_row,weight_row,column_row) AS next_column_row,
       CAST(coef AS NUMERIC) AS coef
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             weight_row,
            column_row,
             JSONB_ARRAY_ELEMENTS(grad_like) AS coef
      FROM {affix}_models_grad_likes_0
      WHERE JSONB_TYPEOF(grad_like) = 'array')
UNION ALL
SELECT epoch,
       load_0,
       load_1,
       batch,
       layer_row,
       weight_row,
       column_row,
       0 AS next_column_row,
       CAST(coef AS NUMERIC) AS coef
FROM (SELECT epoch,
             load_0,
             load_1,
             batch,
             layer_row,
             weight_row,
             column_row,
             grad_like AS coef
      FROM {affix}_models_grad_likes_0
      WHERE JSONB_TYPEOF(grad_like) = 'number'));