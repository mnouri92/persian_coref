#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
import coref_model as cm
import util

export_path = sys.argv[2]
model_version = sys.argv[3]
export_dir = os.path.join(export_path,model_version)

if __name__ == "__main__":
  config = util.initialize_from_env()

  model = cm.CorefModel(config)
  with tf.Session() as session:
    model.restore(session)
  
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    tokens_inputs_info              = tf.saved_model.utils.build_tensor_info(model.queue_input_tensors[0])
    context_embedding_inputs_info   = tf.saved_model.utils.build_tensor_info(model.queue_input_tensors[1])
    head_embedding_inputs_info      = tf.saved_model.utils.build_tensor_info(model.queue_input_tensors[2])
    lm_embedding_inputs_info        = tf.saved_model.utils.build_tensor_info(model.queue_input_tensors[3])
    char_index_inputs_info          = tf.saved_model.utils.build_tensor_info(model.queue_input_tensors[4])
    text_len_inputs_info            = tf.saved_model.utils.build_tensor_info(model.queue_input_tensors[5])
    speaker_id_inputs_info          = tf.saved_model.utils.build_tensor_info(model.queue_input_tensors[6])
    genre_inputs_info               = tf.saved_model.utils.build_tensor_info(model.queue_input_tensors[7])
    is_training_inputs_info         = tf.saved_model.utils.build_tensor_info(model.queue_input_tensors[8])
    gold_starts_inputs_info         = tf.saved_model.utils.build_tensor_info(model.queue_input_tensors[9])
    gold_ends_inputs_info           = tf.saved_model.utils.build_tensor_info(model.queue_input_tensors[10])
    cluster_ids_inputs_info         = tf.saved_model.utils.build_tensor_info(model.queue_input_tensors[11])

    top_span_starts_outputs_info          = tf.saved_model.utils.build_tensor_info(model.predictions[3])
    top_span_ends_outputs_info             = tf.saved_model.utils.build_tensor_info(model.predictions[4])
    top_antecedents_outputs_info          = tf.saved_model.utils.build_tensor_info(model.predictions[5])
    top_antecedents_scores_outputs_info   = tf.saved_model.utils.build_tensor_info(model.predictions[6])
    prediction_signature = (
		tf.saved_model.signature_def_utils.build_signature_def(
			inputs={'tokens':tokens_inputs_info, 'context_embeddings':context_embedding_inputs_info, 'head_embeddings':head_embedding_inputs_info, 'lm_embeddings':lm_embedding_inputs_info, 'char_indexes':char_index_inputs_info, 'text_lengths':text_len_inputs_info, 'speaker_ids':speaker_id_inputs_info, 'genres':genre_inputs_info, 'is_training':is_training_inputs_info, 'gold_starts':gold_starts_inputs_info, 'gold_ends':gold_ends_inputs_info, 'cluster_ids':cluster_ids_inputs_info},
			outputs={'top_span_starts':top_span_starts_outputs_info, 'top_span_ends':top_span_ends_outputs_info, 'top_antecedents':top_antecedents_outputs_info, 'top_antecedents_scores':top_antecedents_scores_outputs_info},
			method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
		)
	)

    legacy_init_op = tf.group(tf.tables_initializer(),name='legacy_init_op')
    builder.add_meta_graph_and_variables(
		session, [tf.saved_model.tag_constants.SERVING],
		signature_def_map={
			'serving_default':prediction_signature
		},
		legacy_init_op=legacy_init_op)

  builder.save()
