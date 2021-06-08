from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import coref_model as cm
import util
import io


if __name__ == "__main__":
  config = util.initialize_from_env()

  # Input file in .jsonlines format.
  input_filename = sys.argv[2]
  
  print("input file is in {}".format(sys.argv[2]))
  # Predictions will be written to this file in .jsonlines format.
  output_filename = sys.argv[3]

  model = cm.CorefModel(config)

  with tf.Session() as session:
    model.restore(session)

    with io.open(output_filename, "w", encoding='utf-8') as output_file:
      with io.open(input_filename, encoding='utf-8') as input_file:
        for example_num, line in enumerate(input_file.readlines()):
          example =json.loads(u'{}'.format(line))
          tensorized_example = model.tensorize_example(example, is_training=False)
	  print("cluster_ids {} \n head_we[0,20] {} \n speaker_ids is {} \n gold starts is {}".format(tensorized_example[11],tensorized_example[2][0,0],tensorized_example[6],tensorized_example[9]))
          #feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
          #_, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)
          #predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
          #example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)

          #output_file.write(unicode(json.dumps(example, ensure_ascii=False, encoding='utf8')))
          #output_file.write(u"\n")
          #if example_num % 100 == 0:
          #  print("Decoded {} examples.".format(example_num + 1))
