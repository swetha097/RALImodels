# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""tf.data.Dataset builder.

Creates data sources for DetectionModels from an InputReader config. See
input_reader.proto for options.

Note: If users wishes to also use their own InputReaders with the Object
Detection configuration framework, they should define their own builder function
that wraps the build function.
"""
import functools
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import multiprocessing as mp
# HASH_KEY = 'hash'
# HASH_BINS = 1 << 31

# tf.enable_eager_execution()
from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import input_reader_pb2


def make_initializable_iterator(dataset):
  """Creates an iterator, and initializes tables.

  This is useful in cases where make_one_shot_iterator wouldn't work because
  the graph contains a hash table that needs to be initialized.

  Args:
    dataset: A `tf.data.Dataset` object.

  Returns:
    A `tf.data.Iterator`.
  """
  iterator = dataset.make_initializable_iterator()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  return iterator


def read_dataset(file_read_func, input_files, config):
  """Reads a dataset, and handles repetition and shuffling.

  Args:
    file_read_func: Function to use in tf.contrib.data.parallel_interleave, to
      read every individual file into a tf.data.Dataset.
    input_files: A list of file paths to read.
    config: A input_reader_builder.InputReader object.

  Returns:
    A tf.data.Dataset of (undecoded) tf-records based on config.
  """
  # Shard, shuffle, and read files.
  filenames = tf.gfile.Glob(input_files)
  if not filenames:
      raise ValueError('Invalid input path specified in '
                       '`input_reader_config`.')
  num_readers = config.num_readers
  if num_readers > len(filenames):
    num_readers = len(filenames)
    tf.logging.warning('num_readers has been reduced to %d to match input file '
                       'shards.' % num_readers)
  filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
  if config.shuffle:
    filename_dataset = filename_dataset.shuffle(
        config.filenames_shuffle_buffer_size)
  elif num_readers > 1:
    tf.logging.warning('`shuffle` is false, but the input data stream is '
                       'still slightly shuffled since `num_readers` > 1.')
  filename_dataset = filename_dataset.repeat(config.num_epochs or None)
  records_dataset = filename_dataset.apply(
      tf.contrib.data.parallel_interleave(
          file_read_func,
          cycle_length=num_readers,
          block_length=config.read_block_length,
          sloppy=config.shuffle))
  if config.shuffle:
    records_dataset = records_dataset.shuffle(config.shuffle_buffer_size)
  return records_dataset

def get_onehot(image_labels_array, numClasses):
	one_hot_vector_list = []
	for label in image_labels_array:
		one_hot_vector = np.zeros(numClasses)
		if label[0] != 0:
			np.put(one_hot_vector, label[0] - 1, 1)
		one_hot_vector_list.append(one_hot_vector)
	one_hot_vector_array = np.array(one_hot_vector_list)

	return one_hot_vector_array

def get_weights(num_bboxes):
	weights_array = np.zeros(100)
	for pos in list(range(num_bboxes)):
		np.put(weights_array, pos, 1)
	
	return weights_array

def get_shapes(image_array):
  
  return len(image_array), len(image_array[0]), len(image_array[0,0])

def get_normalized(image_bboxes, image_height, image_width):
  image_bboxes_normalized = np.empty([0, 4], dtype = np.float32)
  for element in image_bboxes:
    image_bboxes_normalized = np.append(image_bboxes_normalized, np.array([
      [
        element[1] / image_height,
        element[0] / image_width,
        (element[1] + element[3]) / image_height,
        (element[0] + element[2]) / image_width
      ]
    ], dtype = np.float32), axis = 0)
  
  return image_bboxes_normalized

def rali_parallelized_tensor_generator(data_tuple):
  image = data_tuple[0]
  bboxes_in_image = data_tuple[1]
  labels_in_image = data_tuple[2]
  num_bboxes_in_image = data_tuple[3]
  unique_key_for_image = data_tuple[4]
  numClasses = data_tuple[5]

  hash_key_tensor = str(unique_key_for_image)
  images_tensor = image
  num_groundtruth_boxes_tensor = num_bboxes_in_image
  true_image_shapes_tensor = get_shapes(image)
  groundtruth_boxes_tensor = get_normalized(bboxes_in_image, len(image), len(image[0]))
  groundtruth_classes_tensor = get_onehot(labels_in_image, numClasses)
  groundtruth_weights_tensor = get_weights(num_bboxes_in_image)

  return [hash_key_tensor, 
  images_tensor, 
  num_groundtruth_boxes_tensor, 
  true_image_shapes_tensor, 
  groundtruth_boxes_tensor, 
  groundtruth_classes_tensor, 
  groundtruth_weights_tensor]

def rali_processed_tensors_generator(rali_batch_size, num_classes, enum):
  
  # global_step = tf.train.get_global_step()

  result = []
  hash_key_tensor = []
  images_tensor = []
  num_groundtruth_boxes_tensor = []
  true_image_shapes_tensor = []
  groundtruth_boxes_tensor = []
  groundtruth_classes_tensor = []
  groundtruth_weights_tensor = []

  pool = mp.Pool(mp.cpu_count())
  # sourceID = 1000000 + (global_step * batch_size)

  flag = True

  while flag == True:
    try:
      i, (images_array, bboxes_array, labels_array, num_bboxes_array) = enum.__next__()
    except:
      break

    print("RALI augmentation pipeline - Processing RALI batch %d....." % i)
    
    sourceID = 1000000 + (i * rali_batch_size)
    
    images_array = np.transpose(images_array, [0, 2, 3, 1])

    result = pool.map(rali_parallelized_tensor_generator, np.array([
        [images_array[element], bboxes_array[element], labels_array[element], num_bboxes_array[element], sourceID + element, num_classes] for element in list(range(rali_batch_size))
        ]))

    for image_data in result:
      hash_key_tensor.append(image_data[0])
      images_tensor.append(image_data[1])
      num_groundtruth_boxes_tensor.append(image_data[2])
      true_image_shapes_tensor.append(image_data[3])
      groundtruth_boxes_tensor.append(image_data[4])
      groundtruth_classes_tensor.append(image_data[5])
      groundtruth_weights_tensor.append(image_data[6])
    
    flag = not(i % int(1500 / rali_batch_size) == 0)

  pool.close()




  
  # FOR THE "CALLING rali_processed_tensors_generator() VERSION"

  features_dict = {
    "image" : np.array(images_tensor, dtype=np.float32),
    "hash" : np.array(hash_key_tensor),
    "true_image_shape" : np.array(true_image_shapes_tensor, dtype=np.int32)
  }
  labels_dict = {
    "num_groundtruth_boxes" : np.array(num_groundtruth_boxes_tensor, dtype=np.int32),
    "groundtruth_boxes" : np.array(groundtruth_boxes_tensor, dtype=np.float32),
    "groundtruth_classes" : np.array(groundtruth_classes_tensor, dtype=np.float32),
    "groundtruth_weights" : np.array(groundtruth_weights_tensor, dtype=np.float32)
  }

  processed_tensors = (features_dict, labels_dict)

  

  # FOR THE "WITH INITIALIZABLE ITERATOR VERSION"

  # processed_tensors = (
  #   np.array(images_tensor, dtype=np.float32),
  #   np.array(hash_key_tensor),
  #   np.array(true_image_shapes_tensor, dtype=np.int32),
  #   np.array(num_groundtruth_boxes_tensor, dtype=np.int32),
  #   np.array(groundtruth_boxes_tensor, dtype=np.float32),
  #   np.array(groundtruth_classes_tensor, dtype=np.float32),
  #   np.array(groundtruth_weights_tensor, dtype=np.float32)
  # )
  
  return processed_tensors

# def rali_build(enum, input_reader_config, iterator_initializer_hook, rali_batch_size=32, batch_size = 4):
def rali_build(enum, input_reader_config, rali_batch_size=32, batch_size = 4):
  # global_step = tf.train.get_global_step()
  # print ("\n\n\n\nGLOBAL STEP =", global_step)
  # tf.print ("\n\n\n\nGLOBAL STEP =", global_step)
  num_classes = 90
  # enum = enumerate(iterator, 0)
  
  




  # SERIAL CODE
  '''
  images_tensor = np.empty([0, 320, 320, 3], dtype = np.float32)
  true_image_shapes_tensor = np.empty([0, 3], dtype = np.int32)
  num_groundtruth_boxes_tensor = np.empty([0], dtype = np.int32)
  groundtruth_boxes_tensor = np.empty([0, 100, 4], dtype = np.float32)
  groundtruth_classes_tensor = np.empty([0, 100, numClasses], dtype = np.float32)
  groundtruth_weights_tensor = np.empty([0, 100], dtype = np.float32)
  # hash_key_tensor = np.empty([0], dtype = np.int32)
  hash_key_tensor = []

  print("\n######################################################################################################\n")
  print("\nStarting RALI augmentation pipeline...")
  sourceID = 1000000
  for i, (images_array, bboxes_array, labels_array, num_bboxes_array) in enumerate(iterator, 0):
    images_array = np.transpose(images_array, [0, 2, 3, 1])
    # bboxes_array = normalize_bbox_values(bboxes_array, batch_size)
    print("RALI augmentation pipeline - Processing batch %d....." % i)
    for element in list(range(rali_batch_size)):
      # print("Processing image %d....." % element)
      # hash_key = generate_hash_key().eval(session=tf.Session())
      # print("hash_key::",hash_key)
      # hash_key_tensor = np.append(hash_key_tensor,hash_key,axis = 0)
      hash_key_tensor.append(str(sourceID))
      images_tensor = np.append(images_tensor, np.array([images_array[element]], dtype = np.float32), axis = 0)
      num_groundtruth_boxes_tensor = np.append(num_groundtruth_boxes_tensor, np.array([num_bboxes_array[element]], dtype = np.int32), axis = 0)
      true_image_shapes_tensor = np.append(true_image_shapes_tensor, np.array([get_shapes(images_array[element])], dtype = np.int32), axis = 0)
      groundtruth_boxes_tensor = np.append(groundtruth_boxes_tensor, np.array([get_normalized(bboxes_array[element], len(images_array[element]), len(images_array[element,0]))], dtype = np.float32), axis = 0)
      groundtruth_classes_tensor = np.append(groundtruth_classes_tensor, np.array([get_onehot(labels_array[element], numClasses)], dtype = np.float32), axis = 0)
      groundtruth_weights_tensor = np.append(groundtruth_weights_tensor, np.array([get_weights(num_bboxes_array[element])], dtype = np.float32), axis = 0)
      sourceID += 1

    # if i >= 1:
    #   break
  '''






  # PARALLEL CODE
  '''
  result = []
  hash_key_tensor = []
  images_tensor = []
  num_groundtruth_boxes_tensor = []
  true_image_shapes_tensor = []
  groundtruth_boxes_tensor = []
  groundtruth_classes_tensor = []
  groundtruth_weights_tensor = []
  
  pool = mp.Pool(mp.cpu_count())
  sourceID = 1000000

  print("\n######################################################################################################\n")
  print("\nStarting RALI augmentation pipeline...")

  for i, (images_array, bboxes_array, labels_array, num_bboxes_array) in enumerate(iterator, 0):
    
    print("RALI augmentation pipeline - Processing batch %d....." % i)

    images_array = np.transpose(images_array, [0, 2, 3, 1])

    result = pool.map(rali_parallelized_tensor_generator, np.array([
      [images_array[element], bboxes_array[element], labels_array[element], num_bboxes_array[element], sourceID + element, num_classes] for element in list(range(rali_batch_size))
      ]))

    for image_data in result:
      hash_key_tensor.append(image_data[0])
      images_tensor.append(image_data[1])
      num_groundtruth_boxes_tensor.append(image_data[2])
      true_image_shapes_tensor.append(image_data[3])
      groundtruth_boxes_tensor.append(image_data[4])
      groundtruth_classes_tensor.append(image_data[5])
      groundtruth_weights_tensor.append(image_data[6])
    
    sourceID += rali_batch_size
  
  pool.close()

  features_dict = {
    "image" : np.array(images_tensor, dtype=np.float32),
    "hash" : np.array(hash_key_tensor),
    "true_image_shape" : np.array(true_image_shapes_tensor, dtype=np.int32)
  }
  labels_dict = {
    "num_groundtruth_boxes" : np.array(num_groundtruth_boxes_tensor, dtype=np.int32),
    "groundtruth_boxes" : np.array(groundtruth_boxes_tensor, dtype=np.float32),
    "groundtruth_classes" : np.array(groundtruth_classes_tensor, dtype=np.float32),
    "groundtruth_weights" : np.array(groundtruth_weights_tensor, dtype=np.float32)
  }
  
  processed_tensors = (features_dict, labels_dict)
  # print("\nPROCESSED_TENSORS:\n",processed_tensors)
  print("\nFinished RALI augmentation pipeline!\n")
  print("\n######################################################################################################\n")

  rali_dataset = tf.data.Dataset.from_tensor_slices(processed_tensors)
  # rali_dataset = tf.data.Dataset.from_tensor_slices(rali_process_tensors(batch_size, num_classes, iterator))
  # print("\nDATASET after creation:\n",rali_dataset)
  
  rali_dataset = rali_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  # print("\nDATASET after batching:\n",rali_dataset)
  
  rali_dataset = rali_dataset.prefetch(input_reader_config.num_prefetch_batches)
  # print("\nDATASET after prefetching:\n",rali_dataset)
  '''






  # CALLING rali_processed_tensors_generator() VERSION
  # '''
  rali_dataset = tf.data.Dataset.from_tensor_slices(rali_processed_tensors_generator(rali_batch_size = rali_batch_size, num_classes = num_classes, enum = enum))
  # print("\nDATASET after creation:\n",rali_dataset)
  
  rali_dataset = rali_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  # print("\nDATASET after batching:\n",rali_dataset)
  
  rali_dataset = rali_dataset.prefetch(input_reader_config.num_prefetch_batches)
  # print("\nDATASET after prefetching:\n",rali_dataset)
  # '''







  # TRIAL VERSION FOR GENERATOR
  '''

  def rali_generator_function(start_index):
    
    end_index = start_index + (10 * rali_batch_size)
    print("\n\nstart_index, end_index =" + str(start_index) + "," + str(end_index))

    hash_key_tensor_slice = hash_key_tensor[start_index:end_index]
    images_tensor_slice = images_tensor[start_index:end_index]
    num_groundtruth_boxes_tensor_slice = num_groundtruth_boxes_tensor[start_index:end_index]
    true_image_shapes_tensor_slice = true_image_shapes_tensor[start_index:end_index]
    groundtruth_boxes_tensor_slice = groundtruth_boxes_tensor[start_index:end_index]
    groundtruth_classes_tensor_slice = groundtruth_classes_tensor[start_index:end_index]
    groundtruth_weights_tensor_slice = groundtruth_weights_tensor[start_index:end_index]
  
    features_dict = {
      "image" : np.array(images_tensor_slice, dtype=np.float32),
      "hash" : np.array(hash_key_tensor_slice),
      "true_image_shape" : np.array(true_image_shapes_tensor_slice, dtype=np.int32)
    }
    labels_dict = {
      "num_groundtruth_boxes" : np.array(num_groundtruth_boxes_tensor_slice, dtype=np.int32),
      "groundtruth_boxes" : np.array(groundtruth_boxes_tensor_slice, dtype=np.float32),
      "groundtruth_classes" : np.array(groundtruth_classes_tensor_slice, dtype=np.float32),
      "groundtruth_weights" : np.array(groundtruth_weights_tensor_slice, dtype=np.float32)
    }

    start_index = end_index + 1
  
    yield [start_index, (features_dict, labels_dict)]

  start_index = 0
  # processed_tensors = (features_dict, labels_dict)
  processed_tensors_function = rali_generator_function(start_index = start_index)
  # print("\nPROCESSED_TENSORS:\n",processed_tensors)
  print("\nFinished RALI augmentation pipeline!\n")
  print("\n######################################################################################################\n")

  output_of_generator = next(processed_tensors_function)
  start_index = output_of_generator[0]
  sliced_dataset_tuple = output_of_generator[1]
  rali_dataset = tf.data.Dataset.from_tensor_slices(sliced_dataset_tuple)
  # rali_dataset = tf.data.Dataset.from_tensor_slices(rali_process_tensors(batch_size, num_classes, iterator))
  # print("\nDATASET after creation:\n",rali_dataset)
  
  rali_dataset = rali_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  # print("\nDATASET after batching:\n",rali_dataset)
  
  rali_dataset = rali_dataset.prefetch(input_reader_config.num_prefetch_batches)
  # print("\nDATASET after prefetching:\n",rali_dataset)
  '''







  # WITH INITIALIZABLE ITERATOR
  '''
  with tf.name_scope('training_data'):
    processed_tensors = rali_processed_tensors_generator(rali_batch_size, num_classes, enum)
    
    # features_placeholder = tf.placeholder(processed_tensors[0].dtype, processed_tensors[0].shape)
    # labels_placeholder = tf.placeholder(processed_tensors[1].dtype, processed_tensors[1].shape)

    images_placeholder = tf.placeholder(processed_tensors[0].dtype, processed_tensors[0].shape)
    hash_key_placeholder = tf.placeholder(processed_tensors[1].dtype, processed_tensors[1].shape)
    true_image_shapes_placeholder = tf.placeholder(processed_tensors[2].dtype, processed_tensors[2].shape)
    num_groundtruth_boxes_placeholder = tf.placeholder(processed_tensors[3].dtype, processed_tensors[3].shape)
    groundtruth_boxes_placeholder = tf.placeholder(processed_tensors[4].dtype, processed_tensors[4].shape)
    groundtruth_classes_placeholder = tf.placeholder(processed_tensors[5].dtype, processed_tensors[5].shape)
    groundtruth_weights_placeholder = tf.placeholder(processed_tensors[6].dtype, processed_tensors[6].shape)
    
    rali_dataset = tf.data.Dataset.from_tensor_slices(
      (
        {
          "image" : images_placeholder,
          "hash" : hash_key_placeholder,
          "true_image_shape" : true_image_shapes_placeholder
        },
        {
          "num_groundtruth_boxes" : num_groundtruth_boxes_placeholder,
          "groundtruth_boxes" : groundtruth_boxes_placeholder,
          "groundtruth_classes" : groundtruth_classes_placeholder,
          "groundtruth_weights" : groundtruth_weights_placeholder
        }
      )
    )
    # rali_dataset = rali_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    # rali_dataset = rali_dataset.prefetch(input_reader_config.num_prefetch_batches)
    # rali_dataset = rali_dataset.repeat()
    rali_dataset = rali_dataset.batch(batch_size)
    # rali_dataset = rali_dataset.repeat()
    iterator = rali_dataset.make_initializable_iterator()
    next_feature_set, next_label_set = iterator.get_next()

    iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(
      iterator.initializer,
      feed_dict={
        images_placeholder : processed_tensors[0], 
        hash_key_placeholder : processed_tensors[1], 
        true_image_shapes_placeholder : processed_tensors[2], 
        num_groundtruth_boxes_placeholder : processed_tensors[3], 
        groundtruth_boxes_placeholder : processed_tensors[4], 
        groundtruth_classes_placeholder : processed_tensors[5], 
        groundtruth_weights_placeholder : processed_tensors[6]
      }
    )
    
    return next_feature_set, next_label_set
  '''

  return rali_dataset
  # return tf.compat.v1.data.make_one_shot_iterator(rali_dataset)

def build(input_reader_config, batch_size=None, transform_input_data_fn=None, multi_gpu=True):
  """Builds a tf.data.Dataset.

  Builds a tf.data.Dataset by applying the `transform_input_data_fn` on all
  records. Applies a padded batch to the resulting dataset.

  Args:
    input_reader_config: A input_reader_pb2.InputReader object.
    batch_size: Batch size. If batch size is None, no batching is performed.
    transform_input_data_fn: Function to apply transformation to all records,
      or None if no extra decoding is required.

  Returns:
    A tf.data.Dataset based on the input_reader_config.

  Raises:
    ValueError: On invalid input reader proto.
    ValueError: If no input paths are specified.
  """
  if not isinstance(input_reader_config, input_reader_pb2.InputReader):
    raise ValueError('input_reader_config not of type '
                     'input_reader_pb2.InputReader.')

  if input_reader_config.WhichOneof('input_reader') == 'tf_record_input_reader':
    config = input_reader_config.tf_record_input_reader
    if not config.input_path:
      raise ValueError('At least one input path must be specified in '
                       '`input_reader_config`.')

    label_map_proto_file = None
    if input_reader_config.HasField('label_map_path'):
      label_map_proto_file = input_reader_config.label_map_path
    decoder = tf_example_decoder.TfExampleDecoder(
        load_instance_masks=input_reader_config.load_instance_masks,
        instance_mask_type=input_reader_config.mask_type,
        label_map_proto_file=label_map_proto_file,
        use_display_name=input_reader_config.use_display_name,
        num_additional_channels=input_reader_config.num_additional_channels)

    def process_fn(value):
      """Sets up tf graph that decodes, transforms and pads input data."""
      processed_tensors = decoder.decode(value)
      print("\n\n\n\n\nprocessed_tensors::",processed_tensors)
      print("\n\n\n\n\nprocessed_tensors::",type(processed_tensors))
      
      if transform_input_data_fn is not None:
        processed_tensors = transform_input_data_fn(processed_tensors)
      print("\n\n\n\n\nprocessed_tensors::",processed_tensors)
      print("\n\n\n\n\nprocessed_tensors::",type(processed_tensors))
      
      return processed_tensors

    dataset = read_dataset(
        functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
        config.input_path[:], input_reader_config)
    
    print("\n\n\n\n\n1st - DATSET::",dataset)
    print("\n\n\n\n\n1st - TYPE OF DATASET::",type(dataset))
    
    if multi_gpu:
        dataset = dataset.shard(hvd.size(), hvd.rank())
    # TODO(rathodv): make batch size a required argument once the old binaries
    # are deleted.
    
    if batch_size:
      num_parallel_calls = batch_size * input_reader_config.num_parallel_batches
    else:
      num_parallel_calls = input_reader_config.num_parallel_map_calls
    
    dataset = dataset.map(
        process_fn,
        num_parallel_calls=num_parallel_calls)
    print("\n\n\n\n\n2nd - DATSET::",dataset)
    print("\n\n\n\n\n2nd - TYPE OF DATASET::",type(dataset))
    
    if batch_size:
      dataset = dataset.apply(
          tf.contrib.data.batch_and_drop_remainder(batch_size))
    print("\n\n\n\n\n3rd - DATSET::",dataset)
    print("\n\n\n\n\n3rd - TYPE OF DATASET::",type(dataset))
    
    dataset = dataset.prefetch(input_reader_config.num_prefetch_batches)
    print("\n\n\n\n\nDATSET::",dataset)
    print("\n\n\n\n\nTYPE OF DATASET::",type(dataset))
    
    return dataset

  raise ValueError('Unsupported input_reader_config.')
