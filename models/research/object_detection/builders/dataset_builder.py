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
# import multiprocessing as mp
from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import input_reader_pb2
import object_detection.rali as rali

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
        element[1] ,
        element[0] ,
        (element[1] + element[3]) ,
        (element[0] + element[2]) 
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

def rali_processed_train_tensors_generator(rali_batch_size, num_classes, iterator):

  result = []
  hash_key_tensor = []
  images_tensor = []
  num_groundtruth_boxes_tensor = []
  true_image_shapes_tensor = []
  groundtruth_boxes_tensor = []
  groundtruth_classes_tensor = []
  groundtruth_weights_tensor = []

  # pool = mp.Pool(mp.cpu_count())
  flag = True
  count = 0

  while flag == True:

    try:
      i, (images_array, bboxes_array, labels_array, num_bboxes_array) = rali.RALI_TRAIN_ENUM.__next__()
    except:
      rali.initialize_enumerator(iterator, 0)
      i, (images_array, bboxes_array, labels_array, num_bboxes_array) = rali.RALI_TRAIN_ENUM.__next__()

    print("RALI augmentation pipeline - Processing RALI batch %d....." % i)

    sourceID = 1000000 + (i * rali_batch_size)

    images_array = np.transpose(images_array, [0, 2, 3, 1])

    result = rali_parallelized_tensor_generator( np.array([
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

    count += 1

    flag = not(count % int(1536 / rali_batch_size) == 0)

  # pool.close()

  return [np.array(images_tensor, dtype=np.float32),
  np.array(hash_key_tensor),
  np.array(true_image_shapes_tensor, dtype=np.int32),
  np.array(num_groundtruth_boxes_tensor, dtype=np.int32),
  np.array(groundtruth_boxes_tensor, dtype=np.float32),
  np.array(groundtruth_classes_tensor, dtype=np.float32),
  np.array(groundtruth_weights_tensor, dtype=np.float32)]


def rali_processed_val_tensors_generator(rali_batch_size, num_classes, iterator):

  result = []
  hash_key_tensor = []
  images_tensor = []
  num_groundtruth_boxes_tensor = []
  true_image_shapes_tensor = []
  groundtruth_boxes_tensor = []
  groundtruth_classes_tensor = []
  groundtruth_weights_tensor = []

  # pool = mp.Pool(mp.cpu_count())
  flag = True
  count = 0

  while flag == True:

    try:
      i, (images_array, bboxes_array, labels_array, num_bboxes_array) = rali.RALI_VAL_ENUM.__next__()
    except:
      rali.initialize_enumerator(iterator, 1)
      i, (images_array, bboxes_array, labels_array, num_bboxes_array) = rali.RALI_VAL_ENUM.__next__()

    print("RALI augmentation pipeline - Processing RALI batch %d....." % i)

    sourceID = 1000000 + (i * rali_batch_size)

    images_array = np.transpose(images_array, [0, 2, 3, 1])

    result = rali_parallelized_tensor_generator( np.array([
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

    count += 1

    flag = not(count % int(1536 / rali_batch_size) == 0)

  # pool.close()

  return [np.array(images_tensor, dtype=np.float32),
  np.array(hash_key_tensor),
  np.array(true_image_shapes_tensor, dtype=np.int32),
  np.array(num_groundtruth_boxes_tensor, dtype=np.int32),
  np.array(groundtruth_boxes_tensor, dtype=np.float32),
  np.array(groundtruth_classes_tensor, dtype=np.float32),
  np.array(groundtruth_weights_tensor, dtype=np.float32)]

def rali_train_build(iterator, input_reader_config, rali_batch_size=32, batch_size = 4):

  num_classes = 90

  def rali_train_generator():

    i = 0

    images_tensor = np.array([])
    hash_key_tensor = np.array([])
    true_image_shapes_tensor = np.array([])
    num_groundtruth_boxes_tensor = np.array([])
    groundtruth_boxes_tensor = np.array([])
    groundtruth_classes_tensor = np.array([])
    groundtruth_weights_tensor = np.array([])

    while True:

      testElement = hash_key_tensor[(i * batch_size) : ((i * batch_size) + batch_size)]

      if testElement.size == 0:
        print("\nFetching next augmented train-tensor batch from RALI...")
        i = 0

        result = rali_processed_train_tensors_generator(
          rali_batch_size = rali_batch_size,
          num_classes = num_classes,
          iterator = iterator
        )

        images_tensor = result[0]
        hash_key_tensor = result[1]
        true_image_shapes_tensor = result[2]
        num_groundtruth_boxes_tensor = result[3]
        groundtruth_boxes_tensor = result[4]
        groundtruth_classes_tensor = result[5]
        groundtruth_weights_tensor = result[6]

      start = i * batch_size
      stop = start + batch_size

      features_dict = {
        "image" : images_tensor[start:stop],
        "hash" : hash_key_tensor[start:stop],
        "true_image_shape" : true_image_shapes_tensor[start:stop],
      }
      labels_dict = {
        "num_groundtruth_boxes" : num_groundtruth_boxes_tensor[start:stop],
        "groundtruth_boxes" : groundtruth_boxes_tensor[start:stop],
        "groundtruth_classes" : groundtruth_classes_tensor[start:stop],
        "groundtruth_weights" : groundtruth_weights_tensor[start:stop]
      }

      processed_tensors_tuple = (features_dict, labels_dict)

      yield processed_tensors_tuple

      i += 1

  rali_dataset = tf.data.Dataset.from_generator(rali_train_generator,
    output_types = (
      {
        "image" : tf.float32,
        "hash" : tf.string,
        "true_image_shape" : tf.int32
      },
      {
        "num_groundtruth_boxes" : tf.int32,
        "groundtruth_boxes" : tf.float32,
        "groundtruth_classes" : tf.float32,
        "groundtruth_weights" : tf.float32
      }
    ),
    output_shapes = (
      {
        "image" : (batch_size, 320, 320, 3),
        "hash" : (batch_size, ),
        "true_image_shape" : (batch_size, 3)
      },
      {
        "num_groundtruth_boxes" : (batch_size, ),
        "groundtruth_boxes" : (batch_size, 100, 4),
        "groundtruth_classes" : (batch_size, 100, 90),
        "groundtruth_weights" : (batch_size, 100)
      }
    )
  )

  rali_one_shot_iterator = rali_dataset.make_one_shot_iterator()

  return rali_one_shot_iterator.get_next()


def rali_val_build(iterator, input_reader_config, rali_batch_size=32, batch_size = 4):

  num_classes = 90

  def rali_val_generator():

    i = 0

    images_tensor = np.array([])
    hash_key_tensor = np.array([])
    true_image_shapes_tensor = np.array([])
    num_groundtruth_boxes_tensor = np.array([])
    groundtruth_boxes_tensor = np.array([])
    groundtruth_classes_tensor = np.array([])
    groundtruth_weights_tensor = np.array([])

    while True:

      testElement = hash_key_tensor[(i * batch_size) : ((i * batch_size) + batch_size)]

      if testElement.size == 0:
        print("\nFetching next augmented val-tensor batch from RALI...")
        i = 0

        result = rali_processed_val_tensors_generator(
          rali_batch_size = rali_batch_size,
          num_classes = num_classes,
          iterator = iterator
        )

        images_tensor = result[0]
        hash_key_tensor = result[1]
        true_image_shapes_tensor = result[2]
        num_groundtruth_boxes_tensor = result[3]
        groundtruth_boxes_tensor = result[4]
        groundtruth_classes_tensor = result[5]
        groundtruth_weights_tensor = result[6]

      start = i * batch_size
      stop = start + batch_size

      features_dict = {
        "image" : images_tensor[start:stop],
        "hash" : hash_key_tensor[start:stop],
        "true_image_shape" : true_image_shapes_tensor[start:stop],
      }
      labels_dict = {
        "num_groundtruth_boxes" : num_groundtruth_boxes_tensor[start:stop],
        "groundtruth_boxes" : groundtruth_boxes_tensor[start:stop],
        "groundtruth_classes" : groundtruth_classes_tensor[start:stop],
        "groundtruth_weights" : groundtruth_weights_tensor[start:stop]
      }

      processed_tensors_tuple = (features_dict, labels_dict)

      yield processed_tensors_tuple

      i += 1

  rali_dataset = tf.data.Dataset.from_generator(rali_val_generator,
    output_types = (
      {
        "image" : tf.float32,
        "hash" : tf.string,
        "true_image_shape" : tf.int32
      },
      {
        "num_groundtruth_boxes" : tf.int32,
        "groundtruth_boxes" : tf.float32,
        "groundtruth_classes" : tf.float32,
        "groundtruth_weights" : tf.float32
      }
    ),
    output_shapes = (
      {
        "image" : (batch_size, 320, 320, 3),
        "hash" : (batch_size, ),
        "true_image_shape" : (batch_size, 3)
      },
      {
        "num_groundtruth_boxes" : (batch_size, ),
        "groundtruth_boxes" : (batch_size, 100, 4),
        "groundtruth_classes" : (batch_size, 100, 90),
        "groundtruth_weights" : (batch_size, 100)
      }
    )
  )

  rali_one_shot_iterator = rali_dataset.make_one_shot_iterator()

  return rali_one_shot_iterator.get_next()

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

      if transform_input_data_fn is not None:
        processed_tensors = transform_input_data_fn(processed_tensors)

      return processed_tensors

    dataset = read_dataset(
        functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
        config.input_path[:], input_reader_config)

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

    if batch_size:
      dataset = dataset.apply(
          tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(input_reader_config.num_prefetch_batches)

    return dataset

  raise ValueError('Unsupported input_reader_config.')
