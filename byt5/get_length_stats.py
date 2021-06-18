# Copyright 2021 The ByT5 Authors.
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

r"""Get the stats of input and target length for T5/mT5/ByT5 tasks.

If the input length is too long which causes OOM, one could trunk the length to
1.1 * 99 percentile length.

"""

from absl import app
from absl import flags
import byt5.tasks  # pylint:disable=unused-import
import multilingual_t5.tasks  # pylint:disable=unused-import
import t5
import tensorflow_datasets as tfds



FLAGS = flags.FLAGS
flags.DEFINE_string("task_or_mixture", None, "task/mixture name")
flags.DEFINE_string("split", "train", "train, validation, or test.")
flags.DEFINE_boolean("use_cached", True, "Use cached or not.")


def get_perc_99_len(input_length):
  """Get 99 percentile sequence length."""
  lengths = sorted(input_length)
  perc_99 = len(input_length) * 99 // 100
  perc_99_len = lengths[perc_99]
  return perc_99_len


def get_stats(task_or_mixture, split="train"):
  """Get task length stats.

  Args:
    task_or_mixture: string, task or mixture name.
    split: string, split.
  """
  if task_or_mixture in t5.data.seqio.TaskRegistry.names():
    data = t5.data.seqio.TaskRegistry.get(task_or_mixture)
  elif task_or_mixture in t5.data.seqio.MixtureRegistry.names():
    data = t5.data.seqio.MixtureRegistry.get(task_or_mixture)
  else:
    raise ValueError("Task is not registered.")

  data = data.get_dataset(split=split,
                          sequence_length=None,
                          use_cached=FLAGS.use_cached)
  ds = list(tfds.as_numpy(data))

  # Input length stats.
  input_length = [len(ex["inputs"]) for ex in ds]
  recommend_input_length = 1.1 * get_perc_99_len(input_length)
  print(f"Min input length: {min(input_length)}")
  print(f"Max input length: {max(input_length)}")
  print(f"1.1 * 99 percertile of length: {recommend_input_length}")

  # Target length stats.
  target_length = [len(ex["targets"]) for ex in ds]
  print(f"Min target length: {min(target_length)}")
  print(f"Max target length: {max(target_length)}")



def main(_):
  get_stats(FLAGS.task_or_mixture, split=FLAGS.split)

if __name__ == "__main__":
  app.run(main)
