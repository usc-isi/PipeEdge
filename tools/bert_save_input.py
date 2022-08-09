import datasets
import numpy as np

ds_train = datasets.load_dataset('glue', name='cola', split='train')

# The original evaluation input file appears to have randomly sampled 512 strings (with duplicates)
# from the first ~1024 sentences in the dataset.
# For simplicity, we'll just take the first 512 (without duplicates).
bert_input = ds_train[:512]['sentence']
bert_label = ds_train[:512]['label']

# The following custom string was at index=0 in the original evaluation input file.
# The intent seems to be to force the tokenizer to produce input_ids with width=512.
# While there are better ways to achieve that, this is what was done.
bert_input[0] = 'hello ' * 512
bert_label[0] = 0

np.savez('bert_input.npz', input=bert_input, label=bert_label)
