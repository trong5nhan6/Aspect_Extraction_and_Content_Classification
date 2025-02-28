from datasets import load_dataset
from transformers import AutoTokenizer


def tokenize_and_align_labels(examples):
    '''
    input {'Tokens': ['But', 'the', 'staff', 'was', 'so', 'horrible', 'to', 'us', '.'],
    'Tags': ['0', '0', '1', '0', '0', '0', '0', '0', '0'],
    'Polarities': ['-1', '-1', '0', '-1', '-1', '-1', '-1', '-1', '-1']}

    output {'Tokens': ['But', 'the', 'staff', 'was', 'so', 'horrible', 'to', 'us', '.'],
    'Tags': ['0', '0', '1', '0', '0', '0', '0', '0', '0'],
    'Polarities': ['-1', '-1', '0', '-1', '-1', '-1', '-1', '-1', '-1'],
    'input_ids': [2021, 1996, 3095, 2001, 2061, 9202, 2000, 2149, 1012],
    'labels': [0, 0, 1, 0, 0, 0, 0, 0, 0]}
    '''
    tokenized_inputs = []
    labels = []
    for tokens, tags in zip(examples['Tokens'], examples['Tags']):

        bert_tokens = []
        bert_tags = []
        for i in range(len(tokens)):
            t = tokenizer.tokenize(tokens[i])
            bert_tokens += t
            bert_tags += [int(tags[i])]*len(t)

        bert_ids = tokenizer.convert_tokens_to_ids(bert_tokens)

        tokenized_inputs.append(bert_ids)
        labels.append(bert_tags)

    return {
        'input_ids': tokenized_inputs,
        'labels': labels
    }


ds = load_dataset('thainq107/abte-restaurants')

tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

preprocessed_ds = ds.map(tokenize_and_align_labels, batched=True)
