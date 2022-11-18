import pandas as pd

TRANSFORMER_PATH = 'huggingface/pytorch-transformers'

LABEL_MAPPING = {
    'B': 0,
    'I': 1,
    'O': 2
}

def convert_examples_to_features(x_batch, y_batch, tokenizer, max_seq_length):
    """
    Convert preprocessed data with tags to BERT's input format

    ex: After preprocessed
    Text: [", I, am, Alvin, ., "] (original: "I am Alvin.")
    Tags:  O  O   O    B    O  O

    After converted
    tokens: [", I, am, Al, ##vin, ., "]
    labels:  O  O   O   B    I    O  O
    """
    token_batch = []
    label_batch = []
    for train_x, train_y in zip(x_batch, y_batch):
        text, labels_org = train_x, train_y.strip().split()[:max_seq_length]
        tokens = tokenizer.tokenize(text, truncation=True, max_length=max_seq_length)

        labels = []
        token_bias_num = 0
        word_num = 0
        for i, token in enumerate(tokens):
            if token.startswith('##'):
                if labels_org[i - 1 - token_bias_num][0] in ['O', 'I']:
                    label = LABEL_MAPPING[labels_org[i - 1 - token_bias_num][0]]
                    labels.append(label)
                else:
                    labels.append(1)  # 1 is I
                token_bias_num += 1
            else:
                word_num += 1
                label = LABEL_MAPPING[labels_org[i - token_bias_num][0]]
                labels.append(label)

        # manually pad tokens and labels if their lengths are over the max sequence length
        if len(tokens) < max_seq_length:
            tokens += [''] * (max_seq_length - len(tokens))
            labels += [LABEL_MAPPING['O']] * (max_seq_length - len(labels))

        token_batch.append(tokens)
        label_batch.append(labels)

    return token_batch, label_batch

# loading data from input file
def get_data(filename, balanced=False):
    df = pd.read_csv(filename)
    texts = df['texts'].tolist()
    labels = df['labels'].tolist()

    if balanced:
        new_texts = list(texts)
        new_labels = list(labels)
        for text, label in zip(texts, labels):
            if 'B' in label:
                new_texts += [text] * 9
                new_labels += [label] * 9
        return new_texts, new_labels

    return texts, labels