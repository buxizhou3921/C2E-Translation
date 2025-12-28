import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import config


class TranslationDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['zh'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['en'], dtype=torch.long)
        return input_tensor, target_tensor


def collate_fn(batch):
    input_tensors = [item[0] for item in batch]
    target_tensors = [item[1] for item in batch]
    src_lengths = torch.LongTensor([len(seq) for seq in input_tensors])

    input_tensor = pad_sequence(input_tensors, batch_first=True, padding_value=0)
    target_tensor = pad_sequence(target_tensors, batch_first=True, padding_value=0)

    return input_tensor, target_tensor, src_lengths


def get_dataloader(args, mode='train'):
    if mode == 'train':
        path = config.PROCESSED_DATA_DIR / 'train.jsonl'
        shuffle = True
    elif mode == 'valid':
        path = config.PROCESSED_DATA_DIR / 'valid.jsonl'
        shuffle = False
    elif mode == 'test':
        path = config.PROCESSED_DATA_DIR / 'test.jsonl'
        shuffle = False
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'train', 'valid', or 'test'!")

    dataset = TranslationDataset(path)
    return DataLoader(dataset, batch_size=args.bs, shuffle=shuffle, collate_fn=collate_fn)


if __name__ == '__main__':
    pass
