from datasets import load_dataset
from torch.utils.data import DataLoader
import tiktoken
from torch.nn.utils.rnn import pad_sequence

class PretrainDataset:
    def __init__(self, split='train', block_size=1024, tokenizer=None):
        self.dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
        self.block_size = block_size
        if tokenizer is None:
            self.tokenizer = tiktoken.get_encoding("gpt2")
        else:
            self.tokenizer = tokenizer

    def __iter__(self):
        for example in self.dataset:
            text = example['text']
            tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
            for i in range(0, len(tokens) - self.block_size, self.block_size):
                yield tokens[i:i + self.block_size + 1]  # Input + target shift

def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=-1)

def get_dataloader(batch_size=1, block_size=1024):
    dataset = PretrainDataset(block_size=block_size)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)  # Streaming via iter