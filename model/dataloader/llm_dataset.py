from torch.utils.data import Dataset
import torch
import tiktoken

class LLMDataset(Dataset):
    def __init__(self, context_length, stride, raw_text):
        """
        Initialize the LLMDataset.

        Args:
            context_length (int): The size of the sliding window for inputs.
            stride (int): The step size for the sliding window.
            raw_text (str): The raw text to be loaded and encoded.
        """

        # Encode the text using the GPT-2 tokenizer
        self.enc_text = tiktoken.get_encoding("gpt2").encode(raw_text)

        # Create sliding window inputs and targets
        self.window_size = context_length
        self.inputs = []
        self.targets = []
        for i in range(0, len(self.enc_text) - self.window_size, stride):
            self.inputs.append(self.enc_text[i:i + self.window_size])
            self.targets.append(self.enc_text[i + 1:i + self.window_size + 1])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Return the input-target pair at the specified index.

        Args:
            idx (int): Index of the desired input-target pair.

        Returns:
            tuple: (input, target) where both are PyTorch tensors of token IDs.
        """
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])