import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms


spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, lower_threshold, upper_threshold):
        #self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        #self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        
        self.itos = {0: "<PAD>"}
        self.stoi = {"<PAD>": 0}
        
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        tokens = []
        for token in spacy_eng.tokenizer(text):
            tokens.append(token.text.lower())
        return tokens

    def build_vocabulary(self, sentence_list):
        self.frequencies = {}
        self.count_sentences = 0
        idx = 1

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in self.frequencies:
                    self.frequencies[word] = 1

                else:
                    self.frequencies[word] += 1
            self.count_sentences += 1

        for word in self.frequencies:
            if (self.frequencies[word] >= self.lower_threshold) and (self.frequencies[word] <= self.upper_threshold) and (word not in self.stoi):
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1    
        
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        
        numericalized_text = []
        for token in tokenized_text:
            if token in self.stoi:
                numericalized_text.append(self.stoi[token])
            else:
                numericalized_text.append(self.stoi["<UNK>"])
                
        return numericalized_text


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, lower_threshold=0, upper_threshold=500000):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Get img, caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(lower_threshold, upper_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        #numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption = []
        numericalized_caption += self.vocab.numericalize(caption)
        #numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = []
        targets = []
        for item in batch:
            imgs.append(item[0].unsqueeze(0))
            targets.append(item[1])
        
        imgs = torch.cat(imgs, dim=0) # create batch of imgs
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        # changes targets dim from (batch, seq, vocab_size) to (seq, batch, vocab_size)
        return imgs, targets


def data_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    shuffle=True,
    pin_memory=True,
):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return train_loader, dataset


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )

    train_loader, dataset = data_loader(
        "flickr30k/flickr30k_images/", "flickr30k/results.csv", transform=transform
    ) 
    
    for idx, (imgs, captions) in enumerate(train_loader):
        for i in range(captions.size(0)):
            for j in range(captions.size(1)):
                result = dataset.vocab.itos[captions[i][j].item()]  
                print(result, end = " ") 
            print("\n")
            
        import sys
        sys.exit()  
         
# Each img has 5 different annotations. 
# ---> Each annotation uses same img indicated from csv file.
     
# Batch of captions is padded to have equal length
# loader will take a batch of images and captions