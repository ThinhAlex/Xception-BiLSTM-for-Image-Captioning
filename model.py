import torch
import torch.nn as nn
import timm

from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torchmetrics.text import BLEUScore  
from dataset import FlickrDataset      
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                              
class Encoder(nn.Module):
    def __init__(self, embedding_size):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.xception = timm.create_model('legacy_xception', pretrained=True)
        self.xception.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, self.embedding_size),
        )
        
    def forward(self, images):
        return self.xception(images)
    
class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, dataset):
        super(Decoder, self).__init__()
        # this will be used for sentence generation later on
        self.corpus = dataset.vocab
        self.vocab_size = len(self.corpus.itos)     
        
        # for sentence features extraction
        self.embed = nn.Embedding(self.vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(0.5)  
        self.linear = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, self.vocab_size),
            )
        
    def forward(self, features, captions): # captions: batch x seq
        
        embeds = self.embed(captions[:,:-1]) # embeds: batch x (seq-1) x embed_size
        embeds = torch.cat((features.unsqueeze(1), embeds), dim = 1) # embs: batch x seq x embed
        hiddens, _ = self.lstm(embeds) # hidden: batch x seq x hidden_size
        hiddens = self.dropout(hiddens)     
        outputs = self.linear(hiddens) 
        return outputs
   
class CaptionModel(pl.LightningModule):
    def __init__(self, embedding_size, hidden_size, num_layers, dataset, teacher_forcing_ratio = 1):
        super(CaptionModel, self).__init__()
        # Hyperparameters
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dataset = dataset
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # components and data
        self.encoder = Encoder(embedding_size)
        self.decoder = Decoder(embedding_size, hidden_size, num_layers, dataset)     
        self.corpus = self.decoder.corpus
        self.vocab_size = self.decoder.vocab_size
        
        # metrics
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.bleu_score = BLEUScore(n_gram=3, smooth=True)
        
    def forward(self, images, captions):
        prob = torch.rand(1).item()  
          
        if prob < self.teacher_forcing_ratio:
            features = self.encoder.forward(images)
            outputs = self.decoder.forward(features, captions)
            return outputs  
              
        else: 
            results = []
            inputs = self.encoder.forward(images) # inputs: batch x emb
            states = None
            for i in range(captions.shape[1]):
                hiddens, states = self.decoder.lstm(inputs.unsqueeze(1), states) # hiddens: batch x 1 x hidden
                outputs = self.decoder.linear(hiddens.squeeze(1)).unsqueeze(1) # outputs: batch x 1 x vocab
                results.append(outputs)                
                pred_idx = torch.argmax(outputs, dim = 2)
                inputs = self.decoder.embed(pred_idx).squeeze(1)
            
            outputs = torch.cat(results, dim = 1)               
            return outputs       
            
    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss, outputs, captions = self._common_step(batch, batch_idx)
        
        if batch_idx % 10 == 0:
            bleu_score = self.compute_bleu_score(outputs, captions)
            self.log_dict({"train_loss": loss, "train_bleu": bleu_score}, on_epoch=True, prog_bar=True)   
            
        if batch_idx % 50 == 0:  
            # get img and label
            image = images[0] 
            label = labels[0,:]
            label = [self.corpus.itos[idx.item()] for idx in label]
            
            # generate caption and log
            caption = self.generate_caption_example(image)
            print(caption)
            
        return {"loss": loss, "outputs": outputs, "captions": captions}
    
    def validation_step(self, batch, batch_idx):
        loss, outputs, captions = self._common_step(batch, batch_idx)
        
        if batch_idx % 100 == 0:
            bleu_score = self.compute_bleu_score(outputs, captions)
            self.log_dict({"val_loss": loss, "val_bleu": bleu_score}, on_epoch=True)
            
        return {"loss": loss, "outputs": outputs, "captions": captions}
    
    def test_step(self, batch, batch_idx):
        loss, outputs, captions = self._common_step(batch, batch_idx)
        bleu_score = self.compute_bleu_score(outputs, captions) 
        self.log_dict({"test_loss": loss, "test_bleu": bleu_score})
        return {"loss": loss, "outputs": outputs, "captions": captions}
    
    def _common_step(self, batch, batch_idx):
        images, captions = batch
        outputs = self.forward(images, captions)
        loss = self.loss_fn(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
        return loss, outputs, captions
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 3e-4, weight_decay= 1e-4)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode="max", factor = 0.5, patience = 2, verbose = 1, min_lr=0.000001),
            'monitor': 'train_bleu_epoch'
        }
        return [optimizer], [scheduler]
    
    def generate_caption_example(self, image, max_length = 15):
        result_word = []
        image = image.to(device)
        with torch.no_grad(): 
            input = self.encoder.forward(image.unsqueeze(0)) # change image dim = 4
            states = None           
            for _ in range(max_length):
                hidden, states = self.decoder.lstm(input.unsqueeze(1), states)
                hidden = self.decoder.dropout(hidden)                 
                output = self.decoder.linear(hidden.squeeze(1)) 
                
                predicted_idx = torch.argmax(output, dim = 1)    
                predicted_word = self.corpus.itos[predicted_idx.item()]      
                if predicted_word != "<PAD>":
                    result_word.append(predicted_word)            
                #if predicted_word == "<EOS>":
                    #return result_word 
                
                input = self.decoder.embed(predicted_idx)  
        return result_word           
    
    def compute_bleu_score(self, outputs, captions):
        # outputs: batch x seq x vocab_size
        # captions: batch x seq
        # change outputs to ["........", "........", ... "........"]
        predictions = []
        references = []
        outputs = outputs.argmax(dim = 2)
        for i in range(outputs.shape[0]):
            pred_sentence = ""
            ref_sentence = ""
            for j in range(outputs.shape[1]):
                pred_idx = outputs[i, j].item()
                pred_word = self.corpus.itos[pred_idx]
                pred_sentence += pred_word + " "
                
                ref_idx = captions[i, j].item()
                ref_word = self.corpus.itos[ref_idx]
                ref_sentence += ref_word + " "
            pred_sentence = pred_sentence.strip()
            predictions.append(pred_sentence)
            
            ref_sentence = ref_sentence.strip()
            references.append([ref_sentence])
               
        bleu_score = self.bleu_score(predictions, references)
        return bleu_score 
    
    