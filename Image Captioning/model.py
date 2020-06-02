import torch
import torch.nn as nn
import torchvision.models as models
import torch.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(
            input_size=self.embed_size, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
            )
        self.feat2caption = nn.Linear(self.hidden_size, self.vocab_size)
    
    
    def forward(self, features, captions):
        
        batch_size = features.size(0)
        in_captions = self.word_embeddings(captions[:,:-1]).cuda()
        
        hidden = torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda()
        cell = torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda()
        
        all_feat = torch.cat([features.unsqueeze(1),in_captions],dim=1).cuda()
        lstm_out, _ = self.lstm(all_feat, (hidden,cell))
        outputs = self.feat2caption(lstm_out).cuda()
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        hidden = torch.zeros((self.num_layers, 1, self.hidden_size)).cuda()
        if states is None:
            states = torch.zeros((self.num_layers, 1, self.hidden_size)).cuda()
            
        for _ in range(max_len):
            out, (hidden, states) = self.lstm(inputs, (hidden, states))
            output = self.feat2caption(out.squeeze()).cuda()
            output_id = torch.argmax(output)
            outputs.append(output_id.item())
            if output_id == 1:
                return outputs[:-1]
            inputs = self.word_embeddings(output_id).reshape(1,1,-1)
        return outputs