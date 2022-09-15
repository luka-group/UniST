import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaPreTrainedModel, RobertaModel

class UniSTModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)       
        self.roberta = RobertaModel(config)      
        self.margin = config.margin       
        self.init_weights()
        
    def forward(
        self, 
        sent_input_ids,
        pos_input_ids,
        neg_input_ids,
        sent_attention_mask=None,
        pos_attention_mask=None,
        neg_attention_mask=None,
    ):
        
        sent_embeddings = self.embed(sent_input_ids, sent_attention_mask)
        pos_embeddings = self.embed(pos_input_ids, pos_attention_mask)
        neg_embeddings = self.embed(neg_input_ids, neg_attention_mask)
        
        loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.dist_fn,
            margin=self.margin
        )
        
        loss = loss_fn(sent_embeddings, pos_embeddings, neg_embeddings)
        
        return loss, sent_embeddings
        
    
    def embed(
        self,
        input_ids,      
        attention_mask=None,
    ):        
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        embeddings = outputs[1]              
        
        return embeddings
        
    def dist_fn(
        self,
        sent_embeddings,
        label_embeddings
    ):
        return 1.0 - F.cosine_similarity(sent_embeddings, label_embeddings)