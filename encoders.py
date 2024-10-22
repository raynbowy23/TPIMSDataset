import torch
# from sentence_transformers import SentenceTransformer


class IdentityEncoder(object):
    '''Converts a list of floating point values into a PyTorch tensor
    '''
    def __init__(self, dtype=None):
        self.dtype = dtype
    
    def __call__(self, _node):
        return torch.tensor(_node).view(-1, 1).to(self.dtype)


### For desccriptive features
# class SequenceEncoder(object):
#     def __init__(self, model, device=None):
#         self.device = device
#         self.model = model
#         # self.model = SentenceTransformer(model_name, device=device)

#     @torch.no_grad()
#     def __call__(self, df):
#         x = self.model.encode(df.values, show_progress_bar=True,
#                               convert_to_tensor=True, device=self.device)
#         return x.cpu()