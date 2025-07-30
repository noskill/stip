import torch, copy                                                                                                                                
from transformers import LlamaConfig, LlamaForCausalLM                                                                                            
from utils import dense_orthogonal_R, _rotate_model_parameters_with_R, replace_rms_with_rotated                                                   
                                                                                                                                                  
dim=512                                                                                                                                           
cfg=LlamaConfig(hidden_size=dim,intermediate_size=dim*2,num_hidden_layers=1,num_attention_heads=8,vocab_size=100)                                 
base=LlamaForCausalLM(cfg).eval()                                                         
rot=copy.deepcopy(base)                                                                   
R=dense_orthogonal_R(dim, dtype=torch.float32, seed=0)                                    
_rotate_model_parameters_with_R(rot,R)                                                    
replace_rms_with_rotated(rot,R)                                                                                                                                                          
ids=torch.randint(100,(1,5))                                                                                                                                                             
with torch.no_grad():                                                                                                                                                                    
    lb=base(ids).logits[:,-1,:]                                                                                                                                                          
    lr=rot(ids).logits[:,-1,:]                                                                                                                                                           
print((lb-lr).abs().max().item()) 
