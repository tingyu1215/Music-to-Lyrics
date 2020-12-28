from pysnooper import snoop ;
from torch.utils.data import Dataset
import os
import numpy as np ;
goal_folder = os.path.dirname(os.path.abspath(__file__)) + "\\Sentence_and_Word_Parsing\\" ;

print(goal_folder) ;

goal_file= [goal_folder+i for i in os.listdir(goal_folder)] ;
#print(goal_file) ;

print(np.load(goal_file[0],allow_pickle=1))

class music_dataset(Dataset):
    def __init__(self,mode,tokenizer):
        assert mode in ["train","test"] ;
        self.mode = mode ;
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("") ;
