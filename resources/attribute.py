import torch
import numpy as np
import random

class Attribute:

    def __init__(self):
        self.target_attr_list = {5 : "bald", 6 : "bangs", 9 : "black_hair", 10 : "blond_hair", 12 : "brown_hair", 13 : "bushy_eyebrows", 16 : "eyeglasses",
                                21 : "male", 22 : "mouth_open", 23 : "mustache", 25 : "no_beard", 27 : "pale_skin", 40 : "young"}
        
        # FIXME: one hair color must set to 1?"
        # NOTE: bald is also included
        self.hair_color_keys = [5, 9, 10, 12]

    def generate(self):

        attr = torch.zeros(40)

        hair_color_key = random.choice(self.hair_color_keys)

        attr[hair_color_key - 1] = random.randint(0,1)

        for key in self.target_attr_list.keys():

            if key in self.hair_color_keys:
                continue
            
            # Handle bald and bangs collision
            if key == 6 and attr[4] == 1:
                continue

            attr[key - 1] = random.randint(0,1)

        return attr

    
    def generate_from_attr_names(self, names):

        attr = torch.zeros(40)

        for name in names:

            assert name in self.target_attr_list.values()

            for (key, value) in self.target_attr_list.items():

                if value == name:
                    attr[key - 1] = 1
                
        return attr

    def get_attr_names(self, attr_array):

        assert attr_array.shape[0] == 40

        attr  = attr_array.numpy()

        indices = np.where(attr == 1)[0]

        assert len(indices) > 0

        names =  [self.target_attr_list[index+1] for index in indices]

        return names

## USAGE
#attr_class = Attribute()
#generated = attr_class.generate()
#print(attr_class.get_attr_names(generated))