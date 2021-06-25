import torch
import numpy as np
import random


class Attribute:

    def __init__(self):
        self.target_attr_list = {5: "bald", 6: "bangs", 9: "black_hair", 10: "blond_hair", 12: "brown_hair",
                                 13: "bushy_eyebrows", 16: "eyeglasses",
                                 21: "male", 22: "mouth_open", 23: "mustache", 25: "no_beard", 27: "pale_skin",
                                 40: "young"}

        # FIXME: one hair color must set to 1?" -> YES
        # NOTE: bald is also included
        self.hair_color_keys = [5, 9, 10, 12]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.selected_attr_indices = [4, 5, 8, 9, 11, 12, 15, 20, 21, 22, 24, 26, 39]

    def generate(self):

        attr = torch.zeros(40)

        hair_color_key = random.choice(self.hair_color_keys)

        attr[hair_color_key - 1] = 1  # random.randint(0,1)

        for key in self.target_attr_list.keys():

            if key in self.hair_color_keys:
                continue

            # Handle bald and bangs collision
            if key == 6 and attr[4] == 1:
                continue

            attr[key - 1] = random.randint(0, 1)

        # attr = attr[self.selected_attr_indices]

        attr.to(self.device)

        return attr

    def generate_from_attr_names(self, names):

        attr = torch.zeros(40)

        for name in names:

            assert name in self.target_attr_list.values()

            for (key, value) in self.target_attr_list.items():

                if value == name:
                    attr[key - 1] = 1

        attr = attr[self.selected_attr_indices]
        attr.to(self.device)

        return attr

    def get_attr_names(self, attr_array):

        assert attr_array.shape[0] == 40
        # assert attr_array.shape[0] == 13

        attr = attr_array.cpu().numpy()

        indices = np.where(attr == 1)[0]

        assert len(indices) > 0

        names = [self.target_attr_list[index + 1] for index in indices]

        return names

    def get_attr_difference_names(self, attr_s, attr_t):

        added = []
        removed = []

        for (key, value) in self.target_attr_list:

            if attr_s[key - 1] == 0 and attr_t[key - 1] == 1:
                added.append(value)

            elif attr_s[key - 1] == 1 and attr_t[key - 1] == 0:
                removed.append(value)

        return added, removed

## USAGE
# attr_class = Attribute()
# generated = attr_class.generate()
# print(attr_class.get_attr_names(generated))
