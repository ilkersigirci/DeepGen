import torch
import numpy as np
import random


class Attribute:

    def __init__(self):

        self.selected_attr_indices = [4, 5, 8, 9, 11, 12, 15, 20, 21, 22, 24, 26, 39]
        self.attr_num = len(self.selected_attr_indices)

        # FIXME: one hair color must set to 1?" -> YES
        # NOTE: bald is also included
        self.hair_color_indices = [0, 2, 3, 4]

        self.selected_attr_list = ["bald", "bangs", "black_hair", "blond_hair", "brown_hair",
                                    "bushy_eyebrows", "eyeglasses", "male", "mouth_open",
                                    "mustache", "no_beard", "pale_skin","young"]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def transform_original_attributes(self, attr):

        attr = attr[:, self.selected_attr_indices]
        attr.to(self.device)

        return attr

    def generate(self):

        attr = torch.zeros(self.attr_num)

        hair_color_index = random.choice(self.hair_color_indices)

        attr[hair_color_index] = 1  # random.randint(0,1)

        for index in self.selected_attr_list:

            if index in self.hair_color_indices:
                continue

            # Handle bald and bangs collision
            if index == 1 and attr[0] == 1:
                continue

            attr[index] = random.randint(0, 1)

        attr.to(self.device)

        return attr

    def generate_from_attr_names(self, names):

        attr = torch.zeros(self.attr_num)

        for name in names:

            assert name in self.selected_attr_list

            for index, attr_name in enumerate(self.selected_attr_list):

                if attr_name == name:
                    attr[index] = 1

        attr.to(self.device)

        return attr

    def get_attr_names(self, attr_array):

        assert attr_array.shape[0] == self.attr_num

        attr = attr_array.cpu().numpy()

        indices = np.where(attr == 1)[0]

        assert len(indices) > 0

        return self.selected_attr_list[indices]

    def get_attr_difference_names(self, attr_s, attr_t):

        assert attr_s.shape[0] == self.attr_num
        assert attr_t.shape[0] == self.attr_num

        added = []
        removed = []

        for index, attr_name in enumerate(self.selected_attr_list):

            if attr_s[index] == 0 and attr_t[index] == 1:
                added.append(attr_name)

            elif attr_s[index] == 1 and attr_t[index] == 0:
                removed.append(attr_name)

        return added, removed

## USAGE
# attr_class = Attribute()
# generated = attr_class.generate()
# print(attr_class.get_attr_names(generated))
