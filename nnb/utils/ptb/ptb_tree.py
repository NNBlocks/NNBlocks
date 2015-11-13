# NNBlocks is a Deep Learning framework for computational linguistics.
#
#   Copyright (C) 2015 Frederico Tommasi Caroli
#
#   NNBlocks is free software: you can redistribute it and/or modify it under
#   the terms of the GNU General Public License as published by the Free
#   Software Foundation, either version 3 of the License, or (at your option)
#   any later version.
#
#   NNBlocks is distributed in the hope that it will be useful, but WITHOUT ANY
#   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
#   details.
#
#   You should have received a copy of the GNU General Public License along with
#   NNBlocks. If not, see http://www.gnu.org/licenses/.

import numpy as np
import copy

class PTBTreeNode:
    def __init__(self, label, value):
        self.label = label
        self.value = value

    def get_features(self):
        """Extracts all the features for the RNN.
        Returns a tuple.
        Each element in the tuple is a feature:
            0 - The sentence. This is a list of string tokens.
            1 - The comp tree of the sentence
            2 - The label of the sentence. This a list of integers 
                representing the label of each node in the tree, including the
                leafs
        """
        if isinstance(self.value,str):
            return ([self.value], [], [self.label])

        lch = self.value[0].get_features()
        rch = self.value[1].get_features()

        lcomptree = lch[1]
        rcomptree = rch[1]

        lleafs_nr = len(lcomptree) + 1
        rleafs_nr = len(rcomptree) + 1

        leafs_labels = lch[2][:lleafs_nr] + rch[2][:rleafs_nr]
        internal_labels = lch[2][lleafs_nr:] + rch[2][rleafs_nr:]

        for children in lcomptree:
            for i in range(2):
                if children[i] >= lleafs_nr:
                    children[i] += rleafs_nr

        for children in rcomptree:
            for i in range(2):
                if children[i] < rleafs_nr:
                    children[i] += lleafs_nr
                else:
                    children[i] += 2*(lleafs_nr) - 1

        newnode = []

        if lleafs_nr == 1:
            newnode.append(0)
        else:
            newnode.append(2 * lleafs_nr - 2 + rleafs_nr)

        if rleafs_nr == 1:
            newnode.append(lleafs_nr)
        else:
            newnode.append(2 * (lleafs_nr + rleafs_nr) - 3)

        thiscomptree = lcomptree + rcomptree + [newnode]

        return (lch[0] + rch[0], 
                thiscomptree , 
                leafs_labels + internal_labels + [self.label])
                

    def penn_print(self, tabs=0):
        if isinstance(self.value, str):
            print ("\t"*tabs) + "(" + self.label + " " + self.value + ")"
        else:
            print ("\t"*tabs) + "(" + self.label
            for child in self.value:
                child.penn_print(tabs+1)
            print ("\t"*tabs) + ")"
            
    def plain(self):
        while not isinstance(self.value,str) and len(self.value) == 1:
            self.value = self.value[0].value
        if not isinstance(self.value, str):
            for v in self.value:
                v.plain()

