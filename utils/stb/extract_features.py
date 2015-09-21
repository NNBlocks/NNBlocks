def extract_features(t):
        """Extracts all the features for the RNN.
        Returns a tuple.
        Each element in the tuple is a feature:
            0 - The sentence. This is a list of string tokens.
            1 - The comp tree of the sentence
            2 - The label of the sentence. This a list of integers 
                representing the class of each node in the tree, including the
                leafs
        """
        if isinstance(t.value,str):
            return ([t.value], [], [int(t.label)])

        lch = t.value[0].get_features()
        rch = t.value[1].get_features()

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
                leafs_labels + internal_labels + [int(t.label)])
 
