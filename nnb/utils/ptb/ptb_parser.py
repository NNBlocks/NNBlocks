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

from nnb.utils.ptb import PTBTreeNode

class PTBTokenizer:
    def __init__(self):
        self.index = 0
        self.line = None

    def set_line(self, line):
        self.line = line.strip()
        self.index = 0

    def next_token(self):
        if self.line == None:
            raise ValueError(
                "Set a line first with Tokenizer.set_line(self,line)"
            )

        while self.index < len(self.line):
            c = self.line[self.index]
            if c == ' ' or c == '\n':
                self.index += 1
                continue
            if c == '(' or c == ')':
                self.index += 1
                return c
            cb = self.index
            def is_stop_char(c):
                return c == ')' or c == '(' or c == ' ' or c == '\n'
            while self.index < len(self.line) and \
                    not is_stop_char(self.line[self.index]):
                self.index += 1
            return self.line[cb:self.index]

        return None

class PTBParser:
    def __init__(self, filename=None):
        self.read_file = None
        if filename is not None:
            self.read_file = open(filename,'r')
        self.t = PTBTokenizer()

    def __del__(self):
        if self.read_file is not None:
            self.read_file.close()

    def parse(self, string=None, new_tree=True):
        if new_tree and self.read_file is not None:
            line = self.read_file.readline()
            self.t.set_line(line)
            self.t.next_token()

        if string is not None:
            self.t.set_line(string)
            self.t.next_token()

        if self.read_file is None and string is None and new_tree == True:
            raise ValueError(
                'Parser instantiated without a file name. Either specify a ' + \
                'string to parse with the "string" parameter or instantiate' + \
                ' a Parser with the "filename" parameter'
            )

        token = self.t.next_token()
        label = None
        label_read = False
        value = None
        while token is not None:
            if token == '(':
                if value is None:
                    value = [self.parse(new_tree=False)]
                else:
                    value += [self.parse(new_tree=False)]
            elif token != ')':
                if not label_read:
                    label = token
                    label_read = True
                else:
                    value = token
            else:
                return PTBTreeNode(label,value)
            token = self.t.next_token()

