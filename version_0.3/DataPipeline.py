#!/usr/bin/env python

__author__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__copyright__ = "Copyright 2020, Advanced Learning Models"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__email__ = "alvaro.gonzalez-jimenez@grenoble-inp.org"
__status__ = "Develop"

import pandas as pd
from collections import deque
from tqdm import tqdm


class DataPipeline:
    def __init__(self, filename):
        # Read the column of sequences in the given file
        self.X = pd.read_csv(filename)["seq"]
        # Initialize the data with the given sequence
        self.data = self.X
        # Dictionary to store the kmers
        self.kmers = {}
        # Dictionary to store the neighborhoods in order to compute the mismatch
        self.neighborhoods = {}
        # Dictionary to store the precomputed neighboorhood
        self.precomputed = {}
        # The DNA alphabet
        self.alphabet = "GATC"

    def compute_k_mers(self, k):
        """
        @param k: Size of the kmer
        Given a number K return the possibles kmers in the data
        """
        dimensions = len(self.X[0])
        index = 0
        for x in tqdm(self.X):
            for j in range(dimensions - k + 1):
                kmer = x[j : j + k]
                if kmer not in self.kmers:
                    self.kmers[kmer] = kmer
                    # index += 1

    # {'GGG': 34, 'GAG': 21, 'GAA': 10, 'GAC': 21, 'GTG': 23, 'GCG': 28, 'TAG': 15, 'CAG': 21, 'AGG': 22, 'AGA': 8, 'AGT': 11, 'AGC': 25, 'ATG': 16, 'TGG': 25, 'GGA': 27, 'GGT': 25, 'GGC': 29, 'GTC': 17, 'GCC': 30, 'TGC': 21, 'CGC': 21, 'GTT': 9, 'GCA': 20, 'GCT': 19, 'ACT': 15, 'TCT': 12, 'CCT': 20, 'TTG': 17, 'CTG': 19, 'CTA': 15, 'CTT': 15, 'CTC': 17, 'CCG': 25, 'TGA': 15, 'TGT': 13, 'GTA': 8, 'ACC': 14, 'TCC': 17, 'CCC': 27, 'CAC': 17, 'CCA': 16, 'ACA': 17, 'TCA': 11, 'CAA': 15, 'CAT': 12, 'AAA': 5, 'AAC': 9, 'TAC': 10, 'AAT': 4, 'ATT': 2, 'TTC': 8, 'TTT': 4, 'CGT': 7, 'TTA': 1}
    def mismatch(self, k, m):
        """
         @param kmer: Subsequence to compute the mismatch
         @param m: Treshold to add the mismatches
         Given a specific kmer compute the neighboorhood and get the mismatches
         For each query kmer, compute possible all 2-mismatch kmers into a set.
         Interesect set from 1 with set from 2.
        """
        # print(self.kmers, len(self.kmers))
        # assert(False)
        n = self.X.shape[0]
        dimension = len(self.X[0])
        set_kmers = [{} for x in self.X]
        for i, x in enumerate(tqdm(self.X)):
            # print(x)
            for j in range(dimension - k + 1):
                kmer = x[j : j + k]
                if kmer not in self.precomputed:
                    neighborhood = self.neighborhood(kmer, m)
                    # print(self.kmers, len(self.kmers))
                    # assert(False)
                    # print(kmer)
                    # print(neighborhood, len(neighborhood))
                    # assert(False)
                    self.precomputed[kmer] = [
                        self.kmers[neighbor]
                        for neighbor in neighborhood
                        if neighbor in self.kmers
                    ]
                    # print(self.precomputed[kmer],len(self.precomputed[kmer]))
                    # assert (False)

                for index in self.precomputed[kmer]:
                    if index in set_kmers[i]:
                        set_kmers[i][index] += 1
                    else:
                        set_kmers[i][index] = 1
                    # print(set_kmers[i][index], len(set_kmers[i]))
                    # assert(i<3)
        # print(len(set_kmers))
        # print(neighborhood)
        # print(kmer,index)
        # assert False
        # print(set_kmers[i])
        # print(len(set_kmers[i]))
        # assert(i <3)
        # assert(False)
        # assert False, len(set_kmers[3])
        self.data = set_kmers

    def neighborhood(self, kmer, m):
        """
        @param kmer: Subsequence to compute the mismatch
        @param m: Treshold to add the mismatches
        Compute the possibles neighborhood for a given subsequence (kmer)
        and return it in an array.
        """
        set_mismatch = deque([(0, "")])
        for letter in kmer:
            number_candidates = len(set_mismatch)
            for i in range(number_candidates):
                mismatches, candidate = set_mismatch.popleft()
                if mismatches < m:
                    for a in self.alphabet:
                        if a == letter:
                            set_mismatch.append((mismatches, candidate + a))
                        else:
                            set_mismatch.append((mismatches + 1, candidate + a))
                if mismatches == m:
                    set_mismatch.append((mismatches, candidate + letter))
        return [candidate for mismatches, candidate in set_mismatch]
