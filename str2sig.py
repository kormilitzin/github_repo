# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 22:07:49 2016

@author: akormilitzin
"""

class String2Sigtures(object):       
    def __init__(self, data, windSize):
        from sklearn import preprocessing
        # constants  
        self.windSize_ = windSize
        # input data
        labelEncoder = preprocessing.LabelEncoder()
        self.ti = arange(windSize).astype(float) # integer time
        self.data = data
        self.alphabet = sorted(list(set(data)))
        labelEncoder.fit(self.alphabet)
        self.alphabet_size = size(self.alphabet)
        self.data_arrayed = array(list(self.data))
        self.data_arrayed_categorical = labelEncoder.transform(self.data_arrayed)
        self.data_windowed = self.windower(self.data_arrayed, self.windSize_)
        self.data_windowed_categorical = self.windower(self.data_arrayed_categorical, self.windSize_).astype(float)
        data_plus_one = self.windower(self.data_arrayed, self.windSize_ + 1)
        self.data_windowed_plus_one = data_plus_one[:,:-1]
        data_plus_one_categorical = self.windower(self.data_arrayed_categorical, self.windSize_ + 1).astype(float)
        self.data_windowed_plus_one = data_plus_one[:,:-1]
        self.data_windowed_plus_one_categorical = data_plus_one_categorical[:,:-1]
        self.next_character = data_plus_one[:,-1]
        self.next_character_one_hot = self.chars_to_one_hot_encoding(self.next_character)
        self.next_character_categorical = labelEncoder.transform(self.next_character)

    def windower(self, argument, windowSize):
        return as_strided(argument,shape=[len(argument)-windowSize+1, windowSize], strides=[argument.strides[0], argument.strides[0]])
    
    def chars_to_one_hot_encoding(self, input_string):
        tmpOutput = []
        # alphabet_inside = self.alphabet
        # alphabet_size = size(alphabet_inside)
        for ii in range(self.alphabet_size):
            tmpOutput = append(tmpOutput, input_string == self.alphabet[ii])
        return reshape(tmpOutput, (self.alphabet_size, -1)).T.astype(bool)

    def string_to_path(self, input_string):
        tmpOutput = cumsum(self.chars_to_one_hot_encoding(input_string), axis = 0)
        return insert(tmpOutput, 0, zeros(self.alphabet_size), axis=0).astype(float)

    def data_windowed_to_sigtures(self, someData_windowed, sigLevel):
        tmpOut = []
        tmpOutZeroAxisSize = shape(someData_windowed)[0]
        for ii in range(tmpOutZeroAxisSize):
            tmpOut = append(tmpOut, ts.stream2sig(self.string_to_path(someData_windowed[ii]), sigLevel))
        tmpOut = reshape(tmpOut, (tmpOutZeroAxisSize,-1))    
        return tmpOut

    def string_to_sigtures(self, sigLevel, scal=False):
        tmpOut = self.data_windowed_to_sigtures(self.data_windowed, sigLevel)
        if scal:
            tmpOut = preprocessing.scale(tmpOut, axis=0)
        return tmpOut

    def string_to_sigtures_predict_next_char(self, sigLevel, scal=False):
        tmpOut = self.data_windowed_to_sigtures(self.data_windowed_plus_one, sigLevel)
        if scal:
            tmpOut = preprocessing.scale(tmpOut, axis=0)
        return tmpOut, self.next_character_categorical, self.next_character, self.next_character_one_hot