# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 20:15:20 2016

@author: akormilitzin
"""

class Vector2Sigtures(object):       
    def __init__(self, data, windSize, depressiveThreshold):
        # constants
        self.depressiveThreshold = depressiveThreshold    
        self.minNumberOfDataPoints = 1
        self.windSize = windSize
        self.patternOfStartOfdepression = array([False, True,True])
        self.patternOfEndOfdepression = array([True,True,False])
        self.maxWeeksGapToConcatenate = 4
        self.maxDiffNumberOfHighScoresAllowed = 2 # if of 10 days, 6 or 4 are above 10, than it is depression
        # variables
        self.concatenetedBoolArray = []
        self.ti = arange(windSize).astype(float) # integer time
        self.tiz = arange(windSize+1).astype(float) # integer time +1 (zero)
        self.data = data[where(~isnan(data))[0][0]:] # skip all nans from the beginning
        self.data_windowed = self.windower(self.data)
        self.size_of_matrix = shape(self.data_windowed)
        self.data_filled_in = pd.DataFrame(self.data).ffill().as_matrix().flatten()
        self.data_filled_in_windowed = self.windower(self.data_filled_in)
        self.data_filled_in_windowed_cumsum_with_zero = cumsum(insert(self.data_filled_in_windowed, 0,0,axis=1),axis=1).astype(float)
        self.missing_values = isnan(self.data)
        self.missing_values_windowed = self.windower(self.missing_values)
        self.missing_values_cumsum_windowed = cumsum(self.missing_values_windowed, axis=1).astype(float)
        self.missing_values_cumsum_windowed_with_zero = cumsum(insert(self.missing_values_cumsum_windowed, 0,0,axis=1),axis=1).astype(float)
        self.boolArray = self.data > self.depressiveThreshold
        # procedures
        self.startEpisode, self.startWellness, self.concatenetedBoolArray, self.newEpisodes = self.find_episodes()
        self.y_matrix, self.boolMatrix, self.prec_indx = self.find_precursors()
        self.y = self.y_matrix[:,-1]
        self.number_of_episodes = len(self.newEpisodes)
        self.number_of_precursors = sum(self.y)
        #self.args = args
        #self.ti = range(len(self.args))
        #self.q = pd.DataFrame(self.args).ffill().as_matrix().flatten()
        #self.m = isnan(self.args)
        # cszq = cumsum of qids with augmented z: cumsum[0 qids]
        #self.cszq = pd.DataFrame(insert(self.args,0,0.0)).ffill().as_matrix().flatten().cumsum()
        #self.csm = cumsum(self.m)

    def windower(self, argument):
        return as_strided(argument,shape=[len(argument)-self.windSize+1,self.windSize],strides=[argument.strides[0],argument.strides[0]])

    def find_subarray_in_array(self, sub_array, large_array):
        large_array_view = as_strided(large_array, shape=(len(large_array) - len(sub_array) + 1, len(sub_array)), strides=(large_array.dtype.itemsize,) * 2)
        return where(numpy.all(large_array_view == sub_array, axis=1))[0]

    def find_episodes(self):
        # add extra False at the beginning and the end
        tmpBoolArray = np.insert(self.boolArray,(0,len(self.boolArray)),False)
        startOfEpisodeOneWeekEarlier = []
        startOfEpisode = self.find_subarray_in_array(self.patternOfStartOfdepression, tmpBoolArray)+1
        startOfWellness = self.find_subarray_in_array(self.patternOfEndOfdepression, tmpBoolArray)+2
        if ( (size(startOfEpisode)!=0) | (size(startOfWellness)!=0) ):
            for i in range(len(startOfEpisode)-1):
                tmpVal = tmpBoolArray[startOfWellness[i]:startOfEpisode[i+1]]
                if len(tmpVal) <= self.maxWeeksGapToConcatenate:
                    tmpVal = True
                elif np.abs(tmpVal.sum() - len(tmpVal)) <= self.maxDiffNumberOfHighScoresAllowed:
                    tmpVal = True
                else:
                    tmpVal = False
                tmpBoolArray[startOfWellness[i]:startOfEpisode[i+1]] = tmpVal
            tmpBoolArray[startOfWellness[-1]:] = False
            tmpBoolArray[:startOfEpisode[0]] = False
            # must shift the beginning of the episodes one week earlier:
            startOfEpisodeOneWeekEarlier = self.find_subarray_in_array(self.patternOfStartOfdepression, tmpBoolArray)
            tmpBoolArray[startOfEpisodeOneWeekEarlier] = True
            self.concatenetedBoolArray = tmpBoolArray[1:-1]
        return startOfEpisode, startOfWellness, tmpBoolArray[1:-1], startOfEpisodeOneWeekEarlier
    
    def find_precursors(self):
        y_in = zeros(self.size_of_matrix).astype(bool)
        b_in = zeros(self.size_of_matrix).astype(bool)
        precIndices_in = empty(1)
        if size(self.newEpisodes) > 0:
            b_in = self.windower(self.concatenetedBoolArray)
            precIndices_in = where(all(b_in == pad([True],pad_width=(b_in.shape[1]-1,0), mode='constant',constant_values=False), axis=1))
            y_in = zeros_like(b_in).astype(int)
            y_in[precIndices_in[0]] = b_in[precIndices_in[0]]        
        return y_in, b_in, precIndices_in[0]

    def lead_lagger(self, pos, *args):
        outPut = []
        if size(pos)>0:
            tmpMatrix = repeat(array(args),2,axis=1).T
            tmpMatrix[:,pos] = roll(tmpMatrix[:,pos],1,axis=0)
            outPut = tmpMatrix[1:]
        else:
            outPut = array(args).T
        return outPut 
        
#    def plot_qids_with_depression():
#    #z = z.sort_values(by='scheduleOpenedAt')
#   ymin = 0
#   ymax = 28
#   plot(z.response_date, z.summary, 'b:o', markersize=10)
#    plot(z.response_date, self.depressiveThreshold*ones_like(range(z.response_date.shape[0])),'r-')        
#    #plot(z.response_date,((z.response_date - z.scheduleOpenedAt)/pd.Timedelta('1 hour')),'rs')
#    startEpisode, startWellness, boolArray, newEpisodes = old_find_episodes(z.summary.as_matrix(), threshold)
#    xcoords = z.response_date[boolArray]
#    for xc in xcoords:
#        plt.axvline(x=xc,color='k',linestyle='dashed')
#    axes = plt.gca()
#    plot(z.response_date[isnan(z.summary)], z.summary.ffill()[isnan(z.summary)], 'mo', markersize=5)
#    #axes.set_xlim([xmin,xmax])
#    axes.set_ylim([ymin,ymax])

        
###############################################################################        
###############################################################################
        
    def sigtures__ti__q(self, sigLevel):
        numOfVectors = 2
        leadLagFlag = []
        if self.data.shape[0] > self.minNumberOfDataPoints:
            tmpMat = zeros((self.data_filled_in_windowed.shape[0], ts.sigdim(numOfVectors, sigLevel)));     
            for ii in range(tmpMat.shape[0]):
                q = self.data_filled_in_windowed[ii]
                tmpVal = self.lead_lagger(leadLagFlag, self.ti, q)
                sigtures = ts.stream2sig(ascontiguousarray(tmpVal), sigLevel)
                tmpMat[ii,:] = sigtures
        else:
            tmpMat = []
        return tmpMat

###############################################################################        
###############################################################################
        
    def sigtures__q_lead__q_lag(self, sigLevel):
        numOfVectors = 2
        leadLagFlag = [1]            
        if self.data.shape[0] > self.minNumberOfDataPoints:
            tmpMat = zeros((self.data_filled_in_windowed.shape[0], ts.sigdim(numOfVectors, sigLevel)));     
            for ii in range(tmpMat.shape[0]):
                q = self.data_filled_in_windowed[ii]
                tmpVal = self.lead_lagger(leadLagFlag, q, q)
                sigtures = ts.stream2sig(ascontiguousarray(tmpVal), sigLevel)
                tmpMat[ii,:] = sigtures
        else:
            tmpMat = []
        return tmpMat     

###############################################################################        
###############################################################################

    def sigtures__cszq_lead__cszq_lag(self, sigLevel):
        numOfVectors = 2
        leadLagFlag = [1]
        if self.data.shape[0] > self.minNumberOfDataPoints:
            tmpMat = zeros((self.data_filled_in_windowed_cumsum_with_zero.shape[0], ts.sigdim(numOfVectors, sigLevel)));     
            for ii in range(tmpMat.shape[0]):
                csqz = self.data_filled_in_windowed_cumsum_with_zero[ii]
                tmpVal = self.lead_lagger(leadLagFlag, csqz, csqz)
                sigtures = ts.stream2sig(ascontiguousarray(tmpVal), sigLevel)
                tmpMat[ii,:] = sigtures
        else:
            tmpMat = []
        return tmpMat    
        
###############################################################################        
###############################################################################
        
    def sigtures__ti__q__csm(self, sigLevel):
        numOfVectors = 3
        leadLagFlag = []
        if self.data.shape[0] > self.minNumberOfDataPoints:
            tmpMat = zeros((self.data_filled_in_windowed.shape[0], ts.sigdim(numOfVectors, sigLevel)));     
            for ii in range(tmpMat.shape[0]):
                q = self.data_filled_in_windowed[ii]
                csm = self.missing_values_cumsum_windowed[ii]
                tmpVal = self.lead_lagger(leadLagFlag, self.ti, q, csm)
                sigtures = ts.stream2sig(ascontiguousarray(tmpVal), sigLevel)
                tmpMat[ii,:] = sigtures
        else:
            tmpMat = []
        return tmpMat            
        
###############################################################################        
###############################################################################

    def sigtures__q_lead__q_lag__csm(self, sigLevel):
        numOfVectors = 3
        leadLagFlag = [1]
        if self.data.shape[0] > self.minNumberOfDataPoints:
            tmpMat = zeros((self.data_filled_in_windowed.shape[0], ts.sigdim(numOfVectors, sigLevel)));     
            for ii in range(tmpMat.shape[0]):
                q = self.data_filled_in_windowed[ii]
                csm = self.missing_values_cumsum_windowed[ii]
                tmpVal = self.lead_lagger(leadLagFlag, q, q, csm)
                sigtures = ts.stream2sig(ascontiguousarray(tmpVal), sigLevel)
                tmpMat[ii,:] = sigtures
        else:
            tmpMat = []
        return tmpMat

###############################################################################        
###############################################################################

    def sigtures__cszq_lead__cszq_lag__csmz(self, sigLevel):
        numOfVectors = 3
        leadLagFlag = [1]
        if self.data.shape[0] > self.minNumberOfDataPoints:
            tmpMat = zeros((self.data_filled_in_windowed_cumsum_with_zero.shape[0], ts.sigdim(numOfVectors, sigLevel)));     
            for ii in range(tmpMat.shape[0]):
                csqz = self.data_filled_in_windowed_cumsum_with_zero[ii]
                csmz = self.missing_values_cumsum_windowed_with_zero[ii]
                tmpVal = self.lead_lagger(leadLagFlag, csqz, csqz, csmz)
                sigtures = ts.stream2sig(ascontiguousarray(tmpVal), sigLevel)
                tmpMat[ii,:] = sigtures
        else:
            tmpMat = []
        return tmpMat    
        
###############################################################################        
###############################################################################

    def sigtures__tiz__cszq_lead__cszq_lag__csmz(self, sigLevel):
        numOfVectors = 4
        leadLagFlag = [2]
        if self.data.shape[0] > self.minNumberOfDataPoints:
            tmpMat = zeros((self.data_filled_in_windowed_cumsum_with_zero.shape[0], ts.sigdim(numOfVectors, sigLevel)));     
            for ii in range(tmpMat.shape[0]):
                csqz = self.data_filled_in_windowed_cumsum_with_zero[ii]
                csmz = self.missing_values_cumsum_windowed_with_zero[ii]
                tmpVal = self.lead_lagger(leadLagFlag, self.tiz, csqz, csqz, csmz)
                sigtures = ts.stream2sig(ascontiguousarray(tmpVal), sigLevel)
                tmpMat[ii,:] = sigtures
        else:
            tmpMat = []
        return tmpMat

###############################################################################        
###############################################################################
    
    def sigtures__ti__q_lead__q_lag__csm(self, sigLevel):
        numOfVectors = 4
        leadLagFlag = [2]
        if self.data.shape[0] > self.minNumberOfDataPoints:
            tmpMat = zeros((self.data_filled_in_windowed.shape[0], ts.sigdim(numOfVectors, sigLevel)));     
            for ii in range(tmpMat.shape[0]):
                q = self.data_filled_in_windowed[ii]
                csm = self.missing_values_cumsum_windowed[ii]
                tmpVal = self.lead_lagger(leadLagFlag, self.ti, q, q, csm)
                sigtures = ts.stream2sig(ascontiguousarray(tmpVal), sigLevel)
                tmpMat[ii,:] = sigtures
        else:
            tmpMat = []
        return tmpMat
        # return 4, lead_lagger([2], self.ti, self.q, self.q, self.csm)