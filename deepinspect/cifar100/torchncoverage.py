class NCoverage():  # Model under test

    def __init__(self, threshold = 0.2, exclude_layer=['pool', 'fc', 'flatten'], only_layer = ""):       
        
        self.cov_dict = {}
        self.threshold = threshold    

    def scale(self, layer_outputs, rmax=1, rmin=0):
        '''
        scale the intermediate layer's output between 0 and 1
        :param layer_outputs: the layer's output tensor
        :param rmax: the upper bound of scale
        :param rmin: the lower bound of scale
        :return:
        '''
        divider = (layer_outputs.max() - layer_outputs.min())
        if divider == 0:
            return np.zeros(shape=layer_outputs.shape)
        X_std = (layer_outputs - layer_outputs.min()) / divider
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled


    def get_channel_coverage(self, layer_outputs):
        covered_channel = []
        #print("debug")
        #print(layer_outputs.shape)
        for layer_output in layer_outputs:
            scaled = self.scale(layer_output)
            for channel_idx in xrange(len(scaled)):
                #print(scaled[channel_idx].mean())
                if scaled[channel_idx].mean() > self.threshold:
                    if (channel_idx) not in covered_channel:
                        covered_channel.append(channel_idx)
        return covered_channel

    def get_channel_coverage_group(self, layer_outputs):
        covered_group = []
        
        #print("debug")
        #print(layer_outputs.shape)
        for layer_output in layer_outputs:
            covered_channel = []
            scaled = self.scale(layer_output)
            for channel_idx in xrange(len(scaled)):
                #print(scaled[channel_idx].mean())
                if scaled[channel_idx].mean() > self.threshold:
                    if (channel_idx) not in covered_channel:
                        covered_channel.append(channel_idx)
            covered_group.append(covered_channel)
        return covered_group