import copy
from functools import partial
import matplotlib.pyplot as plt
import math
import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
# from skimage.color import gray2rgb

from . import lime_base

# def segmentation_func(vec):
#     assert vec.size <= 9000

#     segments = np.zeros_like(vec)
#     size = 500
#     s = 0 
#     for i in range(0, vec.size, size):
#         segments[i:i+size] = s 
#         s += 1
    
#     return segments 


class VectorExplanation(object):
    def __init__(self, vector, segments):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.vector = vector
        self.segments = segments

        assert self.vector.shape == self.segments.shape 
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = None

    
    def get_vector_and_mask(self, 
                           label, 
                           positive_only=True, 
                           hide_rest=False,
                           num_features=5, 
                           min_weight=0.0):
        if label not in self.local_exp:
            raise ValueError('Label not in explanation')

        segments = self.segments
        vector = self.vector
        exp = self.local_exp[label]
        mask = np.zeros_like(segments)

        for e in exp:
            print(e)

        if hide_rest:
            temp = np.zeros_like(self.vector)
        else:
            temp = self.vector.copy()

        selected_segments_pos = []
        selected_segments_neg = []
        fs = [x[0] for x in exp if x[1] > 0][:num_features]

        for f in fs:
            indices = np.where(segments == f)[0]
            start, stop = indices.min(), indices.max()
            selected_segments_pos.append((start, stop))

        if not positive_only:
            fs = [x[0] for x in exp if x[1] <= 0]
            print(fs)
            fs = fs[:num_features]

            for f in fs:
                indices = np.where(segments == f)[0]
                start, stop = indices.min(), indices.max()
                selected_segments_neg.append((start, stop))

        return self.vector, selected_segments_pos, selected_segments_neg


class LimeVectorExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=0.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """
        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

    def explain_instance(self, vector, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, 
                         num_features=100000, 
                         num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None, peaks=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            vector: 1 dimension vector.

            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            
            labels: iterable with labels to be explained.
            
            hide_color: TODO
            
            top_labels: if not None, ignore labels and produce explanations for
                the 'top_labels' labels with highest prediction probabilities.

            num_features: maximum number of features present in explanation
            
            num_samples: size of the neighborhood to learn the linear model
            
            batch_size: TODO
            
            distance_metric: the distance metric to use for weights.
            
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            
            segm    entation_fn: SegmentationAlgorithm, wrapped skimage
                segmentation function
            
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.

        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """
        assert vector.ndim == 1

        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None:
            print('No segmentation_fn!!')
        try:
            dummy = peaks[0]
        except ValueError as e:
            raise e
        
        try:
            segments = segmentation_fn(vector)
        except ValueError as e:
            raise e


        fudged_vector = vector.copy()
        if hide_color is None:
            assert False 
        else:
            fudged_vector[:] = hide_color

        #return image.copy(), fudged_image
        

        # Create data set
        data, labels = self.data_labels(vector, fudged_vector, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size, peaks=peaks)
        
        top = labels
        
        #print(data.shape, data[0].reshape(1, -1))
    
        # distance original and permutations
        distances = sklearn.metrics.pairwise_distances(
            data, # data[0] orig, rest permuted
            data[0].reshape(1, -1),  # orig image
            metric=distance_metric
        ).ravel()

        ret_exp = VectorExplanation(vector, segments)


        if top_labels:
            # take 'top_labels' with highest confidence
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()

        for label in top:

            d = self.base.explain_instance_with_data(
                data, 
                labels,
                distances, 
                label, 
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection
            )

            ret_exp.intercept[label], ret_exp.local_exp[label], ret_exp.score, ret_exp.local_pred = d

        return ret_exp

    def resample_by_interpolation(self, signal, target_length):
        # output_fs = len(signal)
        # scale = output_fs / input_fs
        # # calculate new length of sample
        # n = round(len(signal) * scale)
        if target_length == 0:
            return []
        # use linear interpolation
        # endpoint keyword means than linspace doesn't go all the way to 1.0
        # If it did, there are some off-by-one errors
        # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
        # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
        # Both are OK, but since resampling will often involve
        # exact ratios (i.e. for 44100 to 22050 or vice versa)
        # using endpoint=False gets less noise in the resampled sound
        resampled_signal = np.interp(
            np.linspace(0.0, 1.0, target_length, endpoint=False),  # where to interpret
            np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
            signal,  # known data points
        )
        # print(f'In resample, length = {len(resampled_signal)}')
        return resampled_signal

    def CropData(self, sample, num_samples=9000):
        # print(f'In CropData, num_samples={num_samples}')
        if len(sample) >= num_samples:
            start_idx = np.random.randint(len(sample) - num_samples + 1)
            sample = sample[start_idx : start_idx + num_samples]
        else:
            left_pad = int(np.ceil((num_samples - len(sample)) / 2))
            right_pad = int(np.floor((num_samples - len(sample)) / 2))
            sample = np.pad(sample, (left_pad, right_pad), "constant")

        return sample

    def align(self, vec, peaks):

        current = 0 
        aligned = []
        size = int(math.ceil(np.median(peaks[1:] - peaks[:-1])))
        peaks = list(peaks)
        peaks.append(len(vec) - 1)
        # print(peaks)
        for next_peak in peaks:
            segment = vec[current:next_peak]
            # print(current, next_peak, len(segment))
            if len(segment) == 0:
                resampled = []
            else:
                resampled = self.resample_by_interpolation(segment, size)
            aligned.extend(resampled)
            
            current = next_peak
            
        return np.asarray(aligned)

    def masking_linear_interpolation(self, vec, peaks, active_segments):

        # print(f'org size = {len(peaks)}')
        if  np.sum(active_segments) == len(active_segments):
            return vec.copy()
        current = 0 
        n_segments = len(active_segments)
        masked_ecg = vec.copy()
        # print(len(masked_ecg))
        if peaks[0] == 0:
            peaks = peaks[1:]
        for next_peak in peaks:
            size = next_peak - current
            
            segment_size = size // n_segments
            rest = size % n_segments 
            for i in range(n_segments - 1):
                if active_segments[i] == 0:
                    s = int(current + i*segment_size)
                    e = int(s + segment_size)
                    dummy_ip = [masked_ecg[s], masked_ecg[e - 1]]
                    if dummy_ip[0] > dummy_ip[1]:
                        dummy_ip = dummy_ip[::-1]
                        dummy = np.interp(np.linspace(dummy_ip[0], dummy_ip[1], segment_size), dummy_ip, dummy_ip)[::-1]
                    else:
                        dummy = np.interp(np.linspace(dummy_ip[0], dummy_ip[1], segment_size), dummy_ip, dummy_ip)
                    # print(len(dummy))
                    # print(len(masked_ecg[s:e]))
                    masked_ecg[s:e] = dummy
            if active_segments[-1] == 0:
                s = int(current + (n_segments - 1)*segment_size)
                e = int(s + segment_size + rest)
                dummy_ip = [masked_ecg[s], masked_ecg[e - 1]]
                if dummy_ip[0] > dummy_ip[1]:
                    dummy_ip = dummy_ip[::-1]
                    dummy = np.interp(np.linspace(dummy_ip[0], dummy_ip[1], segment_size + rest), dummy_ip, dummy_ip)[::-1]
                else:
                    dummy = np.interp(np.linspace(dummy_ip[0], dummy_ip[1], segment_size + rest), dummy_ip, dummy_ip)
                # print(len(dummy))
                # print(len(masked_ecg[s:e]))
                masked_ecg[s:e] = dummy
            current = next_peak
        # print(f'new size = {len(masked_ecg)}')
        return masked_ecg

    def get_segment_idxs(self, vec, peaks, n_segments):
        peaks = list(peaks)
        # print(peaks)
        # print(len(vec))
        idxs = []
        current = peaks[0]
        # current = 0
        # if peaks[0] == 0:
        #     peaks = peaks[1:]
        # if peaks[-1] != len(vec) - 1:
        #     peaks.append(len(vec) - 1)
        # print(peaks)
        for next_peak in peaks[1:]:
            size = next_peak - current
            segment_size = size // n_segments
            rest = size % n_segments

            for i in range(n_segments):
                s = int(current + i*segment_size)
                idxs.append({'segnum':i, 'idx':s})
            current = next_peak
        idxs.append({'segnum':0, 'idx':current})
        return idxs

    def masking_linear_interpolation_cont(self, vec, peaks, active_segments):
        # print(f'active_segments = {active_segments}')
        if  np.sum(active_segments) == len(active_segments):
            return vec.copy()
        # current = 0 
        n_segments = len(active_segments)
        masked_ecg = vec.copy()
        seg_idxs = self.get_segment_idxs(vec, peaks, n_segments)
        # segi = 0
        s = e = seg_idxs[0]['idx']
        ongoing = False
        for i in range(len(seg_idxs)):
            # print(seg_idxs[i])
            if active_segments[seg_idxs[i]['segnum']] == 1:
                if ongoing == True:
                    e = seg_idxs[i]['idx']
                    # print(f'In 1: s = {s}, e = {e}')
                    #Linear interpolation code
                    if e - 1 < 0:
                        continue
                    dummy_ip = [masked_ecg[s], masked_ecg[e - 1]]
                    if dummy_ip[0] > dummy_ip[1]:
                        dummy_ip = dummy_ip[::-1]
                        dummy = np.interp(np.linspace(dummy_ip[0], dummy_ip[1], e - s), dummy_ip, dummy_ip)[::-1]
                    else:
                        dummy = np.interp(np.linspace(dummy_ip[0], dummy_ip[1], e - s), dummy_ip, dummy_ip)
                    masked_ecg[s:e] = dummy
                ongoing = False
            else:
                if ongoing == False:
                    s = seg_idxs[i]['idx']
                    ongoing = True
                # print(f'In 0: s = {s}, e = {e}')
        return masked_ecg

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10, peaks=None):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image

            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            
            segments: segmentation of the image
            
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            
            num_samples: size of the neighborhood to learn the linear model
            
            batch_size: classifier_fn will be called on batches of this size.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        # print(f'In data_labels()...')
        # print(len(image), len(fudged_image), len(segments))
        # np.save('/home/pi242/xai/xai_results/lime_rr_res/' + 'image', image)
        # np.save('/home/pi242/xai/xai_results/lime_rr_res/' + 'fudged_image', fudged_image)
        # np.save('/home/pi242/xai/xai_results/lime_rr_res/' + 'segments', segments)
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * (n_features))
        data = data.reshape((num_samples, (n_features)))

        labels = []
        data[0, :] = 1
        imgs = []

        # print(data.shape)
        # print(data)

        for row in data:
            # print(d, row)
            # temp is the orig image
            # temp = copy.deepcopy(image)

            # # find index of superpixels which should be disabled  
            # zeros = np.where(row == 0)[0]  
            # mask = np.zeros(segments.shape, dtype=bool)

            # # for superpixel index in superpixels
            # for z in zeros:
            #     mask[segments == z] = True

            # # replace superpixel index we don't want in the image 
            # # with a predefined value. 
            # temp[mask] = fudged_image[mask]
            temp = self.masking_linear_interpolation_cont(image, peaks, row)

            imgs.append(temp)
            

            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []

        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)

        return data, np.array(labels)
