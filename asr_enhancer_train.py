'''
================================================================================
Aim: 
    Final script for Nassr
================================================================================
'''

# packages and libraries
import torch; import os; import random; import librosa; import torchaudio;
import pdb; import sys; import math; import numpy as np; import jiwer; 
import matplotlib.pyplot as plt; import torch.nn as nn; import pickle;

# Discriminator Architecture
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            torch.nn.Conv1d(in_channels = 1, 
                            out_channels = 16, 
                            kernel_size = 15, 
                            stride=1, 
                            groups=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels = 16, 
                            out_channels = 64, 
                            kernel_size = 41, 
                            stride=4, 
                            groups=4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels = 64, 
                            out_channels = 256, 
                            kernel_size = 41, 
                            stride=4, 
                            groups=16),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels = 256, 
                            out_channels = 1024, 
                            kernel_size = 41, 
                            stride = 4, 
                            groups = 64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels = 1024, 
                            out_channels = 1024, 
                            kernel_size = 41, 
                            stride = 4, 
                            groups = 256),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels = 1024, 
                            out_channels = 1024, 
                            kernel_size = 5, 
                            stride = 1, 
                            groups = 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels = 1024, 
                            out_channels = 1, 
                            kernel_size = 3, 
                            stride = 1, 
                            groups = 1)
        )
    def forward(self, x):
        x = self.discriminator(x)
        return x 

# U-Net Architecture
class UNet_Architecture(torch.nn.Module):
    def __init__(self):
        super(UNet_Architecture, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels  = 1,
                            out_channels = 16,
                            kernel_size  = 5,
                            stride       = 2,
                            padding      = 2),
            torch.nn.LeakyReLU(0.2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels  = 16,
                            out_channels = 32,
                            kernel_size  = 5,
                            stride       = 2,
                            padding      = 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(0.2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels  = 32,
                            out_channels = 64,
                            kernel_size  = 5,
                            stride       = 2,
                            padding      = 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels  = 64,
                            out_channels = 128,
                            kernel_size  = 5,
                            stride       = 2,
                            padding      = 2),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2))
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels  = 128,
                            out_channels = 256,
                            kernel_size  = 5,
                            stride       = 2,
                            padding      = 2),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2))
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels  = 256,
                            out_channels = 512,
                            kernel_size  = 5,
                            stride       = 2,
                            padding      = 2),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2))

        # discriminator
        self.uplayer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels  = 512,
                                     out_channels = 256,
                                     kernel_size  = 2,
                                     stride       = 2),
            torch.nn.BatchNorm2d(256),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU())
        self.uplayer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels  = 512,
                                     out_channels = 128,
                                     kernel_size  = 2,
                                     stride       = 2),
            torch.nn.BatchNorm2d(128),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU())
        self.uplayer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels  = 256,
                                     out_channels = 64,
                                     kernel_size  = 2,
                                     stride       = 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU())
        self.uplayer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels  = 128,
                                     out_channels = 32,
                                     kernel_size  = 2,
                                     stride       = 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.uplayer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels  = 64,
                                     out_channels = 16,
                                     kernel_size  = 2,
                                     stride       = 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU())
        self.uplayer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels  = 32,
                                     out_channels = 1,
                                     kernel_size  = 2,
                                     stride       = 2),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid())
        
    def forward(self, x):
        # passing through the down-pass
        layer1_output = self.layer1(x)
        layer2_output = self.layer2(layer1_output)
        layer3_output = self.layer3(layer2_output)
        layer4_output = self.layer4(layer3_output)
        layer5_output = self.layer5(layer4_output)
        layer6_output = self.layer6(layer5_output)

        # passing through the up-pass
        uplayer1_output = self.uplayer1(layer6_output)
        uplayer2_output = self.uplayer2(torch.cat((layer5_output, uplayer1_output), dim = 1)) 
        uplayer3_output = self.uplayer3(torch.cat((layer4_output, uplayer2_output), dim = 1)) 
        uplayer4_output = self.uplayer4(torch.cat((layer3_output, uplayer3_output), dim = 1)) 
        uplayer5_output = self.uplayer5(torch.cat((layer2_output, uplayer4_output), dim = 1))
        uplayer6_output = self.uplayer6(torch.cat((layer1_output, uplayer5_output), dim = 1)) 

        # returning
        return uplayer6_output

# Functions = Functions = Functions = Functions = Functions = Functions
# Functions = Functions = Functions = Functions = Functions = Functions
# Functions = Functions = Functions = Functions = Functions = Functions
# Functions = Functions = Functions = Functions = Functions = Functions
# Functions = Functions = Functions = Functions = Functions = Functions
def fFindWavFiles(directory_path):
    wav_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files 
def fCreateTensorFromWavPath(clip):
    '''
    ========================================================
    Aim: Loading wav-file and convert to torch tensor
    ========================================================
    '''
    # loading as numpy array
    audioarray, fs = librosa.load(clip, sr = 16000)

    # converting to torch tensor
    audioarray = torch.from_numpy(audioarray)

    return audioarray
def fCreateBatch(arg_list_of_files,
                 arg_number_of_files_to_take_from = 10,
                 arg_batch_size = 32,
                 arg_N_FFT = 1024,
                 arg_N_HOP = 768):
    '''
    ============================================================================
    Aim: Function to create a dataset tensor
    Notes:
        Take a list of files
        load audio corresponding to argument files
        create batches of data
    ============================================================================
    '''

    # calculating number of pictures to take from each clip
    num_pictures_from_each_clip = math.floor(arg_batch_size/arg_number_of_files_to_take_from)+1

    # going through the files until we create a batch 
    picture_count = 0
    return_tensor = []

    while picture_count<arg_batch_size:
        # loading audio
        clip = random.sample(arg_list_of_files,1)[0]    # choosing a clip
        audioarray = fCreateTensorFromWavPath(clip)     # fetching torch-tensor

        # take STFT
        spectrogram00 = stft(audioarray)[0:512, :].unsqueeze(0).unsqueeze(0)    # to update dimensions

        # fetch n-number of pictures
        pdb.set_trace
        if spectrogram00.shape[3]>=128:
            left_boundary = 0; right_boundary = spectrogram00.shape[3] - 128
            num_possible_starting_points = right_boundary - left_boundary + 1
            if num_possible_starting_points >= num_pictures_from_each_clip:
                startingpoints = random.sample(range(left_boundary, right_boundary), num_pictures_from_each_clip)
                picture_count += num_pictures_from_each_clip
            else:
                startingpoints = random.sample(range(left_boundary, right_boundary), num_possible_starting_points)
                picture_count += num_possible_starting_points
        else:
            continue

        # add to list
        for s_point in startingpoints:
            var00 = spectrogram00[:,:,:,s_point:s_point+128]
            return_tensor.append(var00)

    return_tensor = torch.cat(return_tensor, dim = 0)
    return return_tensor
def fRaiseSignalToPower(inputSignal, powervalue):
    '''
    ============================================================================
    Aim: Convert RMS of a signal to argument value
    ----------------------------------------------------------------------------
    Notes:
        Calculate current dB
        Calculate addition dB to be added
        Multiply with the corresponding value to reach the required dB
    ============================================================================
    '''
    # calculating current dB level
    current_dB = 10 * torch.log(torch.linalg.vector_norm(inputSignal)/math.sqrt(len(inputSignal)))

    # calculating difference in dB level to bring
    diff_dB = powervalue - current_dB

    # changing the power of the signal
    powerRaisedSignal = inputSignal * math.exp(diff_dB/10)
    return powerRaisedSignal
def fAdjustLengthOfSecondary(primarysignal, secondarysignal):
    if len(secondarysignal) > len(primarysignal):
        secondarysignal = secondarysignal[0:len(primarysignal)]
    else:
        num_zeros_to_add = len(primarysignal) - len(secondarysignal)
        zeroTensor = torch.zeros([num_zeros_to_add])
        secondarysignal = torch.cat((secondarysignal, zeroTensor), dim = 0)
    
    return secondarysignal
def fCreateBatchTwoSpeech(arg_list_of_files,
                            arg_number_of_files_to_take_from = 10,
                            arg_batch_size = 32,
                            arg_N_FFT = 1024,
                            arg_N_HOP = 768,
                            primarypower = -10,
                            secondarypower = -15):

    # calculating number of pictures to take from each clip
    num_pictures_from_each_clip = math.floor(arg_batch_size/arg_number_of_files_to_take_from)+1

    # going through the files until we create a batch 
    picture_count = 0
    return_tensor = []
    primary_tensor = []
    

    while picture_count<arg_batch_size:
        
        # loading primary audio
        clip                = random.sample(arg_list_of_files,1)[0]             # choosing a clip
        audioarrayPrimary   = fCreateTensorFromWavPath(clip)                    # fetching torch-tensor

        # loading secondary audio
        clip_secondary          = random.sample(arg_list_of_files,1)[0]         # choosing a clip
        audioarray_secondary    = fCreateTensorFromWavPath(clip_secondary)      # fetching secondary tensor

        # normalizing to a particular dB
        audioarrayPrimary       = fRaiseSignalToPower(audioarrayPrimary,    primarypower)
        audioarray_secondary    = fRaiseSignalToPower(audioarray_secondary, secondarypower)

        # adjusting lengths
        audioarray_secondary    = fAdjustLengthOfSecondary(audioarrayPrimary, audioarray_secondary)

        # adding them together
        audioarray = audioarrayPrimary + audioarray_secondary
        audioarray = audioarray.to(device)
        audioarrayPrimary = audioarrayPrimary.to(device)

        # amplitude normalizing the signals
        audioarray = audioarray/torch.max(torch.abs(audioarray))
        audioarrayPrimary = audioarrayPrimary/torch.max(torch.abs(audioarrayPrimary))
        
        # take STFT
        spectrogram00       = stft(audioarray)[0:512, :].unsqueeze(0).unsqueeze(0)    # to update dimensions
        spectrogramPrimary  = stft(audioarrayPrimary)[0:512, :].unsqueeze(0).unsqueeze(0)

        # fetch n-number of pictures
        if spectrogram00.shape[3]>=128:
            left_boundary = 0; right_boundary = spectrogram00.shape[3] - 128
            num_possible_starting_points = right_boundary - left_boundary + 1
            if num_possible_starting_points >= num_pictures_from_each_clip:
                startingpoints = random.sample(range(left_boundary, right_boundary), num_pictures_from_each_clip)
                picture_count += num_pictures_from_each_clip
            else:
                startingpoints = random.sample(range(left_boundary, right_boundary), num_possible_starting_points)
                picture_count += num_possible_starting_points
        else:
            continue

        # add to list
        for s_point in startingpoints:
            return_tensor.append(spectrogram00[:,:,:,       s_point:s_point+128])
            primary_tensor.append(spectrogramPrimary[:,:,:, s_point:s_point+128])

    # making tensor out of list
    return_tensor   = torch.cat(return_tensor, dim = 0)
    primary_tensor  = torch.cat(primary_tensor, dim = 0)

    # returning
    return return_tensor, primary_tensor
def fCreateListOfTwoSpeechTensors(arg_list_of_files,
                                  arg_numfilestoload = 1,
                                  primarypower = -10,
                                  secondarypower = -15):
    listOfTwoSpeechTensors = []
    listOfPrimaryTensors = []
    for i in range(arg_numfilestoload):
        
        # loading primary audio
        clip = random.sample(arg_list_of_files,1)[0]    # choosing a clip
        audioarrayPrimary = fCreateTensorFromWavPath(clip)     # fetching torch-tensor

        # loading secondary audio
        clip_secondary = random.sample(arg_list_of_files,1)[0]          # choosing a clip
        audioarray_secondary = fCreateTensorFromWavPath(clip_secondary) # fetching secondary tensor

        # normalizing to a particular dB
        audioarrayPrimary = fRaiseSignalToPower(audioarrayPrimary, primarypower)
        audioarray_secondary = fRaiseSignalToPower(audioarray_secondary, secondarypower)

        # adjusting lengths
        audioarray_secondary = fAdjustLengthOfSecondary(audioarrayPrimary, audioarray_secondary)

        # adding them together
        audioarray = audioarrayPrimary + audioarray_secondary
        listOfTwoSpeechTensors.append(audioarray)
        listOfPrimaryTensors.append(audioarrayPrimary)

    return listOfTwoSpeechTensors, listOfPrimaryTensors 
def fCreateListOfTwoSpeechTensorsWhisper(arg_list_of_files,
                                         arg_numfilestoload = 1,
                                         primarypower = -10,
                                         secondarypower = -15):
    listOfTwoSpeechTensors = []
    listOfPrimaryTensors = []
    gt_transcript_list = []

    for i in range(arg_numfilestoload):
        
        # loading primary audio
        clip = random.sample(arg_list_of_files,1)[0]    # choosing a clip
        audioarrayPrimary = fCreateTensorFromWavPath(clip)     # fetching torch-tensor

        # loading primary transcript
        clip_transcript_filename = fCreateNameOfCorrespondingTxtFile(clip)
        clip_gt_transcript = fReturnGroundTruthTranscript(clip_transcript_filename)
        gt_transcript_list.append(clip_gt_transcript)

        # loading secondary audio
        clip_secondary = random.sample(arg_list_of_files,1)[0]          # choosing a clip
        audioarray_secondary = fCreateTensorFromWavPath(clip_secondary) # fetching secondary tensor

        # normalizing to a particular dB
        audioarrayPrimary = fRaiseSignalToPower(audioarrayPrimary, primarypower)
        audioarray_secondary = fRaiseSignalToPower(audioarray_secondary, secondarypower)

        # adjusting lengths
        audioarray_secondary = fAdjustLengthOfSecondary(audioarrayPrimary, audioarray_secondary)

        # adding them together
        audioarray = audioarrayPrimary + audioarray_secondary

        # amplitude normalizing the two signals
        audioarray = audioarray / torch.max(torch.abs(audioarray))
        audioarrayPrimary = audioarrayPrimary/torch.max(torch.abs(audioarrayPrimary))

        # appending to list
        listOfTwoSpeechTensors.append(audioarray)
        listOfPrimaryTensors.append(audioarrayPrimary)

    return listOfTwoSpeechTensors, listOfPrimaryTensors, gt_transcript_list
def fFetchDevice():
    # detecting devices --------------------------------------------------------
    if torch.backends.mps.is_available(): device = torch.device('mps')
    elif torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')

    return device
def fFetchListOfTrainTestFiles(local_or_scc):
    if local_or_scc == 1:
        data_directory = "/projectnb/mlrobot/vrs/MSProject/Data/Timit/data/TRAIN"
        data_test_directory = "/projectnb/mlrobot/vrs/MSProject/Data/Timit/data/TEST"
        list_of_noise_files = ["/projectnb/mlrobot/vrs/MSProject/Data/Noise/restaurantnoise.wav"]
    else:
        data_directory = "/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Data/archive/data/TRAIN"
        data_test_directory = "/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Data/archive/data/TEST"
        list_of_noise_files = ["/Users/vrsreeganesh/Desktop/BUClasses/Thesis/Data/NoiseFiles/restaurantnoise.wav"];

    list_of_train_files = fFindWavFiles(data_directory)
    list_of_test_files = fFindWavFiles(data_test_directory)

    return list_of_train_files, list_of_test_files
def fCreateSignalFromMagnitudeAndPhase(modelOutput_Magnitude,
                                       modelOutput_Phase,
                                       istft):
    # appending zeros to both the magnitude and phase
    modelOutput_Magnitude   = torch.cat((modelOutput_Magnitude, 
                                         torch.zeros([modelOutput_Magnitude.shape[0],
                                                      modelOutput_Magnitude.shape[1],
                                                      1,
                                                      modelOutput_Magnitude.shape[3]]).to(modelOutput_Magnitude.device)), dim = 2)
    modelOutput_Phase       = torch.cat((modelOutput_Phase, 
                                         torch.zeros([modelOutput_Phase.shape[0],
                                                      modelOutput_Phase.shape[1],
                                                      1,
                                                      modelOutput_Phase.shape[3]]).to(modelOutput_Phase.device)), dim = 2)
    
    # taking inverse fourier transform and sending it back
    istft   = istft.to(modelOutput_Phase.device)
    bruh    = istft(modelOutput_Magnitude * torch.exp(1j * modelOutput_Phase))

    return bruh
def fSaveThisModel(model_object, modelname, path):

    # creating path for the model
    import os
    modelsave_fullpath = os.path.join(path, modelname)

    # saving the model
    torch.save(model_object.state_dict(), modelsave_fullpath)
def fFetchCurrentTimeString():
    import datetime
    currenttimestring = datetime.datetime.now(); 
    currenttimestring = currenttimestring.strftime("%m_%d_%H_%M_%S");
    return currenttimestring
def fMovingAverage(values, window_length):
    """
    Computes the moving average of a list of values using a specified window length.
    
    Parameters:
        values (list): List of numerical values.
        window_length (int): The length of the averaging window.
        
    Returns:
        list: A list containing the moving average values.
    """
    if window_length <= 0:
        raise ValueError("Window length must be a positive integer.")
    if not values:
        return []

    averages = []
    for i in range(len(values) - window_length + 1):
        window = values[i:i + window_length]
        averages.append(sum(window) / window_length)
    return averages
def fTakeTensor_Process_ReturnSignal(input_list_of_tensors,
                                     model_magnitude,
                                     model_phase,
                                     stft,
                                     istft):
    
    # going through each tensor and procesing it to save it
    processed_tensors = []
    for input_tensor in input_list_of_tensors: 
        
        # finding stft
        inputSTFT       = stft(input_tensor.to(device))     # calculating stft
        inputMagnitude  = torch.abs(inputSTFT)              # calculating stft-magnitude
        inputPhase      = torch.angle(inputSTFT)            # calculating stft-phase

        # splitting magnitude into multiple "pictures"
        num_pictures                = math.ceil(inputMagnitude.shape[1]/128)    # number of CNN-frames that can be made out of this
        num_zeros_to_add            = num_pictures * 128 - inputMagnitude.shape[1]  # number of zeros to add to create that many frames
        var00                       = torch.zeros([inputMagnitude.shape[0], 
                                                   num_zeros_to_add]).to(inputMagnitude.device)     # zero-tensor, that will later by concatenated
        inputMagnitudeZeroPadded    = torch.cat((inputMagnitude, var00), dim = 1)                   # the zero-padded magnitude-tensor
        inputPhaseZeroPadded        = torch.cat((inputPhase, var00), dim = 1)                       # zero padded phase-tensor

        # creating a tensor out of it
        pictureTensor = []
        pictureTensor_Phase = []
        for picture_index in range(num_pictures):
            # fetching start and end points
            startindex = picture_index*128                                                          # calculating starting point
            endindex = (picture_index+1)*128                                                        # calculating ending point
            
            # slicing and adding to list of magnitude-frames
            var00 = inputMagnitudeZeroPadded[0:512, startindex:endindex];                           # slicing the tensor at teh standing point and ending point
            var00 = var00.unsqueeze(0).unsqueeze(0)                                                 # adding dimensions in the front
            pictureTensor.append(var00)                                                             # appending to list containing tensors

            # slicing and adding to list of phase-frames
            var00 = inputPhaseZeroPadded[0:512, startindex:endindex]
            var00 = var00.unsqueeze(0).unsqueeze(0)
            pictureTensor_Phase.append(var00)

        # list to tensors
        pictureTensor       = torch.cat(pictureTensor, dim = 0).to(device)      # creating a tensor where batch size is the number of cnn-frames
        pictureTensor_Phase = torch.cat(pictureTensor_Phase, dim = 0).to(device)

        # passing through magnitude-processor
        # modeloutput_mask                = 2*model_magnitude(pictureTensor) - 1
        # output                          = modeloutput_mask + pictureTensor
        modeloutput_mask                = model_magnitude(pictureTensor)
        output                          = modeloutput_mask * pictureTensor

        # passing through phase-processor
        modeloutput_phase               = 2*torch.pi*model_phase(pictureTensor_Phase) - torch.pi
        modeloutput_phase               = modeloutput_phase + pictureTensor_Phase

        # assembling it back together 
        singleTensor = []
        singleTensor_Phase = []

        for i572 in range(output.shape[0]):
            singleTensor.append(output[i572,0,:,:])
            singleTensor_Phase.append(modeloutput_phase[i572, 0, :, :])

        singleTensor        = torch.cat(singleTensor,       dim = 1)            # list to tensor
        singleTensor_Phase  = torch.cat(singleTensor_Phase, dim = 1)            # list to tensor
        singleTensor        = singleTensor[:,       0:inputMagnitude.shape[1]]  # trimming to original length
        singleTensor_Phase  = singleTensor_Phase[:, 0:inputPhase.shape[1]]      # trimming to original length

        # adding a row of zeros since the index corresponding to phase = pi was removed
        singleTensor        = torch.cat((singleTensor,       torch.zeros([1, singleTensor.shape[1]]).to(device)),       dim = 0)
        singleTensor_Phase  = torch.cat((singleTensor_Phase, torch.zeros([1, singleTensor_Phase.shape[1]]).to(device)), dim = 0)

        # combining with the phase
        reconstructionTensor = singleTensor * torch.exp(1j * singleTensor_Phase)
        reconstructionSignal = istft(reconstructionTensor)

        # amplitude normalizing signal and saving
        reconstructionSignal = (reconstructionSignal/torch.max(torch.abs(reconstructionSignal))).to(torch.device('cpu'))
        processed_tensors.append(reconstructionSignal)

    # return the tensors
    return processed_tensors
def fCheckIfExistsIfNotCreate(directory_path):
    import os
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
def fMyCosineLoss(x1, x2):
            # finding ||x_1|| and ||x_2||
            l2x1 = torch.linalg.vector_norm(x = x1, ord = 2, dim = 2, keepdim = True)
            l2x2 = torch.linalg.vector_norm(x = x2, ord = 2, dim = 2, keepdim = True)

            # finding dot product
            dotproduct = torch.sum(input = x1 * x2, dim = 2, keepdim = True)

            # finding cosine between the two
            cosine_between_the_two = dotproduct / (l2x1 * l2x2)

            # finding the mean
            average_cosine = torch.mean(cosine_between_the_two)

            # calculating one minus average_cosine
            one_minus_cosine = 1 - average_cosine
            
            # returning the values
            return one_minus_cosine
def fCircularLoss(xhat, xinput, xideal):
    # calculate distance between estimate and ideal
    dist_xhat_xideal = torch.linalg.vector_norm(x       = xideal - xhat,
                                                ord     = 1,
                                                dim     = 2,
                                                keepdim = True)

    # calculate distance between input and estimate
    dist_xhat_xinput = torch.linalg.vector_norm(x       = xhat - xinput,
                                                ord     = 1,
                                                dim     = 2,
                                                keepdim = True)

    # calculate distance between input and ideal
    dist_xinput_xideal = torch.linalg.vector_norm(x = xinput - xideal,
                                                    ord = 1,
                                                    dim = 2,
                                                    keepdim = True)

    # calculating loss
    loss = dist_xhat_xideal - torch.min(dist_xhat_xinput, dist_xinput_xideal+1e-9)
    loss = torch.mean(loss)

    # return loss
    return loss

# Whisper Functions - Whisper Functions - Whisper Functions - Whisper Functions
# Whisper Functions - Whisper Functions - Whisper Functions - Whisper Functions
# Whisper Functions - Whisper Functions - Whisper Functions - Whisper Functions
# Whisper Functions - Whisper Functions - Whisper Functions - Whisper Functions
# Whisper Functions - Whisper Functions - Whisper Functions - Whisper Functions
def fCreateNameOfCorrespondingTxtFile(name_of_wav_file):
    '''
    function that creates corresponding txt-file
    '''
    # creating file-name
    name_of_txt_file = os.path.splitext(os.path.splitext(name_of_wav_file)[0])[0] + ".TXT"

    # returning corresponding file-name
    return name_of_txt_file
def fReturnGroundTruthTranscript(path_to_text_file):
    '''
    Function to just return the ground-truth transcription
    '''
    # reading text-file
    with open(path_to_text_file, 'r') as file:
        content = file.read()

    # removing the useless initial information 
    extracted_text = ' '.join(content.split()[2:]).strip()

    # # removing full-stop if the last character is a fullstop
    # if extracted_text[-1]== '.':
    #     extracted_text = extracted_text[:-1]
    
    # returning
    return extracted_text
def fRemoveEmptyStrings(list_of_gt_transcriptions, list_of_output_transcriptions):
    '''
    Steps
        go through each entry, if empty, we skip 
    '''

    # checking if not the same length
    same_length = 1
    if len(list_of_gt_transcriptions) != len(list_of_output_transcriptions):
        same_length = 0
        return 
    
    filtered_list_of_gt_transcripts = []
    filtered_list_of_output_transcripts = []
    # going through each entry and adding to new files
    for i in range(len(list_of_gt_transcriptions)):

        # fetching entries
        gt_transcript = list_of_gt_transcriptions[i]
        output_transcript = list_of_output_transcriptions[i]

        # checking if either of em are empty
        if len(gt_transcript) == 0 or len(output_transcript) == 0:
            continue
        else:
            filtered_list_of_gt_transcripts.append(gt_transcript)
            filtered_list_of_output_transcripts.append(output_transcript)
    
    # returning the filtered lists
    return filtered_list_of_gt_transcripts, filtered_list_of_output_transcripts
def fUpdateAverageWERList(transcription_train_loss_list,
                          average_wer_list,
                          histerisis_length = 128):
    # adding to the list that stores the average wer -------------------
    if len(transcription_train_loss_list) <= histerisis_length:
        average_wer_list.append(sum(transcription_train_loss_list)/len(transcription_train_loss_list))
    else:
        temp_list = transcription_train_loss_list[-histerisis_length:]
        average_wer_list.append(sum(temp_list)/len(temp_list))

    # returning
    return average_wer_list
def fProcessAndCalculateTranscriptionError(list_of_train_files, 
                                           num_samples,
                                           primarypower, 
                                           secondarypower,
                                           model_magnitude, 
                                           model_phase,
                                           stft, istft):
    
    # calculating transcription error --------------------------------------
    with torch.no_grad():

        # creating input-data ----------------------------------------------
        audio_files = random.sample(list_of_train_files, num_samples) 
        
        # creating a list of noisy-tensors -----------------------------------------------------
        input_list_of_tensors, input_list_of_primary_tensors, gt_files = \
            fCreateListOfTwoSpeechTensorsWhisper(arg_list_of_files = audio_files,
                                                 arg_numfilestoload = num_samples,
                                                 primarypower = primarypower,
                                                 secondarypower = secondarypower)
        
        # running denoiser ---------------------------------------------------------------------
        denoised_samples = fTakeTensor_Process_ReturnSignal(input_list_of_tensors = input_list_of_tensors,
                                                            model_magnitude = model_magnitude,
                                                            model_phase = model_phase,
                                                            stft = stft,
                                                            istft = istft)

        # calculating wer ------------------------------------------------
        list_of_transcriptions = []
        for sample_index in range(len(denoised_samples)):

            # taking a sample out of the list --------------------------------------------------
            denoised_sample_current = denoised_samples[sample_index].type(torch.float32)

            # finding transcript -------------------------------------------
            transcript_of_current_tensor = model_whisper.transcribe(denoised_sample_current)["text"]

            # printing current-transcript ----------------------------------
            print("sample_index = ", sample_index, "/", len(denoised_samples))
            print("output_transcript:", transcript_of_current_tensor)
            print("gt_transcript    :", gt_files[sample_index])
            if len(transcript_of_current_tensor)!=0 and len(gt_files[sample_index])!=0:
                print("wer: ", jiwer.wer(transcript_of_current_tensor, gt_files[sample_index]))
            print("i = ",i,"-"*50)

            # adding whisper-transcript to list of output-transcripts-------
            list_of_transcriptions.append(transcript_of_current_tensor)

        # removing empty strings -----------------------------------------------------------------------------------
        filtered_gt_files, filtered_transcriptions =  fRemoveEmptyStrings(gt_files, list_of_transcriptions)

        # calculating word-error-rate --------------------------------------
        average_wer = jiwer.wer(filtered_gt_files, filtered_transcriptions)
        print("average_wer = ", average_wer)
        
    return average_wer

    #     transcription_train_loss_list.append(average_wer)

    #     # adding to the list that stores the average wer ---------------------------------------
    #     average_wer_list = fUpdateAverageWERList(transcription_train_loss_list, average_wer_list)

    # return transcription

# Dataset setup 
list_of_train_files, list_of_test_files = fFetchListOfTrainTestFiles(local_or_scc = 1)

# Global params
lr_denoiser                             = 1e-3
lr_phase                                = 1e-2 # previously, 1e-4
lr_discriminator                        = 1e-3 
batch_size                              = 32
snr_level                               = 3
primarypower                            = -10
secondarypower                          = primarypower - snr_level
num_samples                             = 10
training_switch_period                  = 400

# fetching device
device = fFetchDevice()

# Whisper 
'''
------------------------------------------------------------
Aim: The following section loads OpenAI's Whisper model. 
Note: 
    * The "download_root" argument allows us to instruct the function where to store the model (weights). If not provided, it will store in a default location it decides by itself. If the weights do not exist in the location specified through "download_root", it downloads it again.  
    * Whisper contains multiple versions corresponding to different speed-accuracy trade offs. "large" is its best model but takes more time. 
------------------------------------------------------------
'''
import whisper
# model_whisper = whisper.load_model("tiny",  
#                                    device = device,
#                                    download_root = "<insert path to location>")
# model_whisper = whisper.load_model("base",  
#                                    device = device,
#                                    download_root = "<insert path to location>")
# model_whisper = whisper.load_model("small",  
#                                    device = device,
#                                    download_root = "<insert path to location>")
# model_whisper = whisper.load_model("medium", 
#                                    device = device,
#                                    download_root = "<insert path to location>")
# model_whisper = whisper.load_model("turbo", 
#                                    device = device,
#                                    download_root = "<insert path to location>")
model_whisper = whisper.load_model("large", 
                                   device = device,
                                   download_root = "<insert path to location>")

# Spectrogram Function 
'''
We use the STFT and ISTFT functions provided by the torchaudio package
'''
N_FFT = 1024        # length of fft
N_HOP = 128         # hop length for STFT
stft    = torchaudio.transforms.Spectrogram(n_fft       = N_FFT, 
                                            hop_length  = N_HOP, 
                                            power       = None, 
                                            normalized  = True).to(device)
istft   = torchaudio.transforms.InverseSpectrogram(n_fft        = N_FFT, 
                                                   hop_length   = N_HOP, 
                                                   normalized   = True).to(device)

# Instantiating the magnitude and phase processing model
model_denoiser      = UNet_Architecture().to(device)                                    # instantiating magnitude processor
optimizer_denoiser  = torch.optim.Adam(model_denoiser.parameters(), lr = lr_denoiser)   # setting up optimizer for magnitude
model_phase         = UNet_Architecture().to(device)                                    # instantiating phase processor
optimizer_phase     = torch.optim.Adam(model_phase.parameters(),    lr = lr_phase)      # setting up optimizer for phase
optimizer_MP        = torch.optim.Adam(list(model_denoiser.parameters())+ list(model_phase.parameters()), 
                                       lr = lr_denoiser)    # optimizer for both (used in adversarial training)

# Instantiating the discriminators and their optimizers
model_discriminator                     = Discriminator().to(device)
model_discriminator_2                   = Discriminator().to(device)
model_discriminator_3                   = Discriminator().to(device)
optimizer_discriminator   = torch.optim.Adam(model_discriminator.parameters(), lr = lr_discriminator)
optimizer_discriminator_2 = torch.optim.Adam(model_discriminator_2.parameters(), lr = lr_discriminator)
optimizer_discriminator_3 = torch.optim.Adam(model_discriminator_3.parameters(), lr = lr_discriminator)

# Loss functions
criterion            = torch.nn.L1Loss()

# Directories = Directories = Directories = Directories = Directories = Directories
# Directories = Directories = Directories = Directories = Directories = Directories
# Directories = Directories = Directories = Directories = Directories = Directories
# Directories = Directories = Directories = Directories = Directories = Directories
# Directories = Directories = Directories = Directories = Directories = Directories
'''
------------------------------------------------------------
Aim: Creating directories to store audio-samples, data, plots, etc
------------------------------------------------------------
'''
# getting current-time as string
currenttimestring = fFetchCurrentTimeString()               # getting current-time as string
currentscriptname = sys.argv[0][:-3]                        # getting name of current script

# directory to save models
modelDirectoryPath = os.path.join("/projectnb/mlrobot/vrs/MSProject/aTENNuate/mymodels",            currentscriptname)
fCheckIfExistsIfNotCreate(modelDirectoryPath)

# directory to save plots
directory_to_save_figures = os.path.join("/projectnb/mlrobot/vrs/MSProject/aTENNuate/plots",        currentscriptname)
fCheckIfExistsIfNotCreate(directory_to_save_figures)

# directory to save audio-samples
directory_to_save_audio = os.path.join("/projectnb/mlrobot/vrs/MSProject/aTENNuate/AudioSamples",   currentscriptname)
fCheckIfExistsIfNotCreate(directory_to_save_audio)

# directory to save data (training loss, transcription loss, etc)
directory_to_save_data = os.path.join("/projectnb/mlrobot/vrs/MSProject/aTENNuate/Data",            currentscriptname)
fCheckIfExistsIfNotCreate(directory_to_save_data)

# record lists 
'''
Following are the lists that we'll be using when training the model. 
'''
training_loss                       = []    # list containing STFT-magnitude error
training_loss_averaged              = []    # list containing previous n-samples averaged STFT-magnitude error
training_loss_phase                 = []    # list containing STFT-loss error
training_loss_phase_averaged        = []    # list containing previous n-samples averaged STFT-phase error

baseline_magnitude_loss_list        = []    
baseline_phase_loss_list            = []

adversarial_loss_magnitude_list     = []
discriminator_magnitude_loss_list   = []
discriminator_phase_loss_list       = []
transcription_train_loss_list       = []
average_wer_list                    = []

discriminator_loss_list             = []
discriminator_loss_list_2           = []
discriminator_loss_list_3           = []

for i in range(math.floor(1e9)):

    # # calculating the transcription error 
    # '''
    # ============================================================================
    # Aim: Calculate transcription error
    # ============================================================================
    # '''
    # if i%4000==0 and i!=0:
    # # if i>1000:
    #     average_wer = fProcessAndCalculateTranscriptionError(list_of_train_files = list_of_test_files, 
    #                                                          num_samples         = len(list_of_test_files),
    #                                                          primarypower        = primarypower, 
    #                                                          secondarypower      = secondarypower,
    #                                                          model_magnitude     = model_denoiser, 
    #                                                          model_phase         = model_phase,
    #                                                          stft                = stft,
    #                                                          istft               = istft)
    #     transcription_train_loss_list.append(average_wer)
    #     average_wer_list = fUpdateAverageWERList(transcription_train_loss_list, average_wer_list)
    #     average_wer_tensor = torch.tensor(average_wer)
    # else:
    #     average_wer_tensor = torch.tensor(1)


    # Training =====================================================================================
    # Training =====================================================================================
    # Training =====================================================================================
    if ((i%training_switch_period) < training_switch_period/2) or i<2e3:
        '''
        ========================================================================
        Aim: Regular Training
        Note:
            * magnitude-network is trained to reduce loss from ideal
            * phase-network is trained to reduce loss from ideal
            * discriminator is trained to classify ideal as 1, processed as zero.
            * There are parts of code where we change the mode (train(), eval()) of the models. These are very important and changing them without understanding how they work will produce a model that won't do what we train it to do. 
        ========================================================================j
        '''
        # printing status
        print("i=",i,"<regular training>")

        # putting models in their corresponding mode 
        '''
        * This section is very important. 
        * Changing this WILL change the training paradigm and will result in not arriving at the trained model. '''
        model_denoiser                  = model_denoiser.train()
        model_phase                     = model_phase.train()
        model_discriminator             = model_discriminator.train()
        model_discriminator_2           = model_discriminator_2.train()

        # fetching a batch
        '''
        * This function takes the list of paths to training wav-files
        * Creates overlapped speech clips (called "noisy" clips). 
        * Returns the STFT of noisy-clips and the STFT of the primary speech-clip '''
        inputTensor, idealOutput = fCreateBatchTwoSpeech(arg_list_of_files = list_of_train_files,
                                                         arg_number_of_files_to_take_from = 10,
                                                         arg_batch_size = batch_size,
                                                         arg_N_FFT = N_FFT,
                                                         arg_N_HOP = N_HOP,
                                                         primarypower = primarypower,
                                                         secondarypower = secondarypower)

        # data-setup
        '''
        * The functions returned by the fCreateBatchTwoSpeech() function is complex. 
        * We take the magnitude and phase below. '''
        inputMagnitude                  = torch.abs(inputTensor).to(device)
        inputPhase                      = torch.angle(inputTensor).to(device)
        idealOutputMagnitude            = torch.abs(idealOutput).to(device)
        idealOutputPhase                = torch.angle(idealOutput).to(device)

        # processing both magnitude and phase
        '''
        * We run the STFT-magnitude through the magnitude-model and produce a mask. 
        * The mask is multiplied with STFT-magnitude of noisy-audio'''
        modeloutput_mask                = model_denoiser(inputMagnitude)
        modelOutput_Magnitude           = inputMagnitude*modeloutput_mask
        modeloutput_phase_mask          = 2*torch.pi*model_phase(inputPhase) - torch.pi
        modelOutput_Phase               = inputPhase + modeloutput_phase_mask
        
        # Calculating loss
        '''
        * The loss function is a composite loss function. Please refer to the document to better understand how it works.
        * We calculate the losses for both the output of our pipeline and the input. 
        * Both are stored for comparative purposes. '''
        loss                    = fCircularLoss(modelOutput_Magnitude,  inputMagnitude, idealOutputMagnitude) * \
                                  fMyCosineLoss(modelOutput_Magnitude,  idealOutputMagnitude)
        baseline_magnitude_loss = fCircularLoss(inputMagnitude,         inputMagnitude, idealOutputMagnitude) * \
                                  fMyCosineLoss(inputMagnitude,         idealOutputMagnitude)
        loss_phase              = fCircularLoss(modelOutput_Phase,      inputPhase,     idealOutputPhase)
        baseline_phase_loss     = fCircularLoss(inputPhase,             inputPhase,     idealOutputPhase)

        training_loss.append(loss.item())
        baseline_magnitude_loss_list.append(baseline_magnitude_loss.item())
        training_loss_averaged = fUpdateAverageWERList(training_loss, training_loss_averaged, histerisis_length = 128)

        training_loss_phase.append(loss_phase.item())
        baseline_phase_loss_list.append(baseline_phase_loss.item())
        training_loss_phase_averaged = fUpdateAverageWERList(training_loss_phase, training_loss_phase_averaged, histerisis_length = 128)
        
        # backproping: magnitude and phase
        '''
        * Backpropagating to improve the enhancement capabilities of the models'''
        optimizer_denoiser.zero_grad(); loss.backward();        optimizer_denoiser.step()
        optimizer_phase.zero_grad();    loss_phase.backward();  optimizer_phase.step()

        
        # Discriminator Section = Discriminator Section = Discriminator Section 
        # Discriminator Section = Discriminator Section = Discriminator Section 
        # Discriminator Section = Discriminator Section = Discriminator Section 
        # Discriminator Section = Discriminator Section = Discriminator Section 
        # Discriminator Section = Discriminator Section = Discriminator Section 

        # reproducing signals from STFT
        '''
        * Reconstruct signal from the processed STFT '''
        inputSignal       = fCreateSignalFromMagnitudeAndPhase(idealOutputMagnitude,           idealOutputPhase,           istft)
        processedSignal   = fCreateSignalFromMagnitudeAndPhase(modelOutput_Magnitude.detach(), modelOutput_Phase.detach(), istft)
        
        '''
        * Downsampling (not decimation) the input-signal and processed signal before feeding to discriminators
        * This downsampling is achieved through the convolution operation with stride = 2
        * This processing is done for the second discriminator.'''
        filter_uniform    = torch.ones([1,1,4]).to(device)
        inputSignal_2     = torch.nn.functional.conv1d(inputSignal, filter_uniform, stride = 2)
        processedSignal_2 = torch.nn.functional.conv1d(processedSignal, filter_uniform, stride = 2)

        # running discriminator
        '''
        * We're using two discriminators for this task.
        * The first discriminator runs on the signal at the currrent-scale
        * The second discriminator runs on the downsampled signal'''
        discriminatorOutput_ideal       = model_discriminator(inputSignal)
        discriminatorOutput_processed   = model_discriminator(processedSignal)
        discriminatorOutput_ideal_2     = model_discriminator_2(inputSignal_2)
        discriminatorOutput_processed_2 = model_discriminator_2(processedSignal_2)

        # producing ideal-data
        '''
        * Creating tensor holding the ideal outputs for each index. 
        * Please refer to the document to better understand what is going on. '''
        idealOutput_Discriminator       = torch.cat((torch.zeros_like(discriminatorOutput_ideal),
                                                     torch.ones_like(discriminatorOutput_processed)),dim=0)
        modelOutput_Discriminator       = torch.cat((discriminatorOutput_ideal,
                                                     discriminatorOutput_processed), dim = 0)
        idealOutput_Discriminator_2     = torch.cat((torch.zeros_like(discriminatorOutput_ideal_2),
                                                     torch.ones_like(discriminatorOutput_processed_2)),dim=0)
        modelOutput_Discriminator_2     = torch.cat((discriminatorOutput_ideal_2,
                                                     discriminatorOutput_processed_2), dim = 0)
        
        # calculating loss according to the paper 
        '''
        * Now that we have the outputs of the discriminator and the output it should've produced, we calculate the error'''
        modelOutput_Discriminator       = torch.nn.functional.sigmoid(modelOutput_Discriminator)
        discriminator_loss              = torch.nn.functional.binary_cross_entropy(modelOutput_Discriminator, idealOutput_Discriminator);
        discriminator_loss_list.append(discriminator_loss.item())

        modelOutput_Discriminator_2     = torch.nn.functional.sigmoid(modelOutput_Discriminator_2)
        discriminator_loss_2            = torch.nn.functional.binary_cross_entropy(modelOutput_Discriminator_2, idealOutput_Discriminator_2);
        discriminator_loss_list_2.append(discriminator_loss_2.item())

        # backpropagating
        '''
        * Here, we train the discriminator to get better at discriminating what a true-signal looks like vs what a processed signal looks like'''
        optimizer_discriminator.zero_grad();   discriminator_loss.backward();   optimizer_discriminator.step()
        optimizer_discriminator_2.zero_grad(); discriminator_loss_2.backward(); optimizer_discriminator_2.step()

    else:
        '''
        ========================================================================
        adversarial training
        * This should give you a better idea about GANs: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        * 
        ========================================================================
        '''
        print("i=",i,"<adversarial training>")
        # putting models in corresponding mode
        '''
        * Again, please do not mess with the training modes of these models. 
        * Doing so will result in errors and not arriving at the trained model'''
        model_denoiser                  = model_denoiser.train()
        model_phase                     = model_phase.train()
        model_discriminator             = model_discriminator.eval()
        model_discriminator_2           = model_discriminator_2.eval()

        # fetching a batch
        '''
        * This function takes the list of paths to training wav-files
        * Creates overlapped speech clips (called "noisy" clips). 
        * Returns the STFT of noisy-clips and the STFT of the primary speech-clip '''
        inputTensor, idealOutput = fCreateBatchTwoSpeech(arg_list_of_files = list_of_train_files,
                                                         arg_number_of_files_to_take_from = 10,
                                                         arg_batch_size = batch_size,
                                                         arg_N_FFT = N_FFT,
                                                         arg_N_HOP = N_HOP,
                                                         primarypower = primarypower,
                                                         secondarypower = secondarypower)

        # data-setup
        '''
        * The functions returned by the fCreateBatchTwoSpeech() function is complex. 
        * We take the magnitude and phase below. '''
        inputPhase                      = torch.angle(inputTensor).to(device)
        inputMagnitude                  = torch.abs(inputTensor).to(device)
        idealOutputMagnitude            = torch.abs(idealOutput).to(device)
        idealOutputPhase                = torch.angle(idealOutput).to(device)

        # processing both magnitude and phase
        '''
        * We run the STFT-magnitude through the magnitude-model and produce a mask. 
        * The mask is multiplied with STFT-magnitude of noisy-audio'''
        modeloutput_mask                = 2*model_denoiser(inputMagnitude) - 1
        modelOutput_Magnitude           = inputMagnitude + modeloutput_mask
        modeloutput_phase_mask          = 2*torch.pi*model_phase(inputPhase) - torch.pi
        modelOutput_Phase               = inputPhase + modeloutput_phase_mask

        # Discriminator - Discriminator - Discriminator - Discriminator - Discriminator - Discriminator
        # Discriminator - Discriminator - Discriminator - Discriminator - Discriminator - Discriminator
        # Discriminator - Discriminator - Discriminator - Discriminator - Discriminator - Discriminator

        # reproducing signals from STFT
        '''
        * Reconstruct signal from the processed STFT '''
        inputSignal         = fCreateSignalFromMagnitudeAndPhase(idealOutputMagnitude, idealOutputPhase, istft)
        processedSignal     = fCreateSignalFromMagnitudeAndPhase(modelOutput_Magnitude.detach(), modelOutput_Phase.detach(), istft)

        '''
        * Downsampling (not decimation) the input-signal and processed signal before feeding to discriminators
        * This downsampling is achieved through the convolution operation with stride = 2
        * This processing is done for the second discriminator.'''
        filter_uniform      = torch.ones([1,1,4]).to(device)
        inputSignal_2       = torch.nn.functional.conv1d(inputSignal, filter_uniform, stride = 2)
        processedSignal_2   = torch.nn.functional.conv1d(processedSignal, filter_uniform, stride = 2)

        # running discriminator
        '''
        * We're using two discriminators for this task.
        * The first discriminator runs on the signal at the currrent-scale
        * The second discriminator runs on the downsampled signal'''
        discriminatorOutput_ideal       = model_discriminator(inputSignal)
        discriminatorOutput_processed   = model_discriminator(processedSignal)
        discriminatorOutput_ideal_2     = model_discriminator_2(inputSignal_2)
        discriminatorOutput_processed_2 = model_discriminator_2(processedSignal_2)

        # producing ideal-data
        '''
        * Creating tensor holding the ideal outputs for each index. 
        * Please refer to the document to better understand what is going on. '''
        idealOutput_Discriminator       =torch.cat((torch.zeros_like(discriminatorOutput_ideal),
                                                     torch.zeros_like(discriminatorOutput_processed)),dim=0)
        modelOutput_Discriminator       = torch.cat((discriminatorOutput_ideal,
                                                     discriminatorOutput_processed), dim = 0)
        idealOutput_Discriminator_2     = torch.cat((torch.zeros_like(discriminatorOutput_ideal_2),
                                                     torch.zeros_like(discriminatorOutput_processed_2)),dim=0)
        modelOutput_Discriminator_2     = torch.cat((discriminatorOutput_ideal_2,
                                                     discriminatorOutput_processed_2), dim = 0)
        
        # loss-calculation
        '''
        * Now that we have the outputs of the discriminator and the output it should've produced, we calculate the error'''
        modelOutput_Discriminator   = torch.nn.functional.sigmoid(modelOutput_Discriminator)
        discriminator_loss          = torch.nn.functional.binary_cross_entropy(modelOutput_Discriminator, idealOutput_Discriminator);
        discriminator_loss_list.append(discriminator_loss.item())

        modelOutput_Discriminator_2 = torch.nn.functional.sigmoid(modelOutput_Discriminator_2)
        discriminator_loss_2 = torch.nn.functional.binary_cross_entropy(modelOutput_Discriminator_2, idealOutput_Discriminator_2);
        discriminator_loss_list_2.append(discriminator_loss_2.item())

        loss_together = discriminator_loss + discriminator_loss_2 + 0.00625 * torch.nn.functional.mse_loss(inputSignal, processedSignal)

        # backproping: magnitude and phase
        '''
        * Here, we train the magnitude and phase model jointly'''
        optimizer_MP.zero_grad(); loss_together.backward(); optimizer_MP.step()
        
    # plotting =================================================================
    # plotting =================================================================
    # plotting =================================================================
    # plotting =================================================================
    if i%200 ==0:
        '''
        Here we plot:
            * Training losses of STFT-processing models
            * Transcription losses 
            * The discriminator losses '''
        
        # plotting training losses
        plt.figure(1); plt.clf(); 
        plt.subplot(1,2,1); 
        plt.plot(training_loss,          linewidth=0.25,    label = 'magnitude-loss'); 
        plt.plot(training_loss_averaged, label = "magnitude-loss:averaged")
        plt.plot(baseline_magnitude_loss_list, linewidth = 0.25, label = "baseline")
        plt.legend(); plt.title("STFT Losses")
        
        plt.subplot(1,2,2); 
        plt.plot(training_loss_phase, linewidth=0.25, label = 'phase-loss'); 
        plt.plot(training_loss_phase_averaged, label = 'phase-loss:averaged'); 
        plt.plot(baseline_phase_loss_list     , linewidth = 0.25, label = "baseline")
        plt.legend(); plt.title("STFT Losses")
        plt.savefig(os.path.join(directory_to_save_figures, currentscriptname + "_StftLosses"))
        plt.draw(); plt.pause(1e-4)

        # plotting transcription loss
        plt.figure(2); plt.clf()
        plt.plot(transcription_train_loss_list, linewidth = 0.25, label = 'transcription-loss: raw')
        plt.plot(average_wer_list, label = 'transcription-loss: averaged')
        plt.legend(); plt.title("transcription loss")
        plt.savefig(os.path.join(directory_to_save_figures, currentscriptname + "_TranscriptionLosses"))
        plt.draw(); plt.pause(1e-4)

        # plotting discriminator loss
        plt.figure(4); plt.clf()
        plt.plot(discriminator_loss_list, linewidth=0.25, label = 'raw-discriminator')
        plt.plot(discriminator_loss_list_2, linewidth=0.25, label = 'raw-x2-discriminator')
        plt.title("discriminator:raw")
        plt.savefig(os.path.join(directory_to_save_figures, currentscriptname + "_DiscriminatorLosses"))
        plt.legend(); plt.draw(); plt.pause(1e-4)

        # # plotting discriminator loss
        # plt.figure(5)
        # plt.plot(discriminator_phase_loss_list)
        # plt.title("discriminator:phase")
        # plt.draw(); plt.pause(1e-4)

        # plt.figure(2)
        # var00 = modeloutput_mask[0,0,:,:].detach().cpu().numpy()
        # var01 = np.log(1 + np.abs(inputMagnitude[0,0,:,:].detach().cpu().numpy()))
        # var02 = np.log(1 + np.abs(output[0,0,:,:].detach().cpu().numpy()))
        # fig, axes = plt.subplots(1, 3, num = 2)
        # axes[0].imshow(var00, cmap='gray')
        # axes[1].imshow(var01, cmap='gray')
        # axes[2].imshow(var02, cmap='gray')
        # # if i==0: plt.colorbar()
        # plt.draw(); 
        # plt.pause(1e-5)

    # saving model
    if i%1000 == 0: 
        '''
        * Saving both the magnitude-processing model and phase-processing model. 
        '''
        torch.save(model_denoiser.state_dict(), os.path.join(modelDirectoryPath, "modelMagnitude_" + currenttimestring))
        torch.save(model_phase.state_dict(),    os.path.join(modelDirectoryPath, "modelPhase_" + currenttimestring))

    # saving lists
    if i%1000 == 0:
        '''
        Saving some lists as pickle files
        * transcription loss list
        * averaged transcription loss list
        * magnitude training loss list
        * averaged magnitude training loss list
        * phase training loss list
        * averaged phaes training loss list '''
        # saving transcript loss-list
        transcription_train_loss_list_filepath = os.path.join(directory_to_save_data, "transcription_train_loss_list.pkl")
        with open(transcription_train_loss_list_filepath, 'wb') as file: pickle.dump(transcription_train_loss_list, file)

        # saving averaged transcription loss
        average_wer_list_filepath = os.path.join(directory_to_save_data, "average_wer_list.pkl")
        with open(average_wer_list_filepath, 'wb') as file: pickle.dump(average_wer_list, file)
        
        # saving magnitude losses
        training_loss_magnitude_filepath = os.path.join(directory_to_save_data, "training_loss_magnitude.pkl")
        with open(training_loss_magnitude_filepath, 'wb') as file: pickle.dump(training_loss, file)
        training_loss_magnitude_averaged_filepath = os.path.join(directory_to_save_data, "training_loss_magnitude_averaged.pkl")
        with open(training_loss_magnitude_averaged_filepath, 'wb') as file: pickle.dump(training_loss_averaged, file)

        # saving phase losses
        training_loss_phase_filepath = os.path.join(directory_to_save_data, "training_loss_phase.pkl")
        with open(training_loss_phase_filepath, 'wb') as file: pickle.dump(training_loss_phase, file)
        training_loss_phase_averaged_filepath = os.path.join(directory_to_save_data, "training_loss_phase_averaged.pkl")
        with open(training_loss_phase_averaged_filepath, 'wb') as file: pickle.dump(training_loss_phase_averaged, file)

        # saving discriminator losses
        discriminator_loss_list_filepath = os.path.join(directory_to_save_data, "discriminator_loss_list.pkl")
        with open(discriminator_loss_list_filepath, 'wb') as file: pickle.dump(discriminator_loss_list, file)
        discriminator_loss_list_2_filepath = os.path.join(directory_to_save_data, "discriminator_loss_list_2.pkl")
        with open(discriminator_loss_list_2_filepath, 'wb') as file: pickle.dump(discriminator_loss_list_2, file)

    # validation: original
    if i%500 == 0:
        '''
        * Here, we produce examples so that we can perceptually evaluate how the training is going. 
        '''
        # load a mix
        listOfSpeechNoiseTensors, listOfPrimaryTensors = \
            fCreateListOfTwoSpeechTensors(arg_list_of_files = list_of_train_files,
                                          arg_numfilestoload = 1,
                                          primarypower = primarypower,
                                          secondarypower = secondarypower)

        # finding stft
        inputSTFT                       = stft(listOfSpeechNoiseTensors[0].to(device))
        inputMagnitude                  = torch.abs(inputSTFT)
        inputPhase                      = torch.angle(inputSTFT)

        # splitting magnitude into multiple "pictures"
        num_pictures                = math.ceil(inputMagnitude.shape[1]/128)
        num_zeros_to_add            = num_pictures * 128 - inputMagnitude.shape[1]
        var00                       = torch.zeros([inputMagnitude.shape[0], num_zeros_to_add]).to(inputMagnitude.device)
        
        inputMagnitudeZeroPadded    = torch.cat((inputMagnitude, var00), dim = 1)
        inputPhaseZeroPadded        = torch.cat((inputPhase, var00),     dim = 1)

        # creating a tensor out of it
        pictureTensor = []
        pictureTensor_phase = []

        for picture_index in range(num_pictures):
            # getting start-index and end-index
            startindex                  = picture_index*128
            endindex                    = (picture_index+1)*128

            # splicing magnitude and phase tensors
            var00                       = inputMagnitudeZeroPadded[0:512, startindex:endindex];
            var00                       = var00.unsqueeze(0).unsqueeze(0)
            var01                       = inputPhaseZeroPadded[0:512, startindex:endindex]
            var01                       = var01.unsqueeze(0).unsqueeze(0)

            # appending to list (that will later create multi-dimensional tensor)
            pictureTensor.append(      var00)
            pictureTensor_phase.append(var01)

        # creating a tensor out of the lists
        pictureTensor       = torch.cat(pictureTensor,          dim = 0).to(device)
        pictureTensor_phase = torch.cat(pictureTensor_phase,    dim = 0).to(device)

        # passing through the model
        modeloutput_mask                = model_denoiser(pictureTensor)
        output                          = modeloutput_mask * pictureTensor
        modeloutput_mask_phase          = 2 * torch.pi * model_phase(pictureTensor_phase) - torch.pi
        output_phase                    = modeloutput_mask_phase + pictureTensor_phase

        # assembling it back together 
        singleTensor       = []
        singleTensor_phase = []
        
        for i572 in range(output.shape[0]):
            singleTensor.append(output[i572,0,:,:])
            singleTensor_phase.append(output_phase[i572,0,:,:])
        
        singleTensor       = torch.cat(singleTensor, dim = 1)
        singleTensor_phase = torch.cat(singleTensor_phase, dim = 1)
        singleTensor       = singleTensor[:,        0:inputMagnitude.shape[1]]
        singleTensor_phase = singleTensor_phase[:,  0:inputPhase.shape[1]]

        # adding a row of zeros 
        singleTensor       = torch.cat((singleTensor, 
                                        torch.zeros([1, singleTensor.shape[1]]).to(device)), dim = 0)
        singleTensor_phase = torch.cat((singleTensor_phase,
                                        torch.zeros([1, singleTensor_phase.shape[1]]).to(device)), dim = 0)

        # version 1: Combining with input-phase
        reconstructionTensor_inputPhase = singleTensor * torch.exp(1j * inputPhase)
        reconstructionSignal_inputPhase = istft(reconstructionTensor_inputPhase)

        # version 2: Combining with processed-phase
        reconstructionTensor = singleTensor * torch.exp(1j * singleTensor_phase)
        reconstructionSignal = istft(reconstructionTensor)

        # amplitude normalizing signal 
        reconstructionSignal            = (reconstructionSignal/torch.max(torch.abs(reconstructionSignal))).to(torch.device('cpu'))
        reconstructionSignal_inputPhase = (reconstructionSignal_inputPhase/torch.max(torch.abs(reconstructionSignal_inputPhase))).to(torch.device('cpu'))
        listOfSpeechNoiseTensors[0]     = listOfSpeechNoiseTensors[0]/torch.max(torch.abs(listOfSpeechNoiseTensors[0])).cpu()
        listOfPrimaryTensors[0]         = listOfPrimaryTensors[0]/torch.max(torch.abs(listOfPrimaryTensors[0])).cpu()
        differenceSignal                = listOfSpeechNoiseTensors[0][:len(reconstructionSignal)] - reconstructionSignal
        differenceSignal                = differenceSignal/torch.max(torch.abs(differenceSignal))

        # cause I wanna show at least four clips from each model to professor
        clipindex = i/1000
        clipindex = clipindex%4
        clipindex = int(clipindex)

        # saving audio
        torchaudio.save(os.path.join(directory_to_save_audio, currentscriptname + "waveunet_processed_" + str(clipindex) + ".wav"),
                        reconstructionSignal.unsqueeze(0).detach().cpu(),
                        sample_rate = 16000)
        torchaudio.save(os.path.join(directory_to_save_audio, currentscriptname + "waveunet_processed_inputphase" + str(clipindex) + ".wav"),
                        reconstructionSignal_inputPhase.unsqueeze(0).detach().cpu(),
                        sample_rate = 16000)
        torchaudio.save(os.path.join(directory_to_save_audio, currentscriptname + "waveunet_input_" + str(clipindex) + ".wav"),
                        listOfSpeechNoiseTensors[0].unsqueeze(0).cpu().detach(),
                        sample_rate = 16000)
        torchaudio.save(os.path.join(directory_to_save_audio, currentscriptname + "waveunet_primary_" + str(clipindex) + ".wav"),
                        listOfPrimaryTensors[0].unsqueeze(0).cpu().detach(),
                        sample_rate = 16000)
        torchaudio.save(os.path.join(directory_to_save_audio, currentscriptname + "waveunet_diff_" + str(clipindex) + ".wav"),
                        differenceSignal.unsqueeze(0).cpu().detach(),
                        sample_rate = 16000)
