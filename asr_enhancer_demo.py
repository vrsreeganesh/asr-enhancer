'''
================================================================================
Aim: 
    To demonstrate the model
================================================================================
'''

# packages and libraries
import torch; import os; import random; import librosa; import torchaudio;
import pdb; import sys; import math; import numpy as np; import jiwer; 
import matplotlib.pyplot as plt; import torch.nn as nn; import pickle;

# Architecure = Architecure = Architecure = Architecure = Architecure
# Architecure = Architecure = Architecure = Architecure = Architecure
# Architecure = Architecure = Architecure = Architecure = Architecure
# Architecure = Architecure = Architecure = Architecure = Architecure
# Architecure = Architecure = Architecure = Architecure = Architecure
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
class WaveUNeT_Architecture(torch.nn.Module):
    def __init__(self):
        super(WaveUNeT_Architecture, self).__init__()
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
        singleTensor            = []
        singleTensor_Phase      = []
        singleTensor_inputPhase = []

        for i572 in range(output.shape[0]):
            singleTensor.append(output[i572,0,:,:])
            singleTensor_Phase.append(modeloutput_phase[i572, 0, :, :])
            singleTensor_inputPhase.append(pictureTensor_Phase[i572, 0, :, :])

        singleTensor            = torch.cat(singleTensor,               dim = 1)            # list to tensor
        singleTensor_Phase      = torch.cat(singleTensor_Phase,         dim = 1)            # list to tensor
        singleTensor_inputPhase = torch.cat(singleTensor_inputPhase,    dim = 1)

        singleTensor            = singleTensor[:,               0:inputMagnitude.shape[1]]  # trimming to original length
        singleTensor_Phase      = singleTensor_Phase[:,         0:inputPhase.shape[1]]      # trimming to original length
        singleTensor_inputPhase = singleTensor_inputPhase[:,    0:inputPhase.shape[1]]

        # adding a row of zeros since the index corresponding to phase = pi was removed
        singleTensor            = torch.cat((singleTensor,              torch.zeros([1, singleTensor.shape[1]]).to(device)),            dim = 0)
        singleTensor_Phase      = torch.cat((singleTensor_Phase,        torch.zeros([1, singleTensor_Phase.shape[1]]).to(device)),      dim = 0)
        singleTensor_inputPhase = torch.cat((singleTensor_inputPhase,   torch.zeros([1, singleTensor_inputPhase.shape[1]]).to(device)), dim = 0)

        # combining with the phase
        # reconstructionTensor = singleTensor * torch.exp(1j * singleTensor_Phase)
        reconstructionTensor = singleTensor * torch.exp(1j * singleTensor_inputPhase)
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
        
        # alpha blending the processed and raw based on <insert technique name> suggested by professor
        alphavalue = 0.75
        for i in range(len(denoised_samples)):
            denoised_samples[i] = alphavalue * denoised_samples[i] + \
                                  (1 - alphavalue) * input_list_of_tensors[i][:len(denoised_samples[i])]

        # calculating wer ------------------------------------------------
        list_of_transcriptions = []
        list_of_transcriptions_primary = []

        for sample_index in range(len(denoised_samples)):

            # taking a sample out of the list --------------------------------------------------
            denoised_sample_current = denoised_samples[sample_index].type(torch.float32)
            sample_current_primary  = input_list_of_tensors[sample_index].type(torch.float32)

            # finding transcript -------------------------------------------
            transcript_of_current_tensor            = model_whisper.transcribe(denoised_sample_current)["text"]
            transcript_of_current_tensor_primary    = model_whisper.transcribe(sample_current_primary)["text"]

            # adding whisper-transcript to list of output-transcripts
            list_of_transcriptions.append(transcript_of_current_tensor)
            list_of_transcriptions_primary.append(transcript_of_current_tensor_primary)

        # removing empty strings -----------------------------------------------------------------------------------
        filtered_gt_files,          filtered_transcriptions         =  fRemoveEmptyStrings(gt_files, list_of_transcriptions)
        filtered_gt_files_primary,  filtered_transcriptions_primary =  fRemoveEmptyStrings(gt_files, list_of_transcriptions_primary)

        # calculating word-error-rate --------------------------------------
        average_wer         = jiwer.wer(filtered_gt_files,          filtered_transcriptions)
        average_wer_primary = jiwer.wer(filtered_gt_files_primary,  filtered_transcriptions_primary)
        
        
    return average_wer, average_wer_primary

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

# Whisper ==================================================
import whisper
# model_whisper = whisper.load_model("tiny",    device = device)
# model_whisper = whisper.load_model("base",    device = device)
# model_whisper = whisper.load_model("small",   device = device)
# model_whisper = whisper.load_model("medium",  device = device)
# model_whisper = whisper.load_model("turbo",   device = device)
model_whisper = whisper.load_model("large",     device = device, 
                                   download_root = "/projectnb/mlrobot/vrs/.conda/envs/attenuate_env_mlrobot/lib/python3.8/site-packages/whisper/models")

# Spectrogram Function 
N_FFT = 1024;   N_HOP = 128
stft    = torchaudio.transforms.Spectrogram(n_fft=N_FFT,        hop_length=N_HOP,   power=None, normalized=True).to(device)
istft   = torchaudio.transforms.InverseSpectrogram(n_fft=N_FFT, hop_length=N_HOP,   normalized=True).to(device)





# Loading model ================================================================

# choosing model file to load
modelindex = 1
if modelindex == 0:
    modelMagnitude_path = "mymodels/train25_05_SNR6_MulMagAddPhase/modelMagnitude_11_29_17_52_31"
    modelPhase_path     = "mymodels/train25_05_SNR6_MulMagAddPhase/modelPhase_11_29_17_52_31"
elif modelindex == 1:
    modelMagnitude_path = "mymodels/train25_05_SNR3_MulMagAddPhase/modelMagnitude_11_29_17_52_29"
    modelPhase_path     = "mymodels/train25_05_SNR3_MulMagAddPhase/modelPhase_11_29_17_52_29"

# instantiating and loading model
model_denoiser      = WaveUNeT_Architecture().to(device)
model_denoiser.load_state_dict(torch.load(modelMagnitude_path, map_location=device))
model_phase         = WaveUNeT_Architecture().to(device)
model_phase.load_state_dict(torch.load(modelPhase_path, map_location=device))


# Directories = Directories = Directories = Directories = Directories = Directories
# Directories = Directories = Directories = Directories = Directories = Directories
# Directories = Directories = Directories = Directories = Directories = Directories
# Directories = Directories = Directories = Directories = Directories = Directories
# Directories = Directories = Directories = Directories = Directories = Directories

# getting current-time as string
currenttimestring = fFetchCurrentTimeString()
currentscriptname = sys.argv[0][:-3]
modelDirectoryPath = os.path.join("/projectnb/mlrobot/vrs/MSProject/aTENNuate/mymodels",            currentscriptname)
fCheckIfExistsIfNotCreate(modelDirectoryPath)

# path to figures/plots
directory_to_save_figures = os.path.join("/projectnb/mlrobot/vrs/MSProject/aTENNuate/plots",        currentscriptname)
fCheckIfExistsIfNotCreate(directory_to_save_figures)

# # path to audios
# directory_to_save_audio = os.path.join("/projectnb/mlrobot/vrs/MSProject/aTENNuate/AudioSamples",   currentscriptname)
# fCheckIfExistsIfNotCreate(directory_to_save_audio)

# path to saving important data
directory_to_save_data = os.path.join("/projectnb/mlrobot/vrs/MSProject/aTENNuate/Data",            currentscriptname)
fCheckIfExistsIfNotCreate(directory_to_save_data)

# record lists 
training_loss                       = [];       training_loss_averaged = []
training_loss_phase                 = []; training_loss_phase_averaged = []
baseline_magnitude_loss_list        = [] 
baseline_phase_loss_list            = []

transcription_train_loss_list       = []
average_wer_list                    = []


# creating two signals =========================================================
random.seed(1001)
with torch.no_grad():
    snr_level       = 9
    secondarypower  = primarypower - snr_level
    chooserandom = True
    if chooserandom == False: primaryindex    = 100; secondaryindex  = 200
    elif chooserandom == True: 
        primaryindex    = random.randint(0, len(list_of_test_files)-1)
        secondaryindex  = random.randint(0, len(list_of_test_files)-1)

    # choosing signals
    primarypath         = list_of_test_files[primaryindex]
    secondarypath       = list_of_test_files[secondaryindex]
    gt_primary          = fReturnGroundTruthTranscript(fCreateNameOfCorrespondingTxtFile(primarypath))
    gt_secondary        = fReturnGroundTruthTranscript(fCreateNameOfCorrespondingTxtFile(secondarypath))
    audioarrayPrimary   = fCreateTensorFromWavPath(primarypath)
    audioarraySecondary = fAdjustLengthOfSecondary(audioarrayPrimary,   fCreateTensorFromWavPath(secondarypath))
    audioarrayPrimary   = fRaiseSignalToPower(audioarrayPrimary,        primarypower)
    audioarraySecondary = fRaiseSignalToPower(audioarraySecondary,      secondarypower)
    audioarray          = audioarrayPrimary + audioarraySecondary
    audioarray          = audioarray/torch.max(torch.abs(audioarray))

    # running denoiser
    audioarrayProcessed = fTakeTensor_Process_ReturnSignal([audioarray],
                                                            model_denoiser,
                                                            model_phase,
                                                            stft,
                                                            istft)[0]
    alphavalue = 0.5
    audioarrayAlphaBlended  = alphavalue * audioarrayProcessed[:min(len(audioarrayProcessed), len(audioarray))] + \
                              (1 - alphavalue) * audioarray[:min(len(audioarrayProcessed), len(audioarray))]
    audioarrayProcessed = audioarrayProcessed/torch.max(torch.abs(audioarrayProcessed))

    # running whisper on both input and processed audio
    os.system('clear')
    print("input.transcribe     = ",    model_whisper.transcribe(audioarray)["text"])
    print("processed.transcribe = ",    model_whisper.transcribe(audioarrayProcessed)["text"])
    print("blended.transcribe   = ",    model_whisper.transcribe(audioarrayAlphaBlended)["text"])
    print("-"*100)
    print("gt:primary       = ", gt_primary)
    print("gt:secondary     = ", gt_secondary)

    # amplitude normalizing 
    audioarray              = audioarray/torch.max(torch.abs(audioarray))
    audioarrayPrimary       = audioarrayPrimary/torch.max(torch.abs(audioarrayPrimary))
    audioarrayProcessed     = audioarrayProcessed/torch.max(torch.abs(audioarrayProcessed))
    audioarraySecondary     = audioarraySecondary/torch.max(torch.abs(audioarraySecondary))
    audioarrayAlphaBlended  = audioarrayAlphaBlended/torch.max(torch.abs(audioarrayAlphaBlended))

    # save audio
    torchaudio.save(currentscriptname+"_input.wav",     audioarray.unsqueeze(0),                16000)
    torchaudio.save(currentscriptname+"_processed.wav", audioarrayProcessed.unsqueeze(0),       16000)
    # torchaudio.save(currentscriptname+"_primary.wav",   audioarrayPrimary.unsqueeze(0),         16000)
    # torchaudio.save(currentscriptname+"_secondary.wav", audioarraySecondary.unsqueeze(0),       16000)
    torchaudio.save(currentscriptname+"_blended.wav",   audioarrayAlphaBlended.unsqueeze(0),    16000)

























sys.exit()


# calculating the transcription error ==========================================================
# calculating the transcription error ==========================================================
# calculating the transcription error ==========================================================
average_wer, \
    average_wer_primary = \
        fProcessAndCalculateTranscriptionError(list_of_train_files = list_of_test_files, 
                                                num_samples         = len(list_of_test_files),
                                                primarypower        = primarypower, 
                                                secondarypower      = secondarypower,
                                                model_magnitude     = model_denoiser, 
                                                model_phase         = model_phase,
                                                stft                = stft,
                                                istft               = istft)

transcription_train_loss_list.append(average_wer)
average_wer_list    = fUpdateAverageWERList(transcription_train_loss_list, average_wer_list)
print(f"average_wer (SNR = {snr_level}) = {average_wer} | wer_gt = {average_wer_primary}");
