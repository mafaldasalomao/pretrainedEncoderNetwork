%Create an encoder with three downsampling operations based on the SqueezeNet pretrained network.

encoderNet = pretrainedEncoderNetwork('squeezenet',3)
analyzeNetwork(encoderNet)


%%
%Create U-Net from Pretrained GoogLeNet
depth = 4;
%Create a GAN encoder network with four downsampling operations from a pretrained GoogLeNet network.
[encoder,outputNames] = pretrainedEncoderNetwork('googlenet',depth);
%Determine the input size of the encoder network.
inputSize = encoder.Layers(1).InputSize;
%Determine the output size of the activation layers in the encoder network by creating a sample data input and then calling forward, which returns the activations.
exampleInput = dlarray(zeros(inputSize),'SSC');
exampleOutput = cell(1,length(outputNames));
[exampleOutput{:}] = forward(encoder,exampleInput,'Outputs',outputNames);
%Determine the number of channels in the decoder blocks as the length of the third channel in each activation.
numChannels = cellfun(@(x) size(extractdata(x),3),exampleOutput);
numChannels = fliplr(numChannels(1:end-1));
%Define a function that creates an array of layers for one decoder block.
decoderBlock = @(block) [
    transposedConv2dLayer(2,numChannels(block),'Stride',2)
    convolution2dLayer(3,numChannels(block),'Padding','same')
    reluLayer
    convolution2dLayer(3,numChannels(block),'Padding','same')
    reluLayer];

decoder = blockedNetwork(decoderBlock,depth);

%Create the U-Net network by connecting the encoder module and decoder module and adding skip connections
net = encoderDecoderNetwork([224 224 3],encoder,decoder, ...
   'OutputChannels',3,'SkipConnections','concatenate')

analyzeNetwork(net)






