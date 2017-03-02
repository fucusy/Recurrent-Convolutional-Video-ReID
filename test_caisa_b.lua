--
-- Created by IntelliJ IDEA.
-- User: fucus
-- Date: 28/02/2017
-- Time: 10:34 PM
-- To change this template use File | Settings | File Templates.
--

require 'torch'
require 'nn'
require 'nnx'
require 'optim'

require 'cunn'
require 'cutorch'

require 'image'
require 'paths'
require 'rnn'

require 'buildModel'
require 'train'
require 'test'
require 'tool'

local prepDataset = require 'prepareDataset'

cmd = torch.CmdLine()
cmd:option('-loadModelFromFile', './trainedNets/convNet_basicnet_1798.47_val_loss_batch_count_001000.dat,','load rnn model')
cmd:option('-embeddingSize',128)
cmd:option('-sampleSeqLength',16,'length of sequence to train network')
cmd:option('-dataPath','/Volumes/Passport/data/gait-rnn', 'base data path')
cmd:option('-disableOpticalFlow',false,'use optical flow features or not')
cmd:option('-noGPU', false, 'do not use GPU')

opt = cmd:parse(arg)
print(opt)
-- set the GPU
if not opt.noGPU then
    cutorch.setDevice(2)
end

Combined_CNN_RNN = torch.load(opt.loadModelFromFile)
info(string.format('loaded CNN_RNN from %s', opt.loadModelFromFile))
Combined_CNN_RNN:evaluate()
dataset = prepDataset.prepareDatasetCASIA_B_RNN()
avgSame, avgDiff = compute_across_view_precision_casia(dataset['test'], Combined_CNN_RNN, opt.embeddingSize, opt.sampleSeqLength, 1)
