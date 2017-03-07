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


require 'image'
require 'paths'
require 'rnn'


cmd = torch.CmdLine()
cmd:option('-loadModelFromFile', './trainedNets/convNet_basicnet_1.00_eph_1_itera_000110_val_loss_batch_count_000010.dat','load rnn model')
cmd:option('-embeddingSize',64)
cmd:option('-sampleSeqLength',36,'length of sequence to train network')
cmd:option('-dataPath','/Volumes/Passport/data/gait-rnn', 'base data path')
cmd:option('-disableOpticalFlow',true,'use optical flow features or not')
cmd:option('-data', 'test', 'test, val, or train')
cmd:option('-noGPU', false, 'do not use GPU')

opt = cmd:parse(arg)
print(opt)

require 'buildModel'
require 'test'
require 'tool'

local prepDataset = require 'prepareDataset'
-- set the GPU
if opt.gpu then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(2)
end

Combined_CNN_RNN = torch.load(opt.loadModelFromFile)
info(string.format('loaded CNN_RNN from %s', opt.loadModelFromFile))
print(Combined_CNN_RNN)
Combined_CNN_RNN:evaluate()
dataset = prepDataset.prepareDatasetCASIA_B_RNN()
avgSame, avgDiff = compute_across_view_precision_casia(dataset[opt.data], Combined_CNN_RNN, opt.embeddingSize, opt.sampleSeqLength, 1)
