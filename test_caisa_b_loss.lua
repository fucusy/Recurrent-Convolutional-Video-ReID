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
cmd:option('-hingeMargin',2)
cmd:option('-dataPath','/Volumes/Passport/data/gait-rnn', 'base data path')
cmd:option('-disableOpticalFlow',true,'use optical flow features or not')
cmd:option('-data', 'test', 'test, val, or train')
cmd:option('-noGPU', false, 'do not use GPU')
cmd:option('-trainBatchSize', 2, 'how many intance in a traning batch')

opt = cmd:parse(arg)
print(opt)

require 'buildModel'
require 'test'
require 'tool'

local prepDataset = require 'prepareDataset'
-- set the GPU
if not opt.noGPU then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(2)
end
local dim = 3
if opt.disableOpticalFlow then
    dim = 3
else
    dim = 5
end

model = torch.load(opt.loadModelFromFile)
info(string.format('loaded full model from %s', opt.loadModelFromFile))
print(model)
model:evaluate()
criterion = nn.HingeEmbeddingCriterion(opt.hingeMargin)
dataset = prepDataset.prepareDatasetCASIA_B_RNN()
local data = dataset[opt.data]

local inputs = {}
for i=1, 2 * opt.trainBatchSize do
    local x = {}
    local y = {}
    for t = 1, opt.sampleSeqLength do
        if not opt.noGPU then
            table.insert(x, torch.zeros(dim, 56, 40):cuda())
            table.insert(y, torch.zeros(dim, 56, 40):cuda())
        else
            table.insert(x, torch.zeros(dim, 56, 40))
            table.insert(y, torch.zeros(dim, 56, 40))
        end
    end
    local input = { x, y }
    table.insert(inputs, input)
end

local all_loss = 0
local count = 0
for i = 1,data:size(),opt.trainBatchSize  do
    info('start to load batch data')
    local targets = {}

    local inputs_tensor = {}

    local pos_batch = data:next_batch(opt.trainBatchSize, 1)
    local neg_batch = data:next_batch(opt.trainBatchSize, 0)

    for i, val in ipairs(pos_batch) do
        local video1 = val[1]
        local video2 = val[2]
        local target = 1
        table.insert(inputs_tensor, {video1, video2})
        table.insert(targets, target)
    end

    for i, val in ipairs(neg_batch) do
        local video1 = val[1]
        local video2 = val[2]
        local target = -1
        table.insert(inputs_tensor, {video1, video2})
        table.insert(targets, target)
    end

    for i=1, #inputs_tensor do
        for t=1, opt.sampleSeqLength do
            inputs[i][1][t]:copy(inputs_tensor[i][1][{{t}, {}, {}, {}}])
            inputs[i][2][t]:copy(inputs_tensor[i][2][{{t}, {}, {}, {}}])
        end
    end
    info('loaded batch data')
    local margin_loss = 0
    local margin_loss_self_cal = 0
    for i, input in ipairs(inputs) do
        local output = model:forward(input)
        output = output:double()
        local netError = criterion:forward(output, targets[i])
        margin_loss = margin_loss + netError
        if targets[i] == 1 then
            margin_loss_self_cal = margin_loss_self_cal + output[1]
        else
            if opt.hingeMargin - output[1] > 0 then
                margin_loss_self_cal = margin_loss_self_cal + opt.hingeMargin - output[1]
            end
        end
    end
    margin_loss = margin_loss / #inputs
    margin_loss_self_cal = margin_loss_self_cal / #inputs
    all_loss = all_loss + margin_loss
    count = count + 1

    info(string.format('batch loss: %0.2f', margin_loss))
    info(string.format('batch loss self calculate: %0.2f', margin_loss_self_cal))
    info(string.format('batch avg: %0.2f / %05d = %0.2f',all_loss, count, all_loss / count))
end
