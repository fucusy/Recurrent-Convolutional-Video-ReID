-- Copyright (c) 2016 Niall McLaughlin, CSIT, Queen's University Belfast, UK
-- Contact: nmclaughlin02@qub.ac.uk
-- If you use this code please cite:
-- "Recurrent Convolutional Network for Video-based Person Re-Identification",
-- N McLaughlin, J Martinez Del Rincon, P Miller, 
-- IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
-- 
-- This software is licensed for research and non-commercial use only.
-- 
-- The above copyright notice and this permission notice shall be included in
-- all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
-- THE SOFTWARE.

require 'torch'
require 'nn'
require 'nnx'
require 'optim'

require 'image'
require 'paths'
require 'rnn'

cmd = torch.CmdLine()
cmd:option('-nEpochs',2,'number of training epochs')
cmd:option('-trainStart', 1,'start at index when trainig')
cmd:option('-valStart', 1,'start at index when validation')
cmd:option('-dataset',3,'1 -  ilids, 2 - prid, 3 - CASIA B clipped')
cmd:option('-sampleSeqLength',36,'length of sequence to train network')
cmd:option('-gradClip',5,'magnitude of clip on the RNN gradient')
cmd:option('-saveFileName','basicnet','name to save dataset file')
cmd:option('-dropoutFrac',0.6,'fraction of dropout to use between layers')
cmd:option('-dropoutFracRNN',0.6,'fraction of dropout to use between RNN layers')
cmd:option('-samplingEpochs',100,'how often to compute the CMC curve - dont compute too much - its slow!')
cmd:option('-seed',1,'random seed')
cmd:option('-learningRate',1e-3)
cmd:option('-momentum',0.9)
cmd:option('-nConvFilters',32)
cmd:option('-embeddingSize',32)
cmd:option('-hingeMargin',2)
cmd:option('-dataPath','/Volumes/Passport/data/gait-rnn', 'base data path')
cmd:option('-testBatch', 500000000, 'calculate cmc on validation every batch')
cmd:option('-testLossBatch', 10, 'calculate loss on validation every batch')
cmd:option('-testLossBatchCount', 10, 'calculate loss on validation every batch')
cmd:option('-trainBatch', 200000000, 'how many batch you train in every epoch')
cmd:option('-trainBatchSize', 2, 'how many intance in a traning batch')
cmd:option('-testValPor', 0.5, 'test on validation at proportion')
cmd:option('-loadModelFromFile', '', 'load fullmodel, rnn model, cnn model')


cmd:option('-usePredefinedSplit',false,'Use predefined test/training split loaded from a file')
cmd:option('-disableOpticalFlow',true,'use optical flow features or not')
cmd:option('-noGPU', true, 'do not use GPU')
cmd:option('-debug', false, 'debug mode or not')
cmd:option('-fcnModel', false, 'load self define full convontional network model')
cmd:option('-gpuDevice', 2, 'set gpu device')

opt = cmd:parse(arg)
print(opt)


require 'buildModel'
require 'train'
require 'test'
require 'tool'


local datasetUtils = require 'datasetUtils'
local prepDataset = require 'prepareDataset'



-- set the GPU
if not opt.noGPU then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(opt.gpuDevice)
end
function isnan(z)
    return z ~= z
end

torch.manualSeed(opt.seed)
-- cutorch.manualSeed(opt.seed)

-- change these paths to point to the place where you store i-lids or prid datasets
homeDir = paths.home
local seqRootRGB = ''
local seqRootOF = ''
if opt.dataset == 1 then
    seqRootRGB = opt.dataPath
    seqRootOF = homeDir .. '/data/i-LIDS-VID-OF-HVP/sequences/'
elseif opt.dataset == 2 then
    seqRootRGB = opt.dataPath
    seqRootOF = '/Volumes/Passport/data/prid_2011/multi_shot/'
else
    seqRootRGB = opt.dataPath
    seqRootOF = '/Volumes/Passport/data/gait-rnn-OF/'
end

info(string.format('loading Dataset - from %s, %s',seqRootRGB,seqRootOF))
local fileExt = '.jpg'
local dataset = prepDataset.prepareDatasetCASIA_B_RNN()

local train_count = 0

local person_count = 50

-- build the model
if opt.fcnModel then
    opt.embeddingSize = opt.nConvFilters * 10 * 8
    fullModel,criterion,Combined_CNN_RNN,baseCNN = buildModel_MeanPool_RNN_FCN(16,opt.nConvFilters,opt.nConvFilters,person_count)
else
    fullModel,criterion,Combined_CNN_RNN,baseCNN = buildModel_MeanPool_RNN(16,opt.nConvFilters,opt.nConvFilters,person_count)
end

print(fullModel)

-- if loadModelFromFile is set, load model from it
local filenames = splitByComma(opt.loadModelFromFile)

if #filenames >= 3 then
    fullModel = torch.load(filenames[1])
    info(string.format('loaded full model from %s', filenames[1]))

    Combined_CNN_RNN = torch.load(filenames[2])
    info(string.format('loaded CNN_RNN from %s', filenames[2]))

    baseCNN = torch.load(filenames[3])
    info(string.format('loaded cnn from %s', filenames[3]))
end

dataset['train']:set_pos_index(math.floor(opt.trainStart/2+1))
dataset['train']:set_neg_index(math.floor(opt.trainStart/2+1))

dataset['val']:set_pos_index(math.floor(opt.valStart/2+1))
dataset['val']:set_neg_index(math.floor(opt.valStart/2+1))

-- train the model
trainedModel,trainedConvnet,trainedBaseNet = trainSequence(fullModel,Combined_CNN_RNN,baseCNN,criterion,dataset,trainInds,testInds)

dirname = './trainedNets'
os.execute("mkdir  -p " .. dirname)
-- save the Model and Convnet (which is part of the model) to a file
saveFileNameModel = dirname .. '/fullModel_' .. opt.saveFileName .. '.dat'
torch.save(saveFileNameModel,trainedModel)

saveFileNameConvnet = dirname .. '/convNet_' .. opt.saveFileName .. '.dat'
torch.save(saveFileNameConvnet,trainedConvnet)

saveFileNameBasenet = dirname .. '/baseNet_' .. opt.saveFileName .. '.dat'
torch.save(saveFileNameBasenet,trainedBaseNet)

------------------------------------------------------------------------------------------------------------------------------------
-- Evaluation
------------------------------------------------------------------------------------------------------------------------------------

trainedConvnet:evaluate()
nTestImages = {1,2,4,8,16,32,64,128}

for n = 1,#nTestImages do
    info('test multiple images '..nTestImages[n])
    -- default method of computing CMC curve
    computeCMC_MeanPool_RNN(dataset,testInds,trainedConvnet,128,nTestImages[n])
end
