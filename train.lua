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

require 'cunn'
require 'cutorch'

require 'image'
require 'paths'
require 'rnn'

local datasetUtils = require 'datasetUtils'
local prepDataset = require 'prepareDataset'

-- train the model on the given dataset
function trainSequence(model, Combined_CNN_RNN, baseCNN, criterion, dataset, trainInds, testInds)
    local maxDiffSubSame = -1
    local minValLoss = 100000
    local dim = 3
    if opt.disableOpticalFlow then
        dim = 3
    else
        dim = 5
    end
    local parameters, gradParameters = model:getParameters()
    info(string.format('Number of parameters:%d', parameters:size(1)))

    local batchErr = 0
    local nTrainPersons = 0
    if opt.dataset == 3 then
        nTrainPersons = dataset['train']:size()
    else
        nTrainPersons = trainInds:size(1)
    end

    local optim_state = {
        learningRate = opt.learningRate,
        momentum = opt.momentum,
    }

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
    local netInput = { x, y }

    for eph = 1, opt.nEpochs do

        local timer = torch.Timer()
        collectgarbage()

        local order = '';
        local iteration_count = 0
        if opt.dataset == 1 or opt.dataset == 2 then
            iteration_count = nTrainPersons * 2
            order = torch.randperm(nTrainPersons)
        else
            iteration_count = nTrainPersons
        end
        if opt.debug then
            iteration_count = 100
        end
        if opt.trainBatch < iteration_count then
            iteration_count = opt.trainBatch
        end

        local iteration_start = 1
        if eph == 1 then
            iteration_start = opt.trainStart + 1
        end
        for i = iteration_start, iteration_count do

            -- choose the mode / similar - diff
            local pushPull
            local netInputA
            local netInputB
            local classLabel1
            local classLabel2
            local netTarget

            if opt.dataset == 3 then
                local is_pos = 1
                local target = 1
                if i % 2 == 0 then
                    is_pos = 1
                    target = 1
                else
                    is_pos = 2
                    target = -1
                end
                local batch = dataset['train']:next_batch(1, is_pos)
                netInputA = batch[1][1]
                netInputB = batch[1][2]
                local hidA = tonumber(batch[1][3])
                local hidB = tonumber(batch[1][4])
                netTarget = { target, hidA, hidB }

            elseif opt.dataset == 1 or opt.dataset == 2 then
                if i % 2 == 0 then
                    -- choose a positive pair, both sequences show the same person
                    local camA = 1
                    local camB = 2
                    local startA, startB, seq_length
                    startA, startB, seq_length = datasetUtils.getPosSample(dataset, trainInds, order[i / 2], opt.sampleSeqLength)
                    netInputA = dataset[trainInds[order[i / 2]]][camA][{ { startA, startA + seq_length - 1 }, {}, {} }]:squeeze()
                    netInputB = dataset[trainInds[order[i / 2]]][camB][{ { startB, startB + seq_length - 1 }, {}, {} }]:squeeze()
                    netTarget = { 1, (order[i / 2]), (order[i / 2]) }
                else
                    -- choose a negative pair, both sequences show different persons
                    local pushPull = -1
                    local seqA, seqB, camA, camB, startA, startB, seq_length
                    seqA, seqB, camA, camB, startA, startB, seq_length = datasetUtils.getNegSample(dataset, trainInds, opt.sampleSeqLength)
                    netInputA = dataset[trainInds[seqA]][camA][{ { startA, startA + seq_length - 1 }, {}, {}, {} }]:squeeze()
                    netInputB = dataset[trainInds[seqB]][camB][{ { startB, startB + seq_length - 1 }, {}, {}, {} }]:squeeze()
                    netTarget = { -1, seqA, seqB }
                end
            end


            -- set the parameters for data augmentation. Note that we apply the same augmentation
            -- to the whole sequence, rather than individual images
            local crpxA = torch.floor(torch.rand(1):squeeze() * 8) + 1
            local crpyA = torch.floor(torch.rand(1):squeeze() * 8) + 1
            local crpxB = torch.floor(torch.rand(1):squeeze() * 8) + 1
            local crpyB = torch.floor(torch.rand(1):squeeze() * 8) + 1
            local flipA = torch.floor(torch.rand(1):squeeze() * 2) + 1
            local flipB = torch.floor(torch.rand(1):squeeze() * 2) + 1

            -- deal with the case where we have only a single image i.e. meanpool size == 1
            if netInputA:dim() == 3 then
                netInputA:resize(1, netInputA:size(1), netInputA:size(2), netInputA:size(3))
                netInputB:resize(1, netInputB:size(1), netInputB:size(2), netInputB:size(3))
            end

            -- we can't (easily) deal with sequenes that are too short (complicates the code) - just skip them for now...
            -- will try to deal with this later...
            if netInputA:size(1) ~= opt.sampleSeqLength or netInputB:size(1) ~= opt.sampleSeqLength then
                goto continue -- yuck!
            end

            netInputA = doDataAug(netInputA, crpxA, crpyA, flipA)
            netInputB = doDataAug(netInputB, crpxB, crpyB, flipB)

            for t = 1, opt.sampleSeqLength do
                netInput[1][t]:copy(netInputA[{ { t }, {}, {}, {} }]:squeeze())
                netInput[2][t]:copy(netInputB[{ { t }, {}, {}, {} }]:squeeze())
            end

            -- note that due to a problem with SuperCriterion we must cast
            -- from CUDA to double and back before passing data to/from the
            -- criteiron layer - may be fixed in a future update of Torch...
            -- ... or maybe I'm just not using it right!
            local feval = function(x)

                local batchError = 0
                if x ~= parameters then
                    parameters:copy(x)
                end
                gradParameters:zero()

                --forward
                local output = model:forward(netInput)
                for p = 1, #output do
                    output[p] = output[p]:double()
                end
                local netError = criterion:forward(output, netTarget)

                --backward
                local gradCriterion = criterion:backward(output, netTarget)
                for c = 1, #gradCriterion do
                    if not opt.noGPU then
                        gradCriterion[c] = gradCriterion[c]:cuda()
                    end
                end
                model:backward(netInput, gradCriterion)

                batchErr = batchErr + netError
                gradParameters:clamp(-opt.gradClip, opt.gradClip)

                return batchError, gradParameters
            end
            optim.sgd(feval, parameters, optim_state)
            if opt.dataset == 3 then
                if i % 50 == 0 then
                    local time = timer:time().real
                    timer:reset()
                    local avg_loss = batchErr / 50
                    info(string.format('%05dth/%05d Batch Error %0.2f, time %0.1f', i, iteration_count, avg_loss, time))
                    batchErr = 0
                end
                if i % opt.testLossBatch == 0 then
                    model:evaluate()
                    Combined_CNN_RNN:evaluate()
                    local val_loss = 0.0
                    for tmp_i=1, opt.testLossBatchCount do
                        local is_pos, target
                        if tmp_i % 2 == 0 then
                            is_pos = 1
                            target = 1
                        else
                            is_pos = 2
                            target = -1
                        end
                        local batch = dataset['val']:next_batch(1, is_pos)
                        netInputA = batch[1][1]
                        netInputB = batch[1][2]
                        local crpxA = torch.floor(torch.rand(1):squeeze() * 8) + 1
                        local crpyA = torch.floor(torch.rand(1):squeeze() * 8) + 1
                        local crpxB = torch.floor(torch.rand(1):squeeze() * 8) + 1
                        local crpyB = torch.floor(torch.rand(1):squeeze() * 8) + 1
                        local flipA = torch.floor(torch.rand(1):squeeze() * 2) + 1
                        local flipB = torch.floor(torch.rand(1):squeeze() * 2) + 1

                        -- deal with the case where we have only a single image i.e. meanpool size == 1
                        if netInputA:dim() == 3 then
                            netInputA:resize(1, netInputA:size(1), netInputA:size(2), netInputA:size(3))
                            netInputB:resize(1, netInputB:size(1), netInputB:size(2), netInputB:size(3))
                        end

                        -- we can't (easily) deal with sequenes that are too short (complicates the code) - just skip them for now...
                        -- will try to deal with this later...
                        if netInputA:size(1) ~= opt.sampleSeqLength or netInputB:size(1) ~= opt.sampleSeqLength then
                            goto continue -- yuck!
                        end

                        netInputA = doDataAug(netInputA, crpxA, crpyA, flipA)
                        netInputB = doDataAug(netInputB, crpxB, crpyB, flipB)
                        local netInputAtable = {}
                        local netInputBtable = {}
                        for t = 1, opt.sampleSeqLength do
                            if opt.noGPU then
                                netInputAtable[t] = netInputA[{ { t }, {}, {}, {} }]:squeeze():clone()
                                netInputBtable[t] = netInputB[{ { t }, {}, {}, {} }]:squeeze():clone()
                            else
                                netInputAtable[t] = netInputA[{ { t }, {}, {}, {} }]:squeeze():cuda():clone()
                                netInputBtable[t] = netInputB[{ { t }, {}, {}, {} }]:squeeze():cuda():clone()
                            end
                        end
                        
                        local vectorA = Combined_CNN_RNN:forward(netInputAtable):double()
                        local vectorB = Combined_CNN_RNN:forward(netInputBtable):double()
                        local dst = torch.sqrt(torch.sum(torch.pow(vectorA - vectorB,2)))
                        if target == 1 then
                            val_loss = val_loss + dst
                        else
                            local margin = opt.hingeMargin
                            if margin - dst > 0 then
                                val_loss = val_loss + margin - dst
                            end
                        end
                    end

                    local avg_loss = val_loss / opt.testLossBatchCount
                    info(string.format('validation loss:%0.2f at batch:%04d', avg_loss, opt.testLossBatchCount))
                    if avg_loss < minValLoss then
                        info(string.format('change min val loss from %0.2f to %0.2f', minValLoss, avg_loss))
                        minValLoss = avg_loss
                        local dirname = './trainedNets'
                        os.execute("mkdir  -p " .. dirname)
                        -- save the Model and Convnet (which is part of the model) to a file
                        local filename_tmp = string.format('%s/%%s_%s_%0.2f_eph_%d_itera_%06d_val_loss_batch_count_%06d.dat'
                            , dirname,opt.saveFileName, minValLoss, eph, i, opt.testLossBatchCount)
                        local saveFileNameModel = string.format(filename_tmp, 'fullModel')
                        torch.save(saveFileNameModel,model)

                        local saveFileNameConvnet = string.format(filename_tmp, 'convNet')
                        torch.save(saveFileNameConvnet,Combined_CNN_RNN)

                        local saveFileNameBasenet = string.format(filename_tmp, 'baseNet')
                        torch.save(saveFileNameBasenet,baseCNN)
                    else
                        info(string.format('do not change val loss from %0.2f to %0.2f', minValLoss, avg_loss))
                    end
                    model:training()
                    Combined_CNN_RNN:training()
                end
                if i % opt.testBatch == 0 then
                    model:evaluate()
                    Combined_CNN_RNN:evaluate()
                    local avgSame, avgDiff
                    avgSame, avgDiff = compute_across_view_precision_casia(dataset['val'], Combined_CNN_RNN, opt.embeddingSize, opt.sampleSeqLength, opt.testValPor)
                    if avgDiff - avgSame > maxDiffSubSame then
                        info(string.format('change maxdiff from %0.2f to %0.2f', maxDiffSubSame, avgDiff - avgSame))
                        maxDiffSubSame = avgDiff - avgSame
                        local dirname = './trainedNets'
                        os.execute("mkdir  -p " .. dirname)
                        -- save the Model and Convnet (which is part of the model) to a file
                        local filename_tmp = string.format('%s/%%s_%s_%0.2f_cmc_avg_dst_diff.dat', dirname,opt.saveFileName, maxDiffSubSame)
                        local saveFileNameModel = string.format(filename_tmp, 'fullModel')
                        torch.save(saveFileNameModel,model)

                        local saveFileNameConvnet = string.format(filename_tmp, 'convNet')
                        torch.save(saveFileNameConvnet,Combined_CNN_RNN)

                        local saveFileNameBasenet = string.format(filename_tmp, 'baseNet')
                        torch.save(saveFileNameBasenet,baseCNN)
                    else
                        info(string.format('do not change maxdiff from %0.2f to %0.2f', maxDiffSubSame, avgDiff - avgSame))
                    end

                    model:training()
                    Combined_CNN_RNN:training()

                end
            end
            :: continue :: end
            if opt.dataset == 1 or opt.dataset == 2 then
                local time = timer:time().real
                timer:reset()
                info(string.format('%05dth epoch Batch Error %0.2f, time %0.1f', eph, iteration_count, batchErr, time))
                batchErr = 0
            end
            if (eph % opt.samplingEpochs == 0) then
                model:evaluate()
                Combined_CNN_RNN:evaluate()
                if opt.dataset == 1 or opt.dataset == 2 then
                    local cmcTest, cmcTrain, simMatTest, simMatTrain
                    cmcTest, simMatTest = computeCMC_MeanPool_RNN(dataset, testInds, Combined_CNN_RNN, opt.embeddingSize, opt.sampleSeqLength)
                    cmcTrain, simMatTrain = computeCMC_MeanPool_RNN(dataset, trainInds, Combined_CNN_RNN, opt.embeddingSize, opt.sampleSeqLength)
                    local outStringTest = 'Test  '
                    local outStringTrain = 'Train '
                    local printInds = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }
                    for c = 1, #printInds do
                        if c < nTrainPersons then
                            outStringTest = outStringTest .. torch.floor(cmcTest[printInds[c]]) .. ' '
                            outStringTrain = outStringTrain .. torch.floor(cmcTrain[printInds[c]]) .. ' '
                        end
                    end
                    info(outStringTest)
                    info(outStringTrain)
                else
                end

                model:training()
                Combined_CNN_RNN:training()
            end
        end
        return model, Combined_CNN_RNN, baseCNN
    end

    -- perform data augmentation to a sequence of images stored in a torch tensor
    function doDataAug(seq, cropx, cropy, flip)
        local seqLen = seq:size(1)
        local seqChnls = seq:size(2)
        local seqDim1 = seq:size(3)
        local seqDim2 = seq:size(4)

        -- info(seqLen,seqChnls,seqDim1,seqDim2,cropx,cropy)

        local daData = torch.zeros(seqLen, seqChnls, seqDim1 - 8, seqDim2 - 8)
        for t = 1, seqLen do
            -- do the data augmentation here
            local thisFrame = seq[{ { t }, {}, {}, {} }]:squeeze():clone()
            if flip == 1 then
                thisFrame = image.hflip(thisFrame)
            end

            thisFrame = image.crop(thisFrame, cropx, cropy, 40 + cropx, 56 + cropy)
            thisFrame = thisFrame - torch.mean(thisFrame)

            daData[{ { t }, {}, {}, {} }] = thisFrame
        end
        return daData
    end