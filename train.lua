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

if not opt.noGPU then
    require 'cunn'
    require 'cutorch'
end

local datasetUtils = require 'datasetUtils'
local prepDataset = require 'prepareDataset'

-- train the model on the given dataset
function trainSequence(model, Combined_CNN_RNN, baseCNN, criterion, dataset, trainInds, testInds)
    local maxDiffSubSame = -1
    local minValLoss = 100000
    local dim = 3
    batchErr = 0
    if opt.disableOpticalFlow then
        dim = 3
    else
        dim = 5
    end
    local parameters, gradParameters = model:getParameters()
    info(string.format('Number of parameters:%d', parameters:size(1)))
    local nTrainPersons = dataset['train']:size()

    local optim_state = {
        learningRate = opt.learningRate,
        momentum = opt.momentum,
    }
    local inputs = {}
    for i=1, opt.trainBatchSize do
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


    for eph = 1, opt.nEpochs do
        local timer = torch.Timer()
        collectgarbage()

        local iteration_count = nTrainPersons
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
        for i = iteration_start,iteration_count,opt.trainBatchSize  do
            -- choose the mode / similar - diff
            local pushPull
            local netInputA
            local netInputB
            local classLabel1
            local classLabel2
            local netTarget
			local targets = {}
      
      local inputs_tensor = {}

			local pos_batch = dataset['train']:next_batch(opt.trainBatchSize / 2, 1)
			local neg_batch = dataset['train']:next_batch(opt.trainBatchSize / 2, 0)

			for i, val in ipairs(pos_batch) do
				local video1 = val[1]
				local video2 = val[2]
				local hid1 = tonumber(val[3])
				local hid2 = tonumber(val[4])
				local target = 1
				table.insert(inputs_tensor, {video1, video2})
				table.insert(targets, target)
			end
      
			for i, val in ipairs(neg_batch) do
				local video1 = val[1]
				local video2 = val[2]
				local hid1 = tonumber(val[3])
				local hid2 = tonumber(val[4])
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

            if ((i -1) / opt.trainBatchSize ) % 10 == 0 then
                model:evaluate()
                Combined_CNN_RNN:evaluate()
                local margin_loss = 0
                for i, input in ipairs(inputs) do
                    local output = model:forward(input)
                    output = output:double()
                    local netError = criterion:forward(output, targets[i])
                    margin_loss = margin_loss + netError
                end
                margin_loss = margin_loss / #inputs
                info(string.format('%05dth/%05d Margin Error %0.2f', i, iteration_count, margin_loss))
                if margin_loss < 0.3 then
                    local dirname = './trainedNets'
                    local saveFileNameConvnet = string.format('%s/convNet_%s_train_%0.2f_%05d.dat', dirname, opt.saveFileName, margin_loss, i)
                    torch.save(saveFileNameConvnet,Combined_CNN_RNN)
                end
                model:training()
                Combined_CNN_RNN:training()
            end

      

            -- note that due to a problem with SuperCriterion we must cast
            -- from CUDA to double and back before passing data to/from the
            -- criteiron layer - may be fixed in a future update of Torch...
            -- ... or maybe I'm just not using it right!
            local feval = function(x)

                batchError = 0.0
                if x ~= parameters then
                    parameters:copy(x)
                end
                gradParameters:zero()
				
				for i, input in ipairs(inputs) do
					--forward
					local output = model:forward(input)
                    output = output:double()
					local netError = criterion:forward(output, targets[i])
					--backward
					local gradCriterion = criterion:backward(output, targets[i])
          if not opt.noGPU then
            gradCriterion = gradCriterion:cuda()
          end
					model:backward(input, gradCriterion)
					batchError = batchError + netError
					gradParameters:clamp(-opt.gradClip, opt.gradClip)
				end
				gradParameters:div(#inputs)
				batchError = batchError/#inputs
                return batchError, gradParameters
            end
            optim.sgd(feval, parameters, optim_state)
            if opt.dataset == 3 then
                if ((i -1) / opt.trainBatchSize)  % 5 == 0 then
                    local time = timer:time().real
                    timer:reset()
                    local avg_loss = batchError
                    info(string.format('%05dth/%05d Batch Error %0.2f, time %0.1f', i, iteration_count, avg_loss, time))
                end
                if i ~= 1 and ((i -1) / opt.trainBatchSize) % opt.testLossBatch == 0 then
                    model:evaluate()
                    Combined_CNN_RNN:evaluate()
                    local val_loss = 0.0
                    for tmp_i = 1, opt.testLossBatchCount do
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
                        local output = model:forward({netInputAtable, netInputBtable})
                        output = output:double()
                        local model_dst = criterion:forward(output, target)
                        val_loss = val_loss + model_dst
                    end
                    dataset['val']:set_pos_index(1)
                    dataset['val']:set_neg_index(1)
                    local avg_loss = val_loss / opt.testLossBatchCount
                    info(string.format('validation loss:%0.2f at batch:%04d', avg_loss, opt.testLossBatchCount))
                    if avg_loss < minValLoss then
                        info(string.format('change min val loss from %0.2f to %0.2f', minValLoss, avg_loss))
                        minValLoss = avg_loss
                        local dirname = './trainedNets'
                        os.execute("mkdir  -p " .. dirname)
                        -- save the Model and Convnet (which is part of the model) to a file
                        local filename_tmp = string.format('%s/%%s_%s_%0.2f_eph_%d_itera_%06d_val_loss_batch_count_%06d.dat', dirname, opt.saveFileName, minValLoss, eph, i, opt.testLossBatchCount)
                        local saveFileNameModel = string.format(filename_tmp, 'fullModel')
                        torch.save(saveFileNameModel, model)

                        local saveFileNameConvnet = string.format(filename_tmp, 'convNet')
                        torch.save(saveFileNameConvnet, Combined_CNN_RNN)

                        local saveFileNameBasenet = string.format(filename_tmp, 'baseNet')
                        torch.save(saveFileNameBasenet, baseCNN)
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
                        local filename_tmp = string.format('%s/%%s_%s_%0.2f_cmc_avg_dst_diff.dat', dirname, opt.saveFileName, maxDiffSubSame)
                        local saveFileNameModel = string.format(filename_tmp, 'fullModel')
                        torch.save(saveFileNameModel, model)

                        local saveFileNameConvnet = string.format(filename_tmp, 'convNet')
                        torch.save(saveFileNameConvnet, Combined_CNN_RNN)

                        local saveFileNameBasenet = string.format(filename_tmp, 'baseNet')
                        torch.save(saveFileNameBasenet, baseCNN)
                    else
                        info(string.format('do not change maxdiff from %0.2f to %0.2f', maxDiffSubSame, avgDiff - avgSame))
                    end
                    model:training()
                    Combined_CNN_RNN:training()
                end
            end
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
