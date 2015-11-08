----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining some tools')

-- model:
local t = require 'model'
local model = t.model
local loss = t.loss

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes) -- faces: yes, no

-- Logger:
local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Batch test:

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining test procedure')

-- test function
function mytest(valData)
   -- local vars
   local time = sys.clock()
print("valdata size: "..valData.data:size(1))
print("valdata size: "..valData.data:size(2))
local inputs = torch.Tensor(opt.batchSize,valData.data:size(2)) -- get size from data
--print("input "..inputs)
local targets = torch.Tensor(opt.batchSize)



if opt.type == 'cuda' then 
   inputs = inputs:cuda()
   targets = targets:cuda()
end

--print("targets "..targets)
   -- test over test data
   print(sys.COLORS.red .. '==> testing on test set:')
   for t = 1,valData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, valData:size())

      -- batch fits?
      if (t + opt.batchSize - 1) > valData:size() then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         inputs[idx] = valData.data[i]
         targets[idx] = valData.labels[i]
         idx = idx + 1
      end

      -- test sample
      local preds = model:forward(inputs)

      -- confusion
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / valData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end
   confusion:zero()
   
end

-- Export:
return mytest

