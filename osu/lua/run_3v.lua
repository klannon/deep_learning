----------------------------------------------------------------------
-- Train a ConvNet on faces.
--
-- original: Clement Farabet
-- new version by: E. Culurciello 
-- Mon Oct 14 14:58:50 EDT 2013
----------------------------------------------------------------------

require 'pl'
require 'trepl'
require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> processing options')

opt = lapp[[
   -r,--learningRate       (default 1e-3)        learning rate
   -d,--learningRateDecay  (default 1e-7)        learning rate decay (in # samples)
   -w,--weightDecay        (default 1e-5)        L2 penalty on the weights
   -m,--momentum           (default 0.1)         momentum
   -d,--dropout            (default 0.5)         dropout amount
   -b,--batchSize          (default 16)         batch size
   -t,--threads            (default 1)           number of threads
   -p,--type               (default float)       float or cuda
   -i,--devid              (default 1)           device ID (if using CUDA)
   -s,--size               (default small)       dataset: small or full or extra
   -o,--save               (default results)     save directory
   -g,--model              (default mlp)     save directory
      --patches            (default all)         percentage of samples to use for testing'
      --visualize          (default true)        visualize dataset
]]
print(opt.save)
print(opt.threads)
-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- type:
if opt.type == 'cuda' then
   print(sys.COLORS.red ..  '==> switching to CUDA')
   require 'cunn'
   print(sys.COLORS.red ..  '==> about to setDevice')
   print(opt.devid)
   cutorch.setDevice(opt.devid) -- opt.devid
   print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> load modules')

require 'csvData'


-- classes: GLOBAL var!
classes = {'ttbar','wjets'}
local nameChannels = {'em','had','jets'}
local path_train = '/scratch365/cdablain/train_all_3v_ttbar_wjet.txt'
local path_test = '/scratch365/cdablain/test_all_3v_ttbar_wjet.txt'
print("getData call:")
trainData,valData,testData = csvData.getData(path_train,path_test)
print("done with data call, now loading train/test")

--print(trainData.data[1])

local train = require 'train'
local mytest  = require 'mytest'

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> training!')

while true do
   train(trainData)
   mytest(valData)
end

