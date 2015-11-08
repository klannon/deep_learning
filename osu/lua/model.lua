----------------------------------------------------------------------
-- Create CNN and loss to optimize.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'Dropout' -- Hinton dropout technique


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> define parameters')

-- 2-class problem: faces!
local noutputs = 2
local ninputs = trainData.data:size(2) -- 15 for the OSU data
print("numinputs "..ninputs)
local nhiddens = 100


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> construct CNN')
print(ninputs)
local CNN = nn.Sequential()
if opt.model == 'linear' then

   -- Simple linear model
   CNN:add(nn.Reshape(ninputs))
   CNN:add(nn.Linear(ninputs,noutputs))
   CNN:add(nn.LogSoftMax())

elseif opt.model == 'mlp' then
   -- Simple 2-layer neural network, with tanh hidden units
   CNN:add(nn.Reshape(ninputs))
   CNN:add(nn.Linear(ninputs,nhiddens))
   CNN:add(nn.ReLU())
   CNN:add(nn.Linear(nhiddens,nhiddens))
   CNN:add(nn.ReLU())
   CNN:add(nn.Linear(nhiddens,nhiddens))
   CNN:add(nn.ReLU())
   CNN:add(nn.Linear(nhiddens,noutputs))
   CNN:add(nn.LogSoftMax())

end

local model = nn.Sequential()
model:add(CNN)

-- Loss: NLL
loss = nn.ClassNLLCriterion()


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the CNN:')
print(model)

if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

-- return package:
return {
   model = model,
   loss = loss,
}

