require "nn"
require "nngraph"
Data = require "data"

local model = {}
model.Data = Data

function model:Initial(params)
  print("--- initialising the discriminator ---")
  self.Data:Initial(params)
  self.params = params
  -- self.lstm_word = self:lstm_(true)
  -- self.lstm_sen = self:lstm_(false)
  self.lstms_word = {}
  self.store_word = {}


end

function model:softmax_()
  local y = nn.Identity()()
  local h = nn.Identity()()
  local h2y = nn.Linear(self.params.dimension, 2)(h)
  local pred = nn.LogSoftMax()(h2y)
  local Criterion = nn.ClassNLLCriterion()
  local err = Criterion({pred, y})
  local module = nn.gModule({h, y}, {err, pred})
  module:getParameters():uniform(-self.params.init_weight, self.params.init_weight)
  return module
end

------------ None training helpers -------------
--@override copy(A)
local function copy(A)
  local B
  if A:nDimension() == 1 then
    B = torch.Tensor(A:size(1))
  end
  if A:nDimension() == 2 then
    B = torch.Tensor(A:size(1), A:size(2))
  end
  if A:nDimension() == 3 then
    B = torch.Tensor(A:size(1), A:size(2), A:size(3))
  end
  B:copy(A)
  return B
end
--@override clone_(A)
function model:clone_(A)
  local B = {}
  for i = 1, #A do
    if A[i]:nDimension() == 2 then
      B[i] = torch.Tensor(A[i]:size(1), A[i]:size(2))
    else
      B[i] = torch.Tensor(A[i]:size(1))
    end
    B[i]:copy(A[i])
  end
  return B
end


------------ Administrative helpers -------------
function model:readModel()
  local file = torch.DiskFile(self.params.model_file, "r"):binary()
  local model_params = file:readObject()
  file:close()
  for i = 1, #self.Modules do
    local parameter, _ = self.Modules[i]:parameters()
    for j = 1, #parameter do
      parameter[j]:copy(model_params[i][j])
    end
  end
  print("Finish reading existing model.")
end

function model:save()
  local params = {}
  for i = 1, #self.Modules do
    params[i], _ = self.Modules[i]:parameters()
  end
  local file = torch.DiskFile(self.params.save_model_path.."/iter"..self.iter, "w"):binary()
  file:writeObject(params)
  file:close()
end

function model:saveParams()
  local file = torch.DiskFile(self.params.save_params_file, "w"):binary()
  file:writeObject(self.params)
  file:close()
end

------------ Temporary evaluating -------------
para = {
        batch_size = 2,
        dialogue_length = 2,
        dimension = 10,
        init_weight = 1
       }
model:Initial(para)
model:softmax_()
