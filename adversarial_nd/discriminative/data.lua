require "torchx"
local stringx = require("pl.stringx")
local Data = {}

local function split(str)
  local split = stringx.split(str, " ");
  local tensor = torch.Tensor(1, #split):zero()
  local count = 0;
  for i = 1, #split do
      if split[i] ~= nil and split[i] ~= "" then
        count = count + 1
        tensor[1][count] = tonumber(split[i]);
      end
  end
  return tensor;
end

function Data:split(str)
  local split = stringx.split(str, " ");
  local tensor = torch.Tensor(1, #split):zero()
  local count = 0;
  for i = 1, #split do
      if split[i] ~= nil and split[i] ~= "" then
        count = count + 1
        tensor[1][count] = tonumber(split[i]);
      end
  end
  return tensor;
end

function Data:Initial(params)
  self.params = params;
end

print(Data)
