require "nngraph"
local params=require("./parse")
local model=require("./atten");

model:Initial(params)
-- model:readModel()
model:train()
