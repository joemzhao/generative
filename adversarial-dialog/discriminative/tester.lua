local Data = require("./data")
local params=require("./dis_parse")

Data:Initial(params)
local open_pos_train_file = io.open(params.pos_train_file,"r")
local open_neg_train_file = io.open(params.neg_train_file,"r")

while true do
  Data:read_train(open_pos_train_file, open_neg_train_file)
end
