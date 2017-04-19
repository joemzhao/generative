local stringx = require('pl.stringx')
local cmd = torch.CmdLine()
cmd:option("-batch_size",128,"batch size")
cmd:option("-dimension",512,"vector dimensionality")
cmd:option("-dropout",0.2,"dropout rate")
cmd:option("-train_file","../data/t_given_s_train.txt","")
cmd:option("-dev_file","../data/t_given_s_dev.txt","")
cmd:option("-test_file","../data/t_given_s_test.txt","")
cmd:option("-init_weight",0.1,"")
cmd:option("-alpha",0.1,"")
cmd:option("-start_halve",6,"")
cmd:option("-max_length",50,"");
cmd:option("-vocab_source",25010,"")
cmd:option("-vocab_target",25010,"")
cmd:option("-thres",5,"gradient clipping thres")
cmd:option("-max_iter",8,"max number of iteration")
cmd:option("-source_max_length",50,"")
cmd:option("-target_max_length",50,"")
cmd:option("-layers",2,"")
cmd:option("-saveFolder","save","")
cmd:option("-reverse",false,"")
cmd:option("-saveModel",true,"")
cmd:option("-dictPath","../data/movie_25000","")
-- cmd:option("-model_file","save/model1","")
local params= cmd:parse(arg)
params.save_prefix=params.saveFolder.."/model"
params.save_params_file=params.saveFolder.."/params"
params.output_file=params.saveFolder.."/log"
params.model_file = params.saveFolder.."/model1"
paths.mkdir(params.saveFolder)
print(params)
return params;
