
require "nn"
require "opt"


local target = torch.randn(3,3)

print("target", target)

local x = torch.Tensor(3,5):fill(0)
local y = torch.Tensor(5,3):fill(0)

local model = nn.MM()
local criterion = nn.MSECriterion()

local dfdx = torch.Tensor():resizeAs(x);
local dfdy = torch.Tensor():resizeAs(y);

local fevalx = function(x)
	dfdx:fill(0)
	
	local loss = criterion:forward( model:forward({x, y}) ,target )
	model:backward( {x, y} , criterion:backward(model.output, target))
	return loss, dfdx
end

local fevaly = function(x)
	dfdy:fill(0)
	local loss = criterion:forward( model:forward({x, y}, target))
	model:backward({x,y},  criterion:backward(model.output, target))
	return loss, dfdy
end

for i=1,100 do
	x, fx = optim.sgd(feval, x, { learningRate=1e-3, learningRateDecay=1e-4 })
    y, fy = optim.sgd(feval, y, { learningRate=1e-3, learningRateDecay=1e-4 })
	print("loss", fx[1], fy[1])
end






	


	

	



