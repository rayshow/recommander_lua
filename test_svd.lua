
require 'torch'
require 'nn'
require 'optim'


local svd, parent = torch.class("nn.svd", "nn.Module")

function svd:__init( user_len, item_len, feature_len)
	parent.__init(self)
	self.weight0 = torch.Tensor( user_len, feature_len):randn(user_len, feature_len)
	self.gradWeight0 = torch.Tensor( user_len, feature_len)
	self.weight1 = torch.Tensor( feature_len, item_len ):randn(feature_len, item_len)
	self.gradWeight1 = torch.Tensor( feature_len, item_len)
	self.gradInput = torch.Tensor()
end

function svd:updateOutput( input )
	self.output:resize( self.weight0:size(1), self.weight1:size(2))
	self.output:mm( self.weight0, self.weight1 )
	--print("output", self.weight0, self.weight1,  self.output)
	return self.output
end

function svd:accGradParameters(input, gradOutput, scale)
	scale =  scale or 1
	self.gradWeight0:addmm( scale, gradOutput, self.weight1:t() )
	self.gradWeight1:addmm( scale, self.weight0:t(), gradOutput )
	return nil
end

function svd:parameters()
	return {self.weight0, self.weight1}, {self.gradWeight0, self.gradWeight1}
end

local nl_svd, _ = torch.class("nn.nl_svd", "nn.Module")
function nl_svd:__init( user_len, item_len, feature_len)
	parent.__init( self )
	self.weight0 = torch.Tensor( user_len, feature_len):randn(user_len, feature_len)
	self.gradWeight0 = torch.Tensor( user_len, feature_len)
	self.weight1 = torch.Tensor( feature_len, item_len ):randn(feature_len, item_len)
	self.gradWeight1 = torch.Tensor( feature_len, item_len)
	self.gradInput = torch.Tensor()
end


function nl_svd:updateOutput( input )
	self.output:resize( self.weight0:size(1), self.weight1:size(2))
	self.output:mm( self.weight0, self.weight1 )
	local pow2 = torch.cmul( self.output, self.output)
	self.output:add( pow2 )

	return self.output
end

function nl_svd:accGradParameters(input, gradOutput, scale)
	scale =  scale or 1
	self.gradWeight0:addmm( scale, gradOutput, self.weight1:t() )
	self.gradWeight1:addmm( scale, self.weight0:t(), gradOutput )
	return nil
end

function nl_svd:parameters()
	return {self.weight0, self.weight1}, {self.gradWeight0, self.gradWeight1}
end



local target = torch.Tensor(5,5):random(0,5)
print( "target",target)

local CONST_K = 2

local x = torch.Tensor(5,CONST_K):fill(0)
local y = torch.Tensor(CONST_K,5):fill(0)

local model = nn.svd(5,5,CONST_K) 
print( model.weight0, model.weight1)
local criterion = nn.MSECriterion()

local x, dx = model:getParameters()
dx:fill(0)
print( model.gradWeight0, model.gradWeight1)

local feval = function(x)
	dx:fill(0)
	
	local loss = criterion:forward( model:forward({}) , target )
	model:backward( {} , criterion:backward( model.output, target))
	
	--print( model:forward({}))
	return loss, dx
end

for i=1,5e4 do
	x, loss = optim.sgd(feval, x, { learningRate=1e-3, learningRateDecay=1e-4 })
	print("loss", loss[1])
end

print( target )
print( model:forward({}) )

