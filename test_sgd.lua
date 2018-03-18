require "nn"
require "optim"
require "csvigo"

torch.setdefaulttensortype('torch.FloatTensor')

function string:split(sep)
	local sep, fields = sep or " ",{}
	local pattern  = string.format("(.-)%s", sep) 
	local pattern2 = string.format(".+%s(.+)", sep)
	local _,n1 = self:gsub( pattern, function(c) fields[#fields+1] = c end)
	local _,n2 = self:gsub( pattern2, function(c) fields[#fields+1] = c end) 
	if n1 ==0 and n2 == 0 then
		fields[#fields+1] = self
	end

	return fields
end


function parse_line(line, format)
	local data = {};
	local fields = line:split("::");
	for i,v in ipairs(fields) do
		if format:sub(i,i) =='i' then
			data[i] = tonumber(v)
		else
			data[i] = v;
		end
	end
	return data
end

function parse_file(filename, format)
	local file = io.open(filename);
	local data = {}
	for line in file:lines() do
		data[#data+1] = parse_line( line, format)
	end
	print(" count:", #data)
	for i,v in ipairs(data[1]) do
		print( i, type(v), v)
	end
	io.close(file)
	return data
end

function item_cos_sim(i2u_mask, user_len, item_len)
	local i2i = torch.FloatTensor( item_len, item_len):fill(0)
	local sum = torch.FloatTensor( item_len):fill(0)
	i2i:addmm( i2u_mask, 2)
	sum:sum( i2u_mask, 2)
	local sum2 = sum * sum:t()
	i2i:cdiv(sum2)
	return i2i;
end

local CONST_DIM = 20
function svd(u2i_rating, user_len )
	local rus = torch.FloatTensor( user_len, CONST_DIM):fill(0)
	local cis = torch.FloatTensor( CONST_DIM, user_len):fill(0)
	local model = nn.Sequential()
	model:add(nn.MM())
	local criterion = nn.MSECriterion()

	local feval = function(x)
		local idx = (idx or 0) +1
		if idx > user_len then idx = 1 end

		local input  = ris[ idx ] 
		local target = u2i_rating[ idx ]
		local loss_x = criterion:forward(model:forward(input),target)
		--local model:backward(input, criterion:backward(model.output, target))
		return loss_x, dl_dx
	end
end

function data_to_tensor( data, dim1, dim2 )
	local tbl = torch.Tensor( dim1, dim2);
	print( dim1, dim2)
	for i=1,#data do
		local item = data[i]
		tbl[{ item[1], item[2] }] = item[3]
	end
	mask = tbl:gt(0)
	return tbl, mask
end

function rmse( pred, truth)
	local result = pred - truth;
	local mean = result:pow(2):mean()
	return math.sqrt( mean )
end

function split_dataset( u2i_tbl, user_len, item_len )
	local rands = torch.rand( user_len*item_len):resize(user_len, item_len)
	local ts_mask = rands:gt(0.2);	
	return ts_mask, 1 - ts_mask, ts_mask:typeAs(rands)
end

function split_u2i( u2i_tbl)
	local sizes = u2i_tbl:size()
	local rands = torch.rand( sizes[1] * sizes[2] ):resize( size[1], size[2] )
	local ts_mask = rands:gt(0.2)
	local ts_tbl = ts_mask:float():cmul( u2i_tbl )
	return ts_tbl, 1-ts_tbl
end


function user_cos_sim( u2i_mask, user_len, item_len )
    local u2u = torch.FloatTensor( user_len, user_len ):fill(0)
	local sum = torch.FloatTensor( user_len):fill(0)
	u2u:addmm(u2i_mask, u2i_mask:t())
	sum:sum( u2i_mask, 2)
	local sum2 = sum * sum:t()
	u2u:cdiv(sum2)
	return u2u
end


function v2str( vec, len )
	local str = "[ ";
	for i=1, len-1 do
		str = str..tostring( vec[i] ) .. ", "
	end
	str = str..tostring(vec[len]).."]"
	return str
end

function print_uim( u2m_tbl, i2m_tbl, mark_len, uid )
   local nbrk = true;
   local i = 1
   print("user and item marks vector:");
   while nbrk do
		local item = rating_data[i]
		if item ~=nil then
			i = i+1
			local user_id = item[1]
			local item_id = item[2]
			if user_id >uid then nbrk = false 
			elseif uid == user_id then
				print( i, v2str( i2m_tbl[item_id], mark_len ) )
			end
		end
   end
   print( 0, v2str( u2m_tbl[ uid ], mark_len )  )
end

function compute_user_marks( rating_data, item_data, user_len, item_len)
   -- 1. collect total marks of item
   local mark_tbl = {}
   local count = 1
   local marks = {}
   for i=1, #item_data do
	    local item_id = item_data[i][1]	
		marks[item_id] = item_data[i][3]:split('|')
		for _,mark in ipairs(marks[item_id]) do
			if mark_tbl[ mark ] == nil then
				mark_tbl[ mark ] = count
				count =  count + 1
			end
		end
   end

   for k,v in pairs(mark_tbl) do
      print( k, v)
   end
   
   local mark_len = count - 1

   -- 2. compute item's marks look table
   local i2m_tbl = torch.Tensor(item_len, mark_len):fill(0);
   for i=1, #item_data do
	   local item_id = item_data[i][1]
       
	   if marks[item_id] then 
		   for k,v in pairs(marks[item_id]) do
			   i2m_tbl[ item_id ][ mark_tbl[v] ] = 1
		   end
       end
   end

   local u2m_tbl = torch.Tensor( user_len, mark_len):fill(0)
   for k,v in pairs(rating_data) do
	    local user_id = v[1]
		local item_id = v[2]
    	u2m_tbl[ user_id ]:add( i2m_tbl[ item_id ] )
   end
   print_uim( u2m_tbl, i2m_tbl, mark_len, 1)
   return u2m_tbl, i2m_tbl, mark_tbl
end

function user_cf(set, user_len, item_len, topk, rmdk) 	
	print("spliting dataset")

	ts_mask, dts_mask, ts_set = split_dataset( set, user_len, item_len )
	print("end spliting")
	
	print(" compute similarity")
	u2u_sim = user_cos_sim( ts_set, user_len, item_len)
	u2u_sim:maskedFill( torch.eye(user_len, user_len):byte(), 0 )

	print(" compute top10 user")
	_, topk_uids = u2u_sim:topk(topk, 2, true)

	print(" compute top1 item")
	
	u2i_recommand = torch.Tensor(user_len, item_len):fill(0)
	for i = 1, user_len do
		u2i_recommand[i]:add(ts_set:index(1, topk_uids[i]):sum(1))
	end
    u2i_recommand:maskedFill(ts_mask, 0)
	_, topk_iids  = u2i_recommand:topk(rmdk, 2, true)
	
	print(" compute success count")
	local success_count = 0
	for i=1, user_len do
		local accept = false
		for j=1, rmdk do
			if set[ i ][ topk_iids[i][j] ]>0 then
				accept = true
			end
		end
		if accept then
			success_count = success_count + 1
		end
	end
	print("precentage: ", success_count / user_len)
end

function item_cf(u2i_tbl, item_set, user_len, item_len)
	print(" compute user marks");

	u2m_tbl, i2m_tbl, marks = compute_user_marks( 
		rating_data, movie_data, user_len, item_len )
    	
	print("spliting dataset")
	
	ts_tbl, rs_tbl = split_dataset( u2i_tbl, user_len, item_len )
    
	print(" compute item heat")
	local item_heat = u2i_tbl:clone():sum(1)
	local weigth_heat = torch.Tensor( item_len, 2)
	weigth_heat[{},1] = item_heat
	
	print(" compute user favor");
	local Favor_TopK = 5
	favor_lvs, favor_mids = u2m_tbl:topk(Favor_TopK, 2, true )
    
	local success_count = 0
	local RMD_Count = 10
	for i=1,user_len do
		local user_faver_mids = favor_mids[i]	
		local item_wight = i2m_tbl[{ {}, user_favor_mids }]:sum()
		print(item_wight:size())
		weight_heat[{},1] = item_wight
		local _, rmd_ids = weight_heat:prod(1):topk(RMD_Count,2,true)
		local j = 1
		local find = false
		while j< RMD_Count and not find do
			if u2i_tbl[i][ rmd_ids[j] ] !=0 then
				success_count = success_count + 1
				find = true
			end
		end
	end
	print("precentage: ", success_count / user_len)

end

user_data = parse_file("ml-1m/users.dat", "isiis");
movie_data = parse_file("ml-1m/movies.dat", "iss");
local max_movie_id = 0;
for i=1, #movie_data do
	if movie_data[i][1] > max_movie_id then
		max_movie_id = movie_data[i][1]
	end
end
print("max movie id:", max_movie_id)
rating_data = parse_file("ml-1m/ratings.dat", "iiis");
local user_len = #user_data;
local item_len = max_movie_id;


--local x = torch.rand(user_len*item_len):resize(user_len,item_len):gt(0.90):float()
--user_cf( x, user_len, item_len, 10,)
--for i=0,user_len do
--	if u2i_mask[1][i]
--	print( u2i_mask[1][i] )
--user_cf( u2i_mask, user_len, item_len, 1, 10)


