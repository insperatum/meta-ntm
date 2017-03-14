--[[

  Training a NTM to memorize input.

  The current version seems to work, giving good output after 5000 iterations
  or so. Proper initialization of the read/write weights seems to be crucial
  here.

--]]

require('../')
require('./util')
require('optim')
require('sys')

function maybeCuda(x)
  return x
  -- return x:cuda()
end

torch.manualSeed(0)

-- NTM config
local config = {
  input_dim = 10,
  output_dim = 10,
  mem_rows = 128,
  mem_cols = 20,
  cont_dim = 100,
  batch_size = 2
}

local input_dim = config.input_dim
local start_symbol = maybeCuda(torch.zeros(config.batch_size, input_dim))
start_symbol[{{}, 1}] = 1
local end_symbol = maybeCuda(torch.zeros(config.batch_size, input_dim))
end_symbol[{{}, 1}] = 1

local zeros = maybeCuda(torch.zeros(config.batch_size, input_dim))

function generate_sequence(len, batch_size, bits)
  local seq = maybeCuda(torch.zeros(len, batch_size, bits + 2))
  for i = 1, len do
    seq[{i, {}, {3, bits + 2}}] = torch.rand(batch_size, bits):round()
  end
  return seq
end

function forward(model, seq, print_flag)
  local len = seq:size(1)
  local loss = 0

  -- present start symbol
  model:forward(start_symbol)

  -- present inputs
  if print_flag then print('write head max') end
  for j = 1, len do
    model:forward(seq[j])
    if print_flag then print_write_max(model) end
  end

  -- present end symbol
  model:forward(end_symbol)

  -- present targets
  local outputs = maybeCuda(torch.Tensor(len, config.batch_size, input_dim))
  local criteria = {}
  if print_flag then print('read head max') end
  for j = 1, len do
    criteria[j] = maybeCuda(nn.MultiCriterion():add(nn.BCECriterion(), 1/len))
    outputs[j] = model:forward(zeros)
    loss = loss + criteria[j]:forward(outputs[j], seq[j])
    if print_flag then print_read_max(model) end
  end
  return outputs, criteria, loss
end

function backward(model, seq, outputs, criteria)
  local len = seq:size(1)
  for j = len, 1, -1 do
    model:backward(zeros, criteria[j]:backward(outputs[j], seq[j]))
  end

  model:backward(end_symbol, zeros)
  for j = len, 1, -1 do
    model:backward(seq[j], zeros)
  end
  model:backward(start_symbol, zeros)
end

local model = maybeCuda(ntm.NTM(config))
local params, grads = model:getParameters()

local num_iters = 10000
local start = sys.clock()
local print_interval = 5
local min_len = 1
local max_len = 20

print(string.rep('=', 80))
print("NTM copy task")
print('training up to ' .. num_iters .. ' iteration(s)')
print('min sequence length = ' .. min_len)
print('max sequence length = ' .. max_len)
print(string.rep('=', 80))
print('num params: ' .. params:size(1))

local adam_state = {learningRate = 0.01}

-- train
for iter = 1, num_iters do
  local print_flag = (iter % print_interval == 0)
  local feval = function(x)
    if print_flag then
      print(string.rep('-', 80))
      print('iter = ' .. iter)
      print('learn rate = ' .. adam_state.learningRate)
      printf('t = %.1fs\n', sys.clock() - start)
    end

    local loss = 0
    grads:zero()

    local len = math.floor(torch.random(min_len, max_len))
    local seq = generate_sequence(len, config.batch_size, input_dim - 2)
    local outputs, criteria, sample_loss = forward(model, seq, print_flag)
    loss = loss + sample_loss
    backward(model, seq, outputs, criteria)
    if print_flag then
      print("target:")
      print(seq[{{}, 1}])
      print("output:")
      print(outputs[{{}, 1}])
    end

    -- clip gradients
    grads:clamp(-10, 10)
    if print_flag then
      print('max grad = ' .. grads:max())
      print('min grad = ' .. grads:min())
      print('loss = ' .. loss)
    end
    return loss, grads
  end

  optim.adam(feval, params, adam_state)
end
