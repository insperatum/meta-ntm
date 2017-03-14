--[[
EDIT: ADDED BATCH DIMENSION
 Input: A table {x, y} of a Tensor x and a scalar y.

 Output: x^y (elementwise)

--]]

local PowTable, parent = torch.class('nn.PowTable', 'nn.Module')

function PowTable:__init()
  parent.__init(self)
  self.gradInput = {}
end

function PowTable:updateOutput(input)
  local v, p = unpack(input)

  local size=v:size()
  for i=2, #size do size[i]=1 end

  return self.output:set(torch.cpow(v, p:view(size):expandAs(v)))
end

function PowTable:updateGradInput(input, gradOutput)
  local v, p = unpack(input)
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[2] = self.gradInput[2] or input[2].new()
  self.gradInput[2]:resizeAs(input[2]):zero()

  local size=v:size()
  for i=2, #size do size[i]=1 end
  local p_expanded = p:view(size):expandAs(v)

  self.gradInput[1]:set(torch.cmul(gradOutput, torch.cmul(torch.cpow(v, p_expanded - 1), p_expanded)))

  local pgrad = torch.cmul(torch.cmul(torch.log(v), self.output), gradOutput)
  while pgrad:dim()>1 do
    pgrad = pgrad:sum(2)[{{}, 1}]
  end

  -- local pgrad = 0
  -- for i = 1, v:size(1) do
  --   if v[i] > 0 then
  --     pgrad = pgrad + math.log(v[i]) * self.output[i] * gradOutput[i]
  --   end
  -- end
  self.gradInput[2]:set(pgrad)
  return self.gradInput
end
