--[[
EDIT: ADDED BATCH DIMENSION
 Input: A table {x, y} of a Tensor x and a scalar y.

 Output: x * y

--]]

local ScalarMulTable, parent = torch.class('nn.ScalarMulTable', 'nn.Module')

function ScalarMulTable:__init()
  parent.__init(self)
  self.gradInput = {}
end

function ScalarMulTable:updateOutput(input)
  local v, scale = unpack(input)

  local size=v:size()
  for i=2, #size do size[i]=1 end
  
  return self.output:set(torch.cmul(v, scale:view(size):expandAs(v)))
end

function ScalarMulTable:updateGradInput(input, gradOutput)
  local v, scale = unpack(input)
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[2] = self.gradInput[2] or input[2].new()
  self.gradInput[2]:resizeAs(input[2])

  local size=v:size()
  for i=2, #size do size[i]=1 end
  local scale_expanded = scale:view(size):expandAs(v)

  self.gradInput[1]:set(torch.cmul(gradOutput, scale_expanded))

  local scalegrad = torch.cmul(gradOutput, v)
  while scalegrad:dim()>1 do
    scalegrad = scalegrad:sum(2)[{{}, 1}]
  end

  self.gradInput[2]:set(scalegrad)
  return self.gradInput
end
