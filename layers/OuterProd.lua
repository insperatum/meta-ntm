--[[
EDIT: ADDED BATCH DIMENSION
  Input: a table of 2 or 3 vectors.

  Output: the outer product of the vectors.

--]]

local OuterProd, parent = torch.class('nn.OuterProd', 'nn.Module')

function OuterProd:__init()
  parent.__init(self)
  self.gradInput = {}
end

function OuterProd:updateOutput(input)
  local order = #input
  self.order = order
  local n = input[1]:size(1)
  local m = {}
  for i=1,order do m[i]=input[i]:size(2) end

  if order == 2 then
    self.output:set(torch.cmul(
      input[1]:view(n, m[1], 1):expand(n, m[1], m[2]), 
      input[2]:view(n, 1, m[2]):expand(n, m[1], m[2])))
    self.size = self.output:size()
  elseif order == 3 then
    self.output:set(
      torch.cmul(torch.cmul(
        input[1]:view(n, m[1], 1, 1):expand(n, m[1], m[2], m[3]), 
        input[2]:view(n, 1, m[2], 1):expand(n, m[1], m[2], m[3])),
        input[3]:view(n, 1, 1, m[3]):expand(n, m[1], m[2], m[3])))        

    self.size = self.output:size()
  else
    error('outer products of order higher than 3 unsupported')
  end
  return self.output
end

function OuterProd:updateGradInput(input, gradOutput)
  local order = #input
  for i = 1, order do
    self.gradInput[i] = self.gradInput[i] or input[1].new()
    self.gradInput[i]:resizeAs(input[i])
  end

  local n = input[1]:size(1)
  local m = {}
  for i=1,order do m[i]=input[i]:size(2) end

  if order == 2 then
    self.gradInput[1]:copy(torch.cmul(
      gradOutput,
      input[2]:view(n, 1, m[2]):expand(n, m[1], m[2])
    ):sum(3))
    self.gradInput[2]:copy(torch.cmul(
      gradOutput,
      input[1]:view(n, m[1], 1):expand(n, m[1], m[2])
    ):sum(2))
  else
    self.gradInput[1]:copy(torch.cmul(torch.cmul(
      gradOutput,
      input[2]:view(n, 1, m[2], 1):expand(n, m[1], m[2], m[3])),
      input[3]:view(n, 1, 1, m[3]):expand(n, m[1], m[2], m[3])
    ):sum(4):sum(3))
    self.gradInput[2]:copy(torch.cmul(torch.cmul(
      gradOutput,
      input[1]:view(n, m[1], 1, 1):expand(n, m[1], m[2], m[3])),
      input[3]:view(n, 1, 1, m[3]):expand(n, m[1], m[2], m[3])
    ):sum(4):sum(2))
    self.gradInput[3]:copy(torch.cmul(torch.cmul(
      gradOutput,
      input[1]:view(n, m[1], 1, 1):expand(n, m[1], m[2], m[3])),
      input[2]:view(n, 1, m[2], 1):expand(n, m[1], m[2], m[3])
    ):sum(3):sum(2))
  end

  return self.gradInput
end
