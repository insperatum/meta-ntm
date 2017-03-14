--[[
EDIT: ADDED BATCH DIMENSION
Input: a table of two inputs {M, k}, where
  M = an b-by-n-by-m matrix
  k = an b-by-m-dimensional vector

Output: an b-by-n-dimensional vector

Each element is an approximation of the cosine similarity between k and the 
corresponding row of M. It's an approximation since we add a constant to the
denominator of the cosine similarity function to remove the singularity when
one of the inputs is zero. 

--]]

local SmoothCosineSimilarity, parent = torch.class('nn.SmoothCosineSimilarity', 'nn.Module')

function SmoothCosineSimilarity:__init(smoothen)
  parent.__init(self)
  self.gradInput = {}
  self.smooth = smoothen or 1e-3
end

function SmoothCosineSimilarity:updateOutput(input)
  local M, k = unpack(input) -- M: b * n * m,  K: b * m 
  self.rownorms = torch.cmul(M, M):sum(3):sqrt():view(M:size(1), M:size(2)) -- b * n
  self.knorm = torch.cmul(k,k):sum(2):sqrt() -- b * 1
  self.dot = k:view(M:size(1), 1, k:size(2)):expandAs(M):cmul(M):sum(3) -- b * n
  self.output:set(torch.cdiv(self.dot, torch.cmul(self.rownorms, self.knorm:expandAs(self.rownorms)):add(self.smooth))) -- b * n
  return self.output
end

function SmoothCosineSimilarity:updateGradInput(input, gradOutput)
  local M, k = unpack(input)
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[2] = self.gradInput[2] or input[2].new()


  -- M gradient
  local rows = M:size(2)
  local Mgrad = self.gradInput[1]
  Mgrad:set(k:repeatTensor(1, rows):view(k:size(1), rows, k:size(2))) -- 
  for i = 1, rows do
    local idxs = torch.nonzero(self.rownorms[{{}, i}]:gt(0))[{{}, 1}]
    local foo = -torch.cdiv(torch.cmul(self.output[{{}, i}], self.knorm), self.rownorms[{{}, i}])
    Mgrad[{{}, i}]:indexAdd(1, idxs, foo:index(1,idxs):expand(idxs:size(1), Mgrad:size(3)))
    Mgrad[{{}, i}]:indexAdd(1, idxs, M[{{}, i}]:index(1,idxs))

    
    Mgrad[{{}, i}]:cmul(gradOutput[{{}, i}]:expandAs(Mgrad[{{}, i}]))
    local foo = torch.cmul(self.rownorms[{{}, i}], self.knorm)
    Mgrad[{{}, i}]:cdiv(foo:view(Mgrad:size(1), 1):expandAs(Mgrad[{{}, i}])+self.smooth)
  end

  -- k gradient
  self.gradInput[2]:set(torch.bmm(M:transpose(2,3), torch.cdiv(gradOutput, torch.cmul(self.rownorms, self.knorm:expandAs(self.rownorms)) + self.smooth))[{{}, {}, 1}])

  local idxs = torch.nonzero(self.knorm:gt(0))[{{}, 1}]

  local scale = torch.cmul(self.output, self.rownorms)
    :cdiv(torch.cmul(self.rownorms, self.knorm:expandAs(self.rownorms)) + self.smooth)
    :cmul(gradOutput):sum(2)
    :cdiv(self.knorm)[{{}, 1}]

  self.gradInput[2]:indexAdd(1, idxs, -scale:expandAs(self.gradInput[2]):index(1,idxs))
  self.gradInput[2]:indexAdd(1, idxs, k:index(1,idxs))

  return self.gradInput
end
