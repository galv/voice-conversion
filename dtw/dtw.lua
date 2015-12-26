require "math"

function dtw(source, target, distanceFunc)
   assert(source:dim() == target:dim() and source:dim() == 2)
   assert(source:size(2) == target:size(2), "Feature dimension should be same for correct comparison.")
   -- distanceFunc is an optional argument. Default being torch.dist
   local distanceFunc = distanceFunc or torch.dist

   local N, M, F = source:size(1), target:size(1), source:size(2)

   -- construct cost matrix
   local function cost(i, j)
      return distanceFunc(sourceExpand[{i,{}}], targetExpand[{j,{}}])
   end

   -- accumulated cost matrix
   local accCostMat = torch.Tensor(N, M)
   accCostMat[{1,1}] = cost(1,1)

   for i=2,N do
      accCostMat[{i,1}] = cost(i,1) + accCostMat[{i - 1,1}]
   end
   for j=2,M do
      accCostMat[{1,j}] = cost(1,j) + accCostMat[{1,j - 1}]
   end

   for i = 2, N do
      for j = 2, M do
         accCostMat[{i,j}] = cost(i,j) + math.min(accCostMat[{i, j - 1}],
                                                  accCostMat[{i - 1, j}],
                                                  accCostMat[{i - 1, j - 1}])
      end
   end

   -- backtrack through accumulated cost matrix
   local path = {{N,M}}
   local i, j = N, M
   while i > 1 and j > 1 do
      if i == 2 then
         assert(j > 2)
         local j = j - 1
      elseif j == 2 then
         assert(i > 2)
         local i = i - 1
      else
         local minSetp = math.min(accCostMat[{i, j - 1}],
                                  accCostMat[{i - 1, j}],
                                  accCostMat[{i - 1, j - 1}])
         if accCostMat[{i, j - 1}] == minStep then
            local j = j - 1
         elseif accCostMat[{i - 1, j}] == minStep then
            local i = i - 1
         else
            local i = i - 1
            local j = j - 1
         end
      end
      table.insert(path, 1, {i, j})
   end
   table.insert(path, 1, {1, 1})
   return path
end

--[[ WARNING: I believe this should only be used for
   magnitude spectra, not log magnitude spectra, but
   am not positive.
   See this for details:
   Nonnegative Matrix Factorization with the
   Itakura-Saito Divergence: With Application to Music Analysis
   ]]
local function itakuraSaito(x, y)
   local ratio = torch.cdiv(x, y)
   return torch.sum(ratio - torch.log(ratio) - 1)
end
