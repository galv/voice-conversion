require 'paths'
require 'nn'
local _ = require 'moses'
local trainPath = "/Users/danielgalvez/Desktop/Fall2015/research/vcc2016/vcc2016_training/"

function table.ofIterator(...)
   local arr = {}
   for v in ... do
      arr[#arr + 1] = v
   end
   return arr
end

function main()
   local speakers = paths.dir(trainPath)
   local sourceSpeakers =
      _.filter(speakers,function(k, dirName) return dirName:find("S") == 1 end)
   local targetSpeakers =
      _.filter(speakers,function(k, dirName) return dirName:find("T") == 1 end)

   for i, sourceSpeaker in ipairs(sourceSpeakers) do
      for j, targetSpeaker in ipairs(targetSpeakers) do
         local sourceFiles = paths.files(trainPath .. "/" .. sourceSpeaker)
         local targetFiles = paths.files(trainPath .. "/" .. targetSpeaker)
         local trainPairs = _.zip(sourceFiles, targetFiles)
      end
   end
end

function getTrainingPairsForPair(sourceSpeaker, targetSpeaker)
   local sourceDir = trainPath .. "/" .. sourceSpeaker .. "/"
   local targetDir = trainPath .. "/" .. targetSpeaker .. "/"
   local sourceFiles = table.ofIterator(paths.files(sourceDir))
   local sourceFiles = _.map(sourceFiles,
                             function(k, fileName) return sourceDir .. fileName
                             end)
   local targetFiles = table.ofIterator(paths.files(targetDir))
   local targetFiles = _.map(targetFiles,
                             function(k, fileName) return targetDir .. fileName
                             end)
   local trainPairs = _.zip(sourceFiles, targetFiles)
   return trainPairs
end


dataPairs = getTrainingPairsForPair("SF1", "TM1")