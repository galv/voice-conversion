require 'paths'
require 'audio'
local _ = require 'moses'
local trainPath = "vcc2016_training"

require 'code.dtw.dtw'
require 'code.world'

function table.ofIterator(...)
   local arr = {}
   for v in ... do
      arr[#arr + 1] = v
   end
   return arr
end

function getTrainSamplesForSpeaker(trainPath, speaker)
   local dir = paths.concat(trainPath, speaker)
   local files = table.ofIterator(paths.iterfiles(dir))
   local files = _.map(files,
                       function(k, fileName) return paths.concat(dir, fileName)
                       end)
   return files
end

function mungeToFeatures(wavFileName)
   local x, fs = audio.load(wavFileName)
   local x = x[{{},1}]
   return table.pack(world.doAll(x, fs))
end

stage = 2

function main()

   local speakers = paths.dir(trainPath)
   local trainFeatPath = trainPath .. "_feat"
   paths.mkdir(trainFeatPath)
   local sourceSpeakers =
   _.filter(speakers,function(k, dirName) return dirName:find("S") == 1 end)
   local targetSpeakers =
   _.filter(speakers,function(k, dirName) return dirName:find("T") == 1 end)


   if stage <= 1 then

   for __, speaker in ipairs(_.union(sourceSpeakers, targetSpeakers)) do
      paths.mkdir(paths.concat(trainFeatPath, speaker))
   end

   for __, speaker in ipairs(_.union(sourceSpeakers, targetSpeakers)) do
      local trainSamples = getTrainSamplesForSpeaker(trainPath, speaker)
      local sampleFeatures = _.map(trainSamples,
                             function(__, fileName)
                                return mungeToFeatures(fileName) end)
      _.eachi(sampleFeatures,
              function(i, sampleFeature)
                 local fileName =
                    string.gsub(
                       string.gsub(trainSamples[i], trainPath, trainFeatPath),
                       ".wav",".feat")
                 -- TODO: Perhaps change from double to float?
                 -- GPU uses float anyway, right?
                 local floatSampleFeature =
                    _.map(sampleFeature, function(__, tensor)
                             return tensor:float()
                    end)
                 torch.save(fileName, floatSampleFeature)
      end)
   end

   end

   do return end

   if stage <= 2 then

   local trainFeatPairPath = trainFeatPath .. "_pairs"
   paths.mkdir(trainFeatPairPath)

   for __, sourceSpeaker in ipairs(sourceSpeakers) do
      for __, targetSpeaker in ipairs(targetSpeakers) do
         local sourceFiles = getTrainSamplesForSpeaker(trainFeatPath,
                                                       sourceSpeaker)
         local targetFiles = getTrainSamplesForSpeaker(trainFeatPath,
                                                       targetSpeaker)
         local trainPairs = _.zip(sourceFiles, targetFiles)

         -- load their features
         -- get spectrogram
         -- do forced alignment
         -- Save:
         -- (1) forced alignment table
         -- (2) path to source spectrogram
         -- (3) path to destination spectrogram

         local pairPath = paths.concat(trainFeatPairPath,
                                       sourceSpeaker .. "_" .. targetSpeaker)
         paths.mkdir(pairPath)

         for uttNum,featFilePair in ipairs(trainPairs) do
            local sourceData = torch.load(featFilePair[1])
            local targetData = torch.load(featFilePair[2])
            local path = dtw(sourceData[3], targetData[3])

            -- WARNING: This is NOT safe. It is mutating featFilePair, which
            -- makes it a triple not a pair.
            -- Unfortunately, I don't know how to make a copy of a table in lua
            -- though. Ugh....
            table.insert(featFilePair, path)

            -- HACK to get the utterance number, until I learn lua pattern matching.
            local uttNum = uttNum + 100000

            local savePath = paths.concat(pairPath, uttNum .. ".feat")
            print(savePath)
            torch.save(savePath, featFilePair)
         end

      end
   end

   end
end

function doubleToFloat()
   local fileNames = io.popen("ls -1 vcc2016_training_feat/*/*.feat")

   for fileName in fileNames:lines() do
      local featTable = torch.load(fileName)
      assert(featTable[1]:type() == "torch.DoubleTensor")
      local floatFeatTable =
         _.map(featTable, function(__, tensor)
                  if torch.type(tensor) == "torch.DoubleTensor" then
                     return tensor:float()
                  else
                     return tensor
                  end
         end)

      assert(floatFeatTable[1]:type() == "torch.FloatTensor")
      print(fileName)
      torch.save(fileName, floatFeatTable)
   end
end

doubleToFloat()
