local ffi = require("ffi")
local os = require("os")

require("audio")
require("gnuplot")
local gfx = require("gfx.js")
local paths = require("paths")

local world_dir = "code/WORLD/"

ffi.cdef [[
const char* getWORLDVersion(void);

typedef struct {
   double f0_floor;
   double f0_ceil;
   double channels_in_octave;
   double frame_period;  // msec
   int speed;  // (1, 2, ..., 12)
   double allowed_range;  // Threshold used for fixing the F0 contour.
} DioOption;


void Dio(double *x, int x_length, int fs, const DioOption option,
         double *time_axis, double *f0);
void InitializeDioOption(DioOption *option);
int GetSamplesForDIO(int fs, int x_length, double frame_period);

void StoneMask(double *x, int x_length, int fs, double *time_axis,
               double *f0, int f0_length, double *refined_f0);


typedef struct {
  double q1;
} CheapTrickOption;

void CheapTrick(double *x, int x_length, int fs, double *time_axis,
                double *f0, int f0_length, CheapTrickOption *option,
                double **spectrogram);

void FlatCheapTrick(double *x, int x_length, int fs, double *time_axis,
                    double *f0, int f0_length, CheapTrickOption *option,
                    double *spectrogram_t7_buffer);

void InitializeCheapTrickOption(CheapTrickOption *option);
int GetFFTSizeForCheapTrick(int fs);

typedef struct {
  double dummy;  // This is the future update.
} D4COption;

void FlatD4C(double *x, int x_length, int fs, double *time_axis, double *f0,
             int f0_length, int fft_freq_size, D4COption *option,
             double *aperiodicity_t7_buffer);
]]

world = {}

if paths.is_mac() then
   world.C = ffi.load(world_dir .. "build/src/libworld.dylib")
elseif paths.is_windows() then
   print("Guessing windows shared library location and name.")
   print("Not sure if forward slashes work.")
   world.C = ffi.load(world_dir .. "build/src/libworld.dll")
else
   world.C = ffi.load(world_dir .. "build/src/libworld.so")
end

function world.dio(x, fs)
   assert(x:dim() == 1, "Expecting monoaural data.")
   local x = x:contiguous()
   local option = ffi.new("DioOption[1]")
   world.C.InitializeDioOption(option)
   local option = option[0]
   local samples = world.C.GetSamplesForDIO(fs, x:size(1), option.frame_period)

   local timeAxis = torch.DoubleTensor(samples):contiguous()
   local f0 = torch.DoubleTensor(samples):contiguous()

   world.C.Dio(x:data(), x:size(1), fs, option, timeAxis:data(), f0:data())

   return f0, timeAxis
end

function world.stoneMask(x, fs, timeAxis, f0)
   local x = x:contiguous()
   local timeAxis = timeAxis:contiguous()
   local f0 = f0:contiguous()
   assert(timeAxis:size(1) == f0:size(1))

   local refinedF0 = torch.DoubleTensor(f0:size(1))
   assert(refinedF0:isContiguous())
   world.C.StoneMask(x:data(), x:size(1), fs, timeAxis:data(), f0:data(),
                     f0:size(1), refinedF0:data())
   return refinedF0
end

function world.CheapTrick(x, fs, timeAxis, f0)
   local option = ffi.new("CheapTrickOption[1]")
   world.C.InitializeCheapTrickOption(option)
   -- WARNING
   -- This value is set specifically to match q1's value in example/test.cpp
   -- I have no idea if it's any good or not.
   option[0].q1 = -.15

   local fftTimeSize = f0:size(1)
   local fftFreqSize = world.C.GetFFTSizeForCheapTrick(fs)

   local spectrogram = torch.DoubleTensor(fftTimeSize, math.floor(fftFreqSize / 2) + 1)
   assert(spectrogram:isContiguous())
   world.C.FlatCheapTrick(x:data(), x:size(1), fs, timeAxis:data(), f0:data(),
                          fftTimeSize, option, spectrogram:data())
   return spectrogram
end

function world.D4C(x, fs, timeAxis, f0)
   local fftTimeSize = f0:size(1)
   local fftSize = world.C.GetFFTSizeForCheapTrick(fs)
   local fftFreqSize = math.floor(fftSize / 2) + 1
   local aperiodicity = torch.DoubleTensor(fftTimeSize, fftFreqSize)
   assert(aperiodicity:isContiguous())

   world.C.FlatD4C(x:data(), x:size(1), fs, timeAxis:data(), f0:data(),
                   fftTimeSize, fftFreqSize, nil, aperiodicity:data())

   return aperiodicity
end

function world.doAll(x, fs)
   local f0, timeAxis = world.dio(x, fs)
   local f0 = world.stoneMask(x, fs, timeAxis, f0)
   local spectrogram = world.CheapTrick(x, fs, timeAxis, f0)
   local aperiodicity = world.D4C(x, fs, timeAxis, f0)

   return f0, timeAxis, spectrogram, aperiodicity
end

-- For autoencoder: All speakers should have same representation for same sentence.
-- May not be allowed by constest rules.
-- kaldi-asr.org may have some pretrained neural net models.

function readTxt1Tensor(txt_file_name)
   local f = io.popen("wc -l " .. txt_file_name)
   local numRows = f:read("*n")
   print("number of rows: " .. numRows)
   local tensor = torch.DoubleTensor(numRows)
   f:close()

   local i = 1
   for line in io.lines(txt_file_name) do
      tensor[i] = tonumber(line)
      i = i + 1
   end

   return tensor
end

function readTxt2Tensor(txt_file_name)
   local f = io.popen("wc -l " .. txt_file_name)
   local numRows = f:read("*n")
   print("number of rows: " .. numRows)
   f:close()

   local tensor = nil
   local numCols = 0

   local i = 1
   for line in io.lines(txt_file_name) do
      if i == 1 then
         for _ in line:gmatch("%S+") do
            numCols = numCols + 1
         end

         local tensor = torch.DoubleTensor(numRows, numCols)
      end

      local j = 1
      for num in line:gmatch("%S+") do
         local num = tonumber(num)
         tensor[{i,j}] = num
         j = j + 1
      end
      i = i + 1
   end

   return tensor
end

--tester = torch.Tester()

--test = {}

function test.test()
   -- WARNING: audio.load does not normalize x. Try to read
   -- WORLD code or Kaldi code to learn how to normalize
   -- a wave.
   local x, fs = audio.load("code/WORLD/example/test16k.wav")
   assert(x:dim() == 2 and x:size(2) == 1)
   local x = x[{{},1}]
   local f0, timeAxis = world.dio(x, fs)

   do -- test dio output
      local trueF0 = readTxt1Tensor("code/WORLD/ground_truth/f0.txt")
      tester:assert(f0:dim() == trueF0:dim() and f0:dim() == 1 and
                       f0:size(1) == trueF0:size(1),
                    "Sizes: " .. f0:size(1) .. ", " .. trueF0:size(1))
      tester:assert(torch.lt(torch.abs(f0 - trueF0), 1e-3):all())
   end

   local f0 = world.stoneMask(x, fs, timeAxis, f0)
   do
      local trueF0 = readTxt1Tensor("code/WORLD/ground_truth/f0_refined.txt")
      tester:assert(f0:dim() == trueF0:dim() and f0:dim() == 1 and
                       f0:size(1) == trueF0:size(1),
                    "Sizes: " .. f0:size(1) .. ", " .. trueF0:size(1))
      tester:assert(torch.lt(torch.abs(f0 - trueF0), 1e-3):all())
      gnuplot.plot(f0 - trueF0)
   end

   local spectrogram = world.CheapTrick(x, fs, timeAxis, f0)

   do
      local trueSpectrogram = readTxt2Tensor("code/WORLD/ground_truth/spectrogram.txt")
      tester:assert(spectrogram:dim() == trueSpectrogram:dim() and
                       spectrogram:size(1) == trueSpectrogram:size(1) and
                       spectrogram:size(2) == trueSpectrogram:size(2))
      local correct = torch.lt(torch.abs(spectrogram - trueSpectrogram), 1e-3):sum()

      gfx.image(torch.log(spectrogram))

      print("Correct / total = " .. correct .. "/" .. spectrogram:numel())
   end

   local aperiodicity = world.D4C(x, fs, timeAxis, f0)
   gfx.image(aperiodicity)

   --[[
   local windowSizeMs = 5
   local windowSizeSec = windowSizeMs / 1000
   local windowSizeSamples = windowSizeSec * fs
   local smSpectrogram = audio.spectrogram(x, windowSizeSamples, 'hamming', windowSizeSamples / 16)
   gfx.image(smSpectrogram)
   return spectrogram, smSpectrogram
   ]]--
end

--tester:add(test)
--tester:run()

--spectrogram, smSpectrogram = test()
--print(spectrogram:size())
