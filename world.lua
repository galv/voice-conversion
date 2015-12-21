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

void D4C(double *x, int x_length, int fs, double *time_axis, double *f0,
  int f0_length, int fft_size, D4COption *option, double **aperiodicity);
]]

local world = {}
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

   local fftTimeSize = f0:size(1)
   local fftFreqSize = world.C.GetFFTSizeForCheapTrick(fs)

   local spectrogram = torch.DoubleTensor(fftTimeSize, fftFreqSize / 2 + 1)
   assert(spectrogram:isContiguous())
   print(spectrogram:stride())
   world.C.FlatCheapTrick(x:data(), x:size(1), fs, timeAxis:data(), f0:data(),
                          fftTimeSize, option, spectrogram:data())
   return spectrogram
end

function world.D4C(x, fs, timeAxis, f0)

   local fftTimeSize = f0:size(1)

   local aperiodicity = torch.DoubleTensor(fftTimeSize)

   --world.C.D4C(x:data(), x:size(1), fs, timeAxis:data(), f0:data(),
   --            fftTimeSize, , nil, )

   return aperiodicity
end

-- For autoencoder: All speakers should have same representation for same sentence.
-- May not be allowed by constest rules.
-- kaldi-asr.org may have some pretrained neural net models.

function readTxt1Tensor(txt_file_name)
   local f = io.popen("wc -l " .. txt_file_name)
   local num_rows = f:read("*n")
   print("number of rows: " .. num_rows)
   local tensor = torch.DoubleTensor(num_rows)
   f:close()

   local i = 1
   for line in io.lines(txt_file_name) do
      tensor[i] = tonumber(line)
      i = i + 1
   end

   return tensor
end

function readTxt2Tensor(txt_file_name)
   -- TODO
end

tester = torch.Tester()

test = {}

function test.test()
   local x, fs = audio.load("code/WORLD/example/test16k.wav")
   assert(x:dim() == 2 and x:size(2) == 1)
   local x = x[{{},1}]
   local f0, timeAxis = world.dio(x, fs)

   do -- test dio output
      local trueF0 = readTxt1Tensor("code/WORLD/ground_truth/f0.txt")
      tester:assert(f0:dim() == trueF0:dim() and f0:dim() == 1 and
                       f0:size(1) == trueF0:size(1),
                    "Sizes: " .. f0:size(1) .. ", " .. trueF0:size(1))
      --gnuplot.plot(trueF0)
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
   gfx.image(spectrogram)

   local windowSizeMs = 5
   local windowSizeSec = windowSizeMs / 1000
   local windowSizeSamples = windowSizeSec * fs
   local smSpectrogram = audio.spectrogram(x, windowSizeSamples, 'hamming', windowSizeSamples / 16)
   gfx.image(smSpectrogram)
   return spectrogram, smSpectrogram
end

tester:add(test)
tester:run()

--spectrogram, smSpectrogram = test()
--print(spectrogram:size())
