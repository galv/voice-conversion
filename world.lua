local ffi = require("ffi")
local os = require("os")

require("audio")
require("gnuplot")
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
void InitializeCheapTrickOption(CheapTrickOption *option);
int GetFFTSizeForCheapTrick(int fs);

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

   local refinedF0 = torch.DoubleTensor(f0:size(1)):contiguous()
   print("Stone mask")

   world.C.StoneMask(x:data(), x:size(1), fs, timeAxis:data(), f0:data(),
                     f0:size(1), refinedF0:data())
   return refinedF0
end

function world.CheapTrick(x, fs, timeAxis, f0)
   local option = ffi.new("CheapTrickOption[1]")
   world.C.InitializeCheapTrickOption(option)

   print("q1: " .. option[0].q1)

   local fftTimeSize = f0:size(1)
   local fftFreqSize = world.C.GetFFTSizeForCheapTrick(fs)

   local spectrogram = torch.DoubleTensor(fftTimeSize, fftFreqSize):contiguous()
   local specArr = ffi.new("double*[?]", fftTimeSize)
   local specArrRows = {}
   for i=0,fftTimeSize do
      specArr[i] = ffi.new("double[?]", fftFreqSize)
      specArrRows[i] = specArr[i]
   end

   world.C.CheapTrick(x:data(), x:size(1), fs, timeAxis:data(), f0:data(),
                      fftTimeSize, option, specArr)

   print("Finished Cheap trick.")
   print("time: " .. fftTimeSize)
   print("freq: " .. fftFreqSize)
   print("q1: " .. option[0].q1)
   local spectrogram = spectrogram:contiguous()
   print(spectrogram:size())
   print(spectrogram:stride())
   -- Dirty hack.
   for i = 0, fftTimeSize - 1 do
      for j = 0, fftFreqSize - 1 do
         assert(i <  spectrogram:size(1) and j < spectrogram:size(2))
         spectrogram[i+1][j+1] = 42 --specArr[i][j]
      end
   end
   return spectogram
end

-- For autoencoder: All speakers should have same representation for same sentence.
-- May not be allowed by constest rules.
-- kaldi-asr.org may have some pretrained neural net models.

function main()
   local x, fs = audio.load("vcc2016_training/TF2/100100.wav")
   assert(x:dim() == 2 and x:size(2) == 1)
   local x = x[{{},1}]
   local f0, timeAxis = world.dio(x, fs)

   local f0 = world.stoneMask(x, fs, timeAxis, f0)
   local spectrogram = world.CheapTrick(x, fs, timeAxis, f0)

   gnuplot.plot(f0)
end

main()
