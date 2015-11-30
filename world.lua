local ffi = require("ffi")
local os = require("os")

require("audio")
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
]]

if paths.is_mac() then
   world = ffi.load(world_dir .. "build/src/libworld.dylib")
elseif paths.is_windows() then
   print("Guessing windows shared library location and name.")
   print("Not sure if forward slashes work.")
   world = ffi.load(world_dir .. "build/src/libworld.dll")
else
   world = ffi.load(world_dir .. "build/src/libworld.so")
end

function dio(x, fs)
   assert(x:dim() == 1, "Expecting monoaural data.")
   local x = x:contiguous()
   local option = ffi.new("DioOption[1]")
   world.InitializeDioOption(option)
   local option = option[0]
   local samples = world.GetSamplesForDIO(fs, x:size(1), option.frame_period)

   local time_axis = torch.DoubleTensor(samples):contiguous()
   local f0 = torch.DoubleTensor(samples):contiguous()

   world.Dio(x:data(), x:size(1), fs, option, time_axis:data(), f0:data())

   return dio, time_axis
end

-- For autoencoder: All speakers should have same representation for same sentence.
-- May not be allowed by constest rules.
-- kaldi-asr.org may have some pretrained neural net models.

local x, fs = audio.load("vcc2016_training/TF2/100100.wav")

main()