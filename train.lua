require "nn"
require "rnn"

rnn = nn.Sequential()
   :add(nn.LSTM(freqDim, hiddenDim))
   :add(nn.LSTM(hiddenDim, hiddenDim))
   :add(nn.Linear(hiddenDim, freqDim))
)

rnn = nn.Sequencer(rnn)

criterion = nn.SequencerCriterion(nn.MSECriterion())

-- dataset: the hard part!
ds

xp = dp.Experiment{
   model = rnn,
   optimizer = train,
   validator = nil,
   tester = nil,
   observer = nil,
   max_epoch = 5
}

xp:run(ds)
