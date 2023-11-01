# semantic-communication

## semantic decoder

multi head = 64 * 64 * 3 * 6 = 73_728
feed forward = 2 * 384 * 4 * 384 = 104_448
x6 = 1_069_056

linear = 30522 * 384 = 11_720_448

n_heads = 6
n_decoder_block = 6

### training
n_samples = 50_000
lr =  4e-5
epoch = 20
batch_size = 64

## channel
n_latent = 128, 256
min_SNR = 3
max_SNR = 21
SNR_step = 3
SNR_window = 3

### training
n_samples = 50_000
lr = 4e-6
epoch = 25
batch_size = 64

## channel2
SNR_diff = 3

### receiver_decoder
same as relay decoder

## end to end
lr = 1e-6
epoch = 15
