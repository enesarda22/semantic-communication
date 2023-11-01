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

```
python train_relay_decoder.py \
--n-samples 50000 \
--batch-size 64 \
--n-epochs 20 \
--lr 4e-5 \
--n-blocks 6 \
--n-heads 6
```


```
python train_receiver_decoder.py \
--relay-decoder-path checkpoints/**.pt \
--n-samples 50000 \
--batch-size 64 \
--n-epochs 20 \
--lr 4e-5 \
--n-blocks 6 \
--n-heads 6
```


## channel
n_latent = 128, 256
min_SNR = 3
max_SNR = 21
SNR_step = 3
SNR_window = 3
channel_type = AWGN

### training
n_samples = 50_000
lr = 4e-6
epoch = 25
batch_size = 64

```
python train_tx_relay_channel_block.py \
--n-samples 50000 \
--batch-size 64 \
--n-epochs 25 \
--lr 4e-6 \
--channel-block-latent-dim 128 \
--SNR-min 3 \
--SNR-max 21 \
--SNR-step 3 \
--SNR-window 3 \
--channel-type AWGN
```


## channel2
SNR_diff = 3

```
python train_tx_relay_rx_channel_block.py \
--relay-decoder-path checkpoints/**.pt \
--n-blocks 6 \
--n-heads 6 \
--n-samples 50000 \
--batch-size 64 \
--n-epochs 25 \
--lr 4e-6 \
--channel-block-latent-dim 128 \
--SNR-min 3 \
--SNR-max 21 \
--SNR-step 3 \
--SNR-window 3 \
--channel-type AWGN \
--SNR-diff 3
```

## end to end
lr = 1e-6
epoch = 15

```
python train_tx_relay_rx_channel_block.py \
--relay-decoder-path checkpoints/**.pt \
--receiver-decoder-path checkpoints/**.pt \
--n-blocks 6 \
--n-heads 6 \
--tx-relay-channel-model-path checkpoints/**.pt \
--tx-relay-rx-channel-model-path checkpoints/**.pt \
--n-samples 50000 \
--batch-size 64 \
--n-epochs 25 \
--lr 1e-6 \
--channel-block-latent-dim 128 \
--SNR-min 3 \
--SNR-max 21 \
--SNR-step 3 \
--SNR-window 3 \
--channel-type AWGN \
--SNR-diff 3
```
