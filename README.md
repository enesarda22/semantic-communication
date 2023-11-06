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
--batch-size 512 \
--n-epochs 20 \
--lr 5e-4 \
--n-blocks 6 \
--n-heads 6 \
--data-fp /data \
--checkpoint-path /data/checkpoints
```


```
python train_receiver_decoder.py \
--relay-decoder-path /data/checkpoints/relay-decoder/relay_decoder_19.pt \
--batch-size 512 \
--n-epochs 20 \
--lr 5e-4 \
--n-blocks 6 \
--n-heads 6 \
--data-fp /data \
--checkpoint-path /data/checkpoints
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
--batch-size 512 \
--n-epochs 25 \
--lr 5e-5 \
--channel-block-latent-dim 128 \
--SNR-min 3 \
--SNR-max 21 \
--SNR-step 3 \
--SNR-window 3 \
--channel-type AWGN \
--data-fp /data \
--checkpoint-path /data/checkpoints
```


## channel2
SNR_diff = 3

```
python train_tx_relay_rx_channel_block.py \
--relay-decoder-path /data/checkpoints/relay-decoder/relay_decoder_19.pt \
--n-blocks 6 \
--n-heads 6 \
--batch-size 512 \
--n-epochs 25 \
--lr 5e-5 \
--channel-block-latent-dim 128 \
--SNR-min 3 \
--SNR-max 21 \
--SNR-step 3 \
--SNR-window 3 \
--channel-type AWGN \
--SNR-diff 3 \
--data-fp /data \
--checkpoint-path /data/checkpoints
```

## end to end
lr = 1e-6
epoch = 15

```
python train_tx_relay_rx_channel_block.py \
--relay-decoder-path /data/checkpoints/relay-decoder/relay_decoder_19.pt \
--receiver-decoder-path /data/checkpoints/receiver-decoder/receiver_decoder_19.pt \
--n-blocks 6 \
--n-heads 6 \
--tx-relay-channel-model-path checkpoints/**.pt \
--tx-relay-rx-channel-model-path checkpoints/**.pt \
--batch-size 512 \
--n-epochs 25 \
--lr 1e-7 \
--channel-block-latent-dim 128 \
--SNR-min 3 \
--SNR-max 21 \
--SNR-step 3 \
--SNR-window 3 \
--channel-type AWGN \
--SNR-diff 3 \
--data-fp /data \
--checkpoint-path /data/checkpoints
```

## Baseline source-relay
```
python train_source_relay.py \
--batch-size 512 \
--n-epochs 25 \
--lr 5e-5 \
--channel-block-latent-dim 128 \
--SNR-min 3 \
--SNR-max 21 \
--SNR-step 3 \
--SNR-window 3 \
--channel-type AWGN  \
--checkpoint-path /data/checkpoints
 ```

## Baseline entire network
```
python train_entire_network.py \
--batch-size 512 \
--n-epochs 25 \
--lr 5e-5 \
--channel-block-latent-dim 128 \
--SNR-min 3 \
--SNR-max 21 \
--SNR-step 3 \
--SNR-window 3 \
--channel-type AWGN  \
--checkpoint-path /data/checkpoints  \
--tx-relay-path /data/checkpoints/baseline-tx-relay/baseline_tx_relay_24.pt
```
