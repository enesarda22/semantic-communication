from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, List, Any, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------
# Modulation selection helper
# ----------------------------

def choose_modulation_from_symbols_per_token(vocab_size: int, symbols_per_token: int) -> Tuple[int, int, int, int]:
    if symbols_per_token <= 0:
        raise ValueError("symbols_per_token must be positive")
    V = int(vocab_size)
    nbits = int(math.ceil(math.log2(V)))
    # required bits per symbol
    k_raw = int(math.ceil(nbits / float(symbols_per_token)))
    # Restrict to QPSK/square QAM => k must be even and >=2
    k = max(2, k_raw)
    if k % 2 != 0:
        k += 1
    M = 2 ** k
    L_actual = int(math.ceil(nbits / float(k)))
    return M, nbits, k, L_actual


# ----------------------------
# Token -> constellation (high-dimensional mapping)
# ----------------------------

def _gray(n: int) -> int:
    return n ^ (n >> 1)


def _qam_points_square(M: int, device=None) -> torch.Tensor:
    """Square M-QAM (Gray per axis), normalized to unit average power. M must be 2^(even)."""
    k = int(round(math.log2(M)))
    if 2 ** k != M:
        raise ValueError(f"M must be a power of 2, got {M}")
    if k % 2 != 0:
        raise ValueError(f"Square QAM requires even log2(M); got M={M} (k={k})")

    m = 2 ** (k // 2)
    levels = torch.arange(m, device=device, dtype=torch.float32)
    levels = 2 * levels - (m - 1)

    pts = torch.empty((M,), device=device, dtype=torch.complex64)
    for sym in range(M):
        i_bits = sym >> (k // 2)
        q_bits = sym & ((1 << (k // 2)) - 1)
        i = _gray(i_bits)
        q = _gray(q_bits)
        pts[sym] = torch.complex(levels[i], levels[q])

    avg_pow = (pts.real ** 2 + pts.imag ** 2).mean()
    return pts / torch.sqrt(avg_pow)


def _qpsk_points(device=None) -> torch.Tensor:
    pts = torch.tensor([1+1j, -1+1j, 1-1j, -1-1j], device=device, dtype=torch.complex64)
    avg_pow = (pts.real ** 2 + pts.imag ** 2).mean()
    return pts / torch.sqrt(avg_pow)


def _bits_to_int(bits_2d: torch.Tensor) -> torch.Tensor:
    k = bits_2d.shape[-1]
    weights = (2 ** torch.arange(k - 1, -1, -1, device=bits_2d.device)).view(1, k)
    return (bits_2d.to(torch.int64) * weights).sum(dim=-1)


def _sanitize_cand_ids(cand_ids: torch.Tensor, Vuse: int, fallback_id: int) -> torch.Tensor:
    """
    Ensure token ids are valid for indexing logp[:Vuse] and constellation tables.
    Always returns a non-empty 1D Long tensor on the same device.
    """
    cand_ids = cand_ids.to(dtype=torch.long).view(-1)
    cand_ids = cand_ids[(cand_ids >= 0) & (cand_ids < Vuse)]
    if cand_ids.numel() == 0:
        return torch.tensor([fallback_id], device=cand_ids.device, dtype=torch.long)
    return torch.unique(cand_ids)


class TokenConstellation(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        M: int,
        device: torch.device,
        signal_power_constraint: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.M = int(M)
        self.device = device

        self.nbits = int(math.ceil(math.log2(self.vocab_size)))
        self.k = int(round(math.log2(self.M)))
        if 2 ** self.k != self.M:
            raise ValueError(f"M must be a power of 2, got {M}")

        self.nbits_padded = int(math.ceil(self.nbits / self.k) * self.k)
        self.L = self.nbits_padded // self.k  # complex symbols per token

        pts = _qpsk_points(device=self.device) if self.M == 4 else _qam_points_square(self.M, device=self.device)

        # 1) Build raw complex constellation rows per token id: const_raw[tid, l]
        const_raw = torch.empty((self.vocab_size, self.L), device=self.device, dtype=torch.complex64)
        for tid in range(self.vocab_size):
            bits = [(tid >> (self.nbits - 1 - i)) & 1 for i in range(self.nbits)]
            b = torch.tensor(bits, device=self.device, dtype=torch.int64)
            if self.nbits_padded > self.nbits:
                b = torch.cat(
                    [b, torch.zeros(self.nbits_padded - self.nbits, device=self.device, dtype=torch.int64)],
                    dim=0,
                )
            b = b.view(self.L, self.k)
            sym_idx = _bits_to_int(b)
            const_raw[tid] = pts[sym_idx]

        # 2) Convert to real stacked [V, 2L]
        const_real_raw = torch.cat([const_raw.real, const_raw.imag], dim=-1).to(torch.float32)

        # 3) Match Channel.signal_process normalization:
        #    scale = sqrt(P * last_dim/2) = sqrt(P * (2L)/2) = sqrt(P * L)
        P = float(signal_power_constraint)
        scale = math.sqrt(P * self.L)
        const_real_tx = scale * F.normalize(const_real_raw, dim=-1, p=2)

        const_tx_complex = torch.complex(const_real_tx[:, : self.L], const_real_tx[:, self.L :])

        # 4) Store normalized versions (ONLY ONCE)
        self.register_buffer("const_real", const_real_tx)                  # [V, 2L] float
        self.register_buffer("const_complex", const_tx_complex.to(torch.complex64))  # [V, L] complex

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.long()
        flat = token_ids.view(-1)
        # ensure const_real is on the same device as token_ids
        const_real = self.const_real.to(token_ids.device)
        return const_real.index_select(0, flat).view(*token_ids.shape, -1)

    def complex_rows(self, token_ids_1d: torch.Tensor) -> torch.Tensor:
        token_ids_1d = token_ids_1d.long()
        const_complex = self.const_complex.to(token_ids_1d.device)
        return const_complex.index_select(0, token_ids_1d)


# ----------------------------
# Channel adaptor + N0 computation
# ----------------------------

def _channel_call(channel, x: torch.Tensor, d: Any):
    if channel.type == "Rayleigh":
        out = channel(x, d, return_h=True)
    else:
        out = channel(x, d)

    if isinstance(out, tuple):
        return out[0], out[1]
    return out, None


def _to_complex(y_real: torch.Tensor) -> torch.Tensor:
    L2 = y_real.shape[-1]
    if L2 % 2 != 0:
        raise ValueError("Expected last dimension 2L (Re||Im).")
    L = L2 // 2
    return torch.complex(y_real[..., :L], y_real[..., L:])


def _h_to_complex(h: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    shape = like.shape[:-1]
    dev = like.device
    if h is None:
        return torch.ones(shape, device=dev, dtype=torch.complex64)
    if torch.is_complex(h):
        return h.to(device=dev, dtype=torch.complex64)
    h = h.to(device=dev)
    if h.dim() > 0 and h.shape[-1] == 2:
        return torch.complex(h[..., 0], h[..., 1]).to(torch.complex64)
    return torch.complex(h, torch.zeros_like(h)).to(torch.complex64)


def _compute_N0(channel, d: Optional[float]) -> Optional[float]:
    """
    For your Channel:
      std_real = sqrt(noise_pow * d^alpha)  => sigma^2 = noise_pow * d^alpha
      N0 = sigma^2 = noise_pow*d^alpha
    """
    if channel is None or d is None:
        raise Exception("Channel or d is none.")
    if not hasattr(channel, "noise_pow") or not hasattr(channel, "alpha"):
        raise Exception("Channel properties are not defined.")
    return float(channel.noise_pow * (float(d) ** float(channel.alpha)))


# ----------------------------
# Beam decoding
# ----------------------------

@dataclass
class _Beam:
    ids: torch.Tensor
    score: float
    ended: bool


def _dist_sq_candidates(
    y_chunk: torch.Tensor,          # [L] complex
    h: torch.Tensor,                # scalar complex
    constellation: TokenConstellation,
    cand_ids: torch.Tensor,         # [K]
) -> torch.Tensor:
    s = constellation.complex_rows(cand_ids)  # [K,L]
    diff = y_chunk.unsqueeze(0) - h.view(1, 1) * s
    return (diff.real ** 2 + diff.imag ** 2).sum(dim=-1)


def _int_to_bits(x: torch.Tensor, nbits: int) -> torch.Tensor:
    # x: [K] int64 -> bits [K, nbits] (MSB first)
    shifts = torch.arange(nbits - 1, -1, -1, device=x.device, dtype=torch.int64)
    return ((x.unsqueeze(-1) >> shifts) & 1).to(torch.int64)

def _bits_to_int_1d(bits: torch.Tensor) -> int:
    # bits: [nbits], MSB first
    val = 0
    for b in bits.tolist():
        val = (val << 1) | int(b)
    return val

@torch.no_grad()
def channel_token_candidates_from_y(
    y_chunk_complex: torch.Tensor,   # [L] complex
    h_scalar: torch.Tensor,          # scalar complex
    *,
    M: int,
    k: int,
    L: int,
    nbits: int,
    nbits_padded: int,
    vocab_size: int,  # <-- ADD THIS
    max_token_cands: int = 32,       # keep small
    sym_topm: int = 2,               # per-symbol branching (2 usually enough)
):
    """
    Make token candidates from channel only:
      - equalize y/h
      - per symbol choose top sym_topm closest constellation points
      - do small beam over L symbols
      - map symbol indices -> bits -> token id
    """
    device = y_chunk_complex.device

    # base constellation
    pts = _qpsk_points(device=device) if M == 4 else _qam_points_square(M, device=device)

    # equalize
    if h_scalar is None:
        h_scalar = torch.tensor(1.0 + 0.0j, device=device, dtype=torch.complex64)
    r = y_chunk_complex / h_scalar  # [L] complex

    # distances to constellation points: [L,M]
    diff = r.unsqueeze(-1) - pts.view(1, -1)
    dist_sym = (diff.real**2 + diff.imag**2)

    topm = min(sym_topm, M)
    sym_dist, sym_idx = torch.topk(dist_sym, k=topm, dim=-1, largest=False)  # [L,topm]

    # symbol-beam search
    beams = [(torch.empty((0,), device=device, dtype=torch.long), 0.0)]
    for t in range(L):
        new_beams = []
        for seq, sc in beams:
            for j in range(topm):
                new_seq = torch.cat([seq, sym_idx[t, j].view(1)], dim=0)
                new_sc = sc + float(sym_dist[t, j].item())
                new_beams.append((new_seq, new_sc))
        new_beams.sort(key=lambda z: z[1])
        beams = new_beams[:max_token_cands]

    # convert each symbol-seq -> token id
    token_ids = []
    for seq, _ in beams:
        bits_sym = _int_to_bits(seq.to(torch.int64), k)      # [L,k]
        bits_flat = bits_sym.reshape(-1)                     # [L*k] = nbits_padded
        bits_used = bits_flat[:nbits]                        # drop padding bits
        token_ids.append(_bits_to_int_1d(bits_used))

    token_ids = torch.tensor(token_ids, device=device, dtype=torch.long)
    token_ids = torch.unique(token_ids)

    # *** CRITICAL: clip to tokenizer vocab size ***
    token_ids = token_ids[(token_ids >= 0) & (token_ids < vocab_size)]

    token_ids = token_ids[:max_token_cands]
    return token_ids


@torch.no_grad()
def llmsc_beam_decode_one_link(
    model,
    tokenizer,
    constellation: TokenConstellation,
    *,
    received_real: torch.Tensor,       # [T,2L]
    h_tokens: Optional[torch.Tensor],  # [T] or None
    n0: float,
    beam_width: int,
    candidate_topk: int,
) -> torch.Tensor:
    device = received_real.device
    T = received_real.shape[0]
    y = _to_complex(received_real)               # [T,L]
    h = _h_to_complex(h_tokens, received_real)   # [T]

    V = constellation.vocab_size
    eos = tokenizer.eos_token_id
    bos = tokenizer.bos_token_id

    if eos is None:
        raise ValueError("Tokenizer has no eos_token_id; fixed-length decoding needs EOS.")

    start = torch.tensor([bos], device=device, dtype=torch.long) if bos is not None else torch.empty((0,), device=device, dtype=torch.long)
    beams: List[_Beam] = [_Beam(start, 0.0, False)]

    for t in range(T):
        expanded: List[_Beam] = []
        yt = y[t]
        ht = h[t].reshape(())

        for b in beams:
            out = model(b.ids.unsqueeze(0))
            logits_last = out.logits[:, -1, :].float()
            logp = F.log_softmax(logits_last, dim=-1).squeeze(0)

            Vuse = min(V, int(logp.numel()))

            # Channel candidates (always available)
            ch_ids = channel_token_candidates_from_y(
                y_chunk_complex=yt,
                h_scalar=ht,
                M=constellation.M,
                k=constellation.k,
                L=constellation.L,
                nbits=constellation.nbits,
                nbits_padded=constellation.nbits_padded,
                max_token_cands=32,
                vocab_size=Vuse,  # <-- ADD
                sym_topm=2,
            )

            if b.ended:
                # after EOS, force EOS ONLY (fixed-length scoring)
                cand_ids = torch.tensor([eos], device=device, dtype=torch.long)
            else:
                # LM top-K
                k_lm = min(candidate_topk, Vuse)
                lm_logp, lm_ids = torch.topk(logp[:Vuse], k=k_lm)

                # Union = LM top-K ∪ channel candidates
                cand_ids = torch.unique(torch.cat([lm_ids, ch_ids], dim=0))
            fallback_id = int(torch.argmax(logp[:Vuse]).item())
            cand_ids = _sanitize_cand_ids(cand_ids, Vuse=Vuse, fallback_id=fallback_id)

            # logp for candidates
            cand_logp = logp.index_select(0, cand_ids)

            # channel distance
            dist = _dist_sq_candidates(yt, ht, constellation, cand_ids)

            # score update (consistent with your current choice)
            scores = b.score + (cand_logp - dist / n0)

            for j in range(cand_ids.numel()):
                tid = int(cand_ids[j].item())
                new_ids = torch.cat([b.ids, cand_ids[j].view(1)], dim=0)
                expanded.append(_Beam(new_ids, float(scores[j].item()), (b.ended or tid == eos)))

        expanded.sort(key=lambda z: z.score, reverse=True)
        beams = expanded[:beam_width]
        # IMPORTANT: no early break

    best = max(beams, key=lambda z: z.score).ids
    if bos is not None and best.numel() and best[0].item() == bos:
        best = best[1:]
    # If somehow shorter, pad to T with EOS
    if best.numel() < T:
        best = torch.cat([best, torch.full((T - best.numel(),), eos, device=device, dtype=torch.long)], dim=0)
    return best[:T]


@torch.no_grad()
def llmsc_beam_decode_two_links(
    model,
    tokenizer,
    constellation: TokenConstellation,
    *,
    received_sd_real: torch.Tensor,    # [T,2L]
    h_sd_tokens: Optional[torch.Tensor],
    received_rd_real: torch.Tensor,    # [T,2L]
    h_rd_tokens: Optional[torch.Tensor],
    n0: float,
    beam_width: int,
    candidate_topk: int,
    rd_weight: float = 1.0,
) -> torch.Tensor:
    device = received_sd_real.device
    T = received_sd_real.shape[0]
    y1 = _to_complex(received_sd_real)
    y2 = _to_complex(received_rd_real)
    h1 = _h_to_complex(h_sd_tokens, received_sd_real)
    h2 = _h_to_complex(h_rd_tokens, received_rd_real)

    V = constellation.vocab_size
    eos = tokenizer.eos_token_id
    bos = tokenizer.bos_token_id

    if eos is None:
        raise ValueError("Tokenizer has no eos_token_id; fixed-length decoding needs EOS.")

    start = torch.tensor([bos], device=device, dtype=torch.long) if bos is not None else torch.empty((0,), device=device, dtype=torch.long)
    beams: List[_Beam] = [_Beam(start, 0.0, False)]

    rd_weight = float(rd_weight)

    for t in range(T):
        expanded: List[_Beam] = []
        y1t = y1[t]; y2t = y2[t]
        h1t = h1[t].reshape(()); h2t = h2[t].reshape(())

        for b in beams:
            out = model(b.ids.unsqueeze(0))
            logits_last = out.logits[:, -1, :].float()
            logp = F.log_softmax(logits_last, dim=-1).squeeze(0)

            Vuse = min(V, int(logp.numel()))

            ch_ids_sd = channel_token_candidates_from_y(
                y_chunk_complex=y1t, h_scalar=h1t,
                M=constellation.M, k=constellation.k, L=constellation.L,
                nbits=constellation.nbits, nbits_padded=constellation.nbits_padded,
                max_token_cands=32, sym_topm=2,     vocab_size=Vuse,              # <-- ADD
            )
            ch_ids_rd = channel_token_candidates_from_y(
                y_chunk_complex=y2t, h_scalar=h2t,
                M=constellation.M, k=constellation.k, L=constellation.L,
                nbits=constellation.nbits, nbits_padded=constellation.nbits_padded,
                max_token_cands=32, sym_topm=2,     vocab_size=Vuse,              # <-- ADD
            )

            if b.ended:
                cand_ids = torch.tensor([eos], device=device, dtype=torch.long)
            else:
                k_lm = min(candidate_topk, Vuse)
                lm_logp, lm_ids = torch.topk(logp[:Vuse], k=k_lm)

                cand_ids = torch.unique(torch.cat([lm_ids, ch_ids_sd, ch_ids_rd], dim=0))
            fallback_id = int(torch.argmax(logp[:Vuse]).item())
            cand_ids = _sanitize_cand_ids(cand_ids, Vuse=Vuse, fallback_id=fallback_id)

            cand_logp = logp.index_select(0, cand_ids)

            d1 = _dist_sq_candidates(y1t, h1t, constellation, cand_ids)
            d2 = _dist_sq_candidates(y2t, h2t, constellation, cand_ids)
            dist = d1 + rd_weight * d2

            scores = b.score + (cand_logp - dist / n0)

            for j in range(cand_ids.numel()):
                tid = int(cand_ids[j].item())
                new_ids = torch.cat([b.ids, cand_ids[j].view(1)], dim=0)
                expanded.append(_Beam(new_ids, float(scores[j].item()), (b.ended or tid == eos)))

        expanded.sort(key=lambda z: z.score, reverse=True)
        beams = expanded[:beam_width]
        # IMPORTANT: no early break

    best = max(beams, key=lambda z: z.score).ids
    if bos is not None and best.numel() and best[0].item() == bos:
        best = best[1:]
    if best.numel() < T:
        best = torch.cat([best, torch.full((T - best.numel(),), eos, device=device, dtype=torch.long)], dim=0)
    return best[:T]



def _ids_to_logits(ids: torch.Tensor, vocab_size: int, on_value: float = 0.0, off_value: float = -1e9) -> torch.Tensor:
    B, T = ids.shape
    logits = torch.full((B, T, vocab_size), off_value, device=ids.device, dtype=torch.float32)
    logits.scatter_(2, ids.unsqueeze(-1), on_value)
    return logits


# ----------------------------
# Modules mirroring your baseline API
# ----------------------------

class Tx_Relay_LLMSC(nn.Module):
    def __init__(
        self,
        model_name: str,
        M: Optional[int],
        channel,
        n0: float = 1.0,
        beam_width: int = 10,
        candidate_topk: int = 256,
        entire_network_train: int = 0,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
        symbols_per_token: Optional[int] = None,
    ):
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        _dtype = getattr(torch, torch_dtype) if torch_dtype else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=_dtype, trust_remote_code=trust_remote_code
        ).to(self.device)
        self.model.eval()

        self.nin = int(self.tokenizer.vocab_size)

        # Auto-select modulation if requested
        if symbols_per_token is not None:
            M_sel, _nbits, _k, L_actual = choose_modulation_from_symbols_per_token(self.nin, int(symbols_per_token))
            self.M = int(M_sel)
            self.symbols_per_token_target = int(symbols_per_token)
            self.symbols_per_token_actual = int(L_actual)
        else:
            if M is None:
                raise ValueError("Either M or symbols_per_token must be provided.")
            self.M = int(M)
            self.symbols_per_token_target = None
            self.symbols_per_token_actual = None

        self.constellation = TokenConstellation(self.nin, self.M, self.device, signal_power_constraint=channel.signal_power_constraint)

        self.channel = channel
        self.n0_fallback = float(n0)
        self.beam_width = int(beam_width)
        self.candidate_topk = int(candidate_topk)
        self.entire_network_train = int(entire_network_train)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, d_sr):
        x = x.to(self.device)
        attention_mask = attention_mask.to(self.device)
        x = x.long()

        ch_input = self.constellation.encode(x)  # [B,T,2L]

        if self.channel is not None:
            y_sr, h_sr = _channel_call(self.channel, ch_input, d_sr)
        else:
            y_sr, h_sr = ch_input, None

        hop_n0 = _compute_N0(self.channel, d_sr) or self.n0_fallback

        B, T, _ = y_sr.shape
        pred_ids = torch.empty((B, T), device=self.device, dtype=torch.long)
        for b in range(B):
            hb = h_sr[b] if h_sr is not None else None
            T_eff = int(attention_mask[b].sum().item())
            yb = y_sr[b, :T_eff]
            hb_eff = hb[:T_eff] if hb is not None else None

            ids_eff = llmsc_beam_decode_one_link(
                self.model, self.tokenizer, self.constellation,
                received_real=yb,
                h_tokens=hb_eff,
                n0=hop_n0,
                beam_width=self.beam_width,
                candidate_topk=self.candidate_topk,
            )

            # pad back to T with eos (or pad)
            pad_id = self.tokenizer.eos_token_id
            if T_eff < T:
                ids_eff = torch.cat([ids_eff, torch.full((T - T_eff,), pad_id, device=self.device, dtype=torch.long)],
                                    dim=0)
            pred_ids[b] = ids_eff[:T]

        x_hat = _ids_to_logits(pred_ids, self.nin)

        if self.entire_network_train == 0:
            B2, T2, V = x_hat.shape
            logits = x_hat.reshape(B2 * T2, V)
            grnd_x = x.reshape(B2 * T2)
            mask = attention_mask.flatten() == 1
            loss = F.cross_entropy(logits[mask, :], grnd_x[mask])
            return x_hat, ch_input, loss

        return x_hat, ch_input


class Tx_Relay_Rx_LLMSC(nn.Module):
    def __init__(
        self,
        model_name: str,
        M: Optional[int],
        channel,
        tx_relay_model: Tx_Relay_LLMSC,
        n0: float = 1.0,
        beam_width: int = 10,
        candidate_topk: int = 256,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
        symbols_per_token: Optional[int] = None,
    ):
        super().__init__()
        self.device = torch.device(device) if device is not None else tx_relay_model.device

        self.tx_relay_model = tx_relay_model
        for p in self.tx_relay_model.parameters():
            p.requires_grad = False

        self.tokenizer = tx_relay_model.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        _dtype = getattr(torch, torch_dtype) if torch_dtype else None
        self.model = tx_relay_model.model
        self.model.eval()

        self.nin = tx_relay_model.nin

        # Match modulation with tx_relay_model if possible
        if hasattr(tx_relay_model, "M"):
            self.M = int(tx_relay_model.M)
        elif symbols_per_token is not None:
            M_sel, _nbits, _k, L_actual = choose_modulation_from_symbols_per_token(self.nin, int(symbols_per_token))
            self.M = int(M_sel)
        else:
            if M is None:
                raise ValueError("Either M or symbols_per_token must be provided.")
            self.M = int(M)

        self.constellation = TokenConstellation(self.nin, self.M, self.device, signal_power_constraint=channel.signal_power_constraint)

        self.channel = channel
        self.n0_fallback = float(n0)
        self.beam_width = int(beam_width)
        self.candidate_topk = int(candidate_topk)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, d_sd, d_sr, d_rd):
        x = x.to(self.device)
        attention_mask = attention_mask.to(self.device)
        x = x.long()

        relay_logits, _ch_in = self.tx_relay_model(x, attention_mask, d_sr)[:2]
        relay_ids = torch.argmax(relay_logits, dim=2)

        # Relay forward
        s_relay = self.constellation.encode(relay_ids)
        if self.channel is not None:
            y_rd, h_rd = _channel_call(self.channel, s_relay, d_rd)
        else:
            y_rd, h_rd = s_relay, None

        # Direct link
        s_src = self.constellation.encode(x)
        if self.channel is not None:
            y_sd, h_sd = _channel_call(self.channel, s_src, d_sd)
        else:
            y_sd, h_sd = s_src, None

        N0_sd = _compute_N0(self.channel, d_sd) or self.n0_fallback
        N0_rd = _compute_N0(self.channel, d_rd) or self.n0_fallback
        rd_weight = float(N0_sd / N0_rd) if (N0_sd is not None and N0_rd is not None and N0_rd != 0) else 1.0

        B, T, _ = y_sd.shape
        dest_ids = torch.empty((B, T), device=self.device, dtype=torch.long)
        for b in range(B):
            hb_sd = h_sd[b] if h_sd is not None else None
            hb_rd = h_rd[b] if h_rd is not None else None

            T_eff = int(attention_mask[b].sum().item())

            ysd_b = y_sd[b, :T_eff]
            yrd_b = y_rd[b, :T_eff]
            hb_sd_eff = hb_sd[:T_eff] if hb_sd is not None else None
            hb_rd_eff = hb_rd[:T_eff] if hb_rd is not None else None

            ids_eff = llmsc_beam_decode_two_links(
                self.model, self.tokenizer, self.constellation,
                received_sd_real=ysd_b,
                h_sd_tokens=hb_sd_eff,
                received_rd_real=yrd_b,
                h_rd_tokens=hb_rd_eff,
                n0=N0_sd,
                beam_width=self.beam_width,
                candidate_topk=self.candidate_topk,
                rd_weight=rd_weight,
            )

            pad_id = self.tokenizer.eos_token_id
            if T_eff < T:
                ids_eff = torch.cat(
                    [ids_eff, torch.full((T - T_eff,), pad_id, device=self.device, dtype=torch.long)],
                    dim=0
                )

            dest_ids[b] = ids_eff[:T]

        x_hat = _ids_to_logits(dest_ids, self.nin)

        B2, T2, V = x_hat.shape
        logits = x_hat.reshape(B2 * T2, V)
        grnd_x = x.reshape(B2 * T2)
        mask = attention_mask.flatten() == 1
        loss = F.cross_entropy(logits[mask, :], grnd_x[mask])
        return x_hat, loss
