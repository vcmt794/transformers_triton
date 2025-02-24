import triton
import triton.language as tl

next_power_of_2 = triton.next_power_of_2
MAX_FUSED_SIZE = 2 ** 16 # Predefined number, based on your GPU's max cuda units.


def calculate_settings(n: int) -> (int, int,):
    BLOCK_SIZE: int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds " \
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps: int = 4
    if BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >= 8192: num_warps = 16
    elif BLOCK_SIZE >= 2048: num_warps = 8
    return BLOCK_SIZE, num_warps
