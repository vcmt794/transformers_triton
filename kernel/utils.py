import triton
import triton.language as tl

next_power_of_2 = triton.next_power_of_2
MAX_FUSED_SIZE = 2 ** 16 # Predefined number, based on your GPU's max cuda units.



'''HOW TO GET YOUR SET UP ?
device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

target = triton.runtime.driver.active.get_current_target()
kernels = {}
print('NUM_SM={nsm}, NUM_REGS={nreg}, SIZE_SMEM={sm}, WARP_SIZE={wsze} '.format(nsm=NUM_SM, nreg = NUM_REGS,sm=SIZE_SMEM,wsze= WARP_SIZE ))

# with SIZE_SMEM is MAX_FUSED_SIZE. 2**16 if using T4, P100 or anything free u can get free that u can get from Colab/Kaggle
'''

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
