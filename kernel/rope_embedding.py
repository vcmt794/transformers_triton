import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(INPUT, INPUT_stride,
                sin_ptr, cos_ptr, 
                sin_stride, cos_stride,
                seq_len,
                n_head,
                head_dim: tl.constexpr,
                BLOCK_SIZE: tl.constexpr,
                IS_BACKY: tl.constexpr):
  GROUP_SIZE = 4 # or 8
  pid_row = tl.program_id(0) # Row index, aka position in seq*bs
  pid_col = tl.program_id(1) # Col index, aka the index of Group of head
  offset  = tl.arange(0,BLOCK_SIZE) 
  half_head_dim = head_dim//2
  mask = offset < half_head_dim

  # load sin and cos base on the position
  # Remember this sin and cos in R^d//2
  sin = tl.load(sin_ptr + sin_stride*(pid_row % seq_len) + 
                offset, mask=mask, other=0) # first part: base on position. 2nd part: load the dimesion.
  cos = tl.load(cos_ptr + cos_stride*(pid_row % seq_len) + \
                offset, mask=mask, other=0)

  #for Backward: 
  # with K: 
  #               rotated_K  = K*cos + Rotate_half(K)*sin  (cos & sin in R^d) (1)
  # or we can rewrite into an MatMul for a rotation matrix:
  #               rotated_K  = K*cos + R @ K*sin
  #               R = [[0, I_d//2],[-I_d//2, 0]] --> R.T =[[0, -I_d//2], [I_d//2, 0]]
  # so derivative of rotated_K w.r.t K:
  #          der((1))/der(K) = cos + R.T*sin (2) --> or we can say that the sign of sin is reverse
  #                 (2)      = cos + R*-sin  (2')
  ###
  if IS_BACKY:
    sin = -sin
  pass
  #preapre head index
  start_head = pid_col * GROUP_SIZE 
  end_head = min( (start_head + GROUP_SIZE), n_head) # sometimes the number of head cant be divide by GROUP SIZE

  #Iterate through all the heads in loaded Group:
  for head in range(start_head, end_head):
    halfhead1_offset = pid_row*INPUT_stride+head*head_dim + offset
    halfhead2_offset = pid_row*INPUT_stride+head*head_dim + offset + half_head_dim

    # Load the K/Q
    halfhead1 = tl.load(INPUT + halfhead1_offset, mask=mask, other=0)
    halfhead2 = tl.load(INPUT + halfhead2_offset, mask=mask, other=0)
    #Apply Rotary and save
    tl.store(INPUT + halfhead1_offset, halfhead1*cos+halfhead2*-sin, mask=mask)
    tl.store(INPUT + halfhead2_offset, halfhead2*cos+halfhead1*sin, mask=mask)



class RoPE(torch.autograd.Function):
  @staticmethod
  def forward(ctx,INPUT, sin, cos):

    assert len(INPUT.shape) == 4 #(bs, sq, n_head, headdim) we will transpose(1,2) later
    bs, sq, n_head, head_dim = INPUT.size()
    sin = sin.squeeze()
    cos = cos.squeeze()
    n_rows = bs*sq
    INPUT = INPUT.view(n_rows, n_head*head_dim)
    BLOCK_SIZE, num_warps = calculate_settings(head_dim//2)

    # with given group size --> How many grp 
    GROUP_SIZE = 4 # 
    div, mod = divmod(n_head, GROUP_SIZE)
    grp_num = div + int(mod != 0)

    #use the kernel
    rope_kernel[(n_rows, grp_num,)](INPUT,INPUT.stride(0),
                      sin, cos,
                      sin.stride(0), cos.stride(0),
                      sq,
                      n_head,
                      head_dim,
                      BLOCK_SIZE,
                      IS_BACKY = False,
                      num_warps = num_warps)
    
    # Save this for backward pass
    ctx.cos = cos
    ctx.sin = sin
    ctx.BLOCK_SIZE = BLOCK_SIZE
    ctx.num_warps = num_warps
    ctx.grp_num = grp_num

    #reshape the Q/K
    INPUT = INPUT.view(bs,sq,n_head, head_dim) # Remember to transpose(1,2)
    return INPUT
  
  @staticmethod
  def backward(ctx, dY):
    bs, sq, n_head, head_dim = dY.size()
    sin = ctx.sin
    cos = ctx.cos
    n_rows = bs*sq
    dY = dY.reshape(n_rows, n_head*head_dim)
    

    rope_kernel[(n_rows, ctx.grp_num,)](dY, dY.stride(0),
                                        sin, cos,
                                        sin.stride(0), cos.stride(0),
                                        sq,
                                        n_head,
                                        head_dim,
                                        ctx.BLOCK_SIZE,
                                        IS_BACKY = True,
                                        num_warps = ctx.num_warps)
    dY = dY.reshape(bs,sq,n_head, head_dim)

    # from torch.autograd.Function.backward doc: "it should return as many tensors, as there were inputs to forward()." 
    # None let model knows that sin and cos aren't paramenters that need update.
    return dY, None, None 
  

@torch.compiler.disable
def apply_RoPE(Q, K, sin, cos):
  # Q in (bs, sq, n_h, h_d)
  Q  = RoPE.apply(Q, sin, cos).transpose(1,2) # (bs, n_h, sq, h_d)
  K  = RoPE.apply(K, sin, cos).transpose(1,2)
  return Q, K
