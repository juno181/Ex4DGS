import torch
import torch.nn
import torch.nn.functional as F


def linear_interp_uniiterval(y1, y2, t):    
    return (y1 * (1 - t) + y2 * t)
    
    
def slerp_interp_uniiterval(v1, v2, t):
    assert t >= 0  and t <= 1
    
    # normalizse the input vectors
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
    
    d = (v1 * v2).sum(-1, keepdim=True) 
    omega = torch.acos(d).clamp_min(1e-4)
    s_omega = torch.sin(omega).clamp_min(1e-4)
    p_0 = torch.sin((1 - t) * omega) / s_omega
    p_1 = torch.sin(t * omega) / s_omega
    p_sum = (p_0 + p_1).clamp_min(1e-4)
    p_0 = p_0 / p_sum
    p_1 = p_1 / p_sum
    
    # prevent zero vector
    ret = (v1 * p_0 + v2 * p_1)
    ret = torch.where(ret.abs().sum(-1, keepdim=True) > 1e-4, ret, v1)
    
    return ret / ret.norm(dim=-1, keepdim=True)
        
    
def quat_slerp_interp_uniiterval(v1, v2, t):
    # https://en.wikipedia.org/wiki/Slerp
    # normalizse the input vectors
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
    
    d = (v1 * v2).sum(-1, keepdim=True).clamp(-1+1e-4, 1-1e-4)
    omega = torch.acos(d).clamp_min(1e-4)
    s_omega = torch.sin(omega).clamp_min(1e-4)
    p_0 = torch.sin((1 - t) * omega) / s_omega
    p_1 = torch.sin(t * omega) / s_omega
    p_sum = (p_0 + p_1).clamp_min(1e-4)
    p_0 = p_0 / p_sum
    p_1 = p_1 / p_sum
    
    # prevent zero vector
    ret = (v1 * p_0 + v2 * p_1)
    ret = torch.where(ret.abs().sum(-1, keepdim=True) > 1e-4, ret, v1)
    
    return ret / ret.norm(dim=-1, keepdim=True)


def time_bigaussian(mean, var, t, training=False, var_min=0.1):
    m =  (t - mean).min(dim=1)[0]
    v = torch.where((t > mean).any(dim=1), var[:, 1], var[:, 0])
    opacity_ = m.pow(2) / (v.exp() + var_min/2.36).pow(2)
    opacity_ = torch.exp(-1*opacity_)
    
    return torch.where( (mean[:, 0] - t) * (mean[:, 1] - t) < 0, torch.ones_like(opacity_), opacity_)


# x time, y values
def pchip_interpolate(y_km1, y_k, y_k1, y_k2, delta_t):
    # hermite basis
    h_00 = lambda x :  2*x**3 - 3*x**2 + 1
    h_10 = lambda x :    x**3 - 2*x**2 + x
    h_01 = lambda x : -2*x**3 + 3*x**2
    h_11 = lambda x :    x**3 -   x**2

    m_k  = torch.where((y_k1 - y_k) * (y_k - y_km1) > 0, (y_k1 - y_k) * (y_k - y_km1) / (y_k1 - y_km1) * 2,torch.zeros_like(y_k))
    m_k1 = torch.where((y_k2 - y_k1) * (y_k1 - y_k) > 0, (y_k2 - y_k1) * (y_k1 - y_k) / (y_k2 - y_k) * 2, torch.zeros_like(y_k))

    p_x = h_00(delta_t) * y_k + h_10(delta_t) * m_k + h_01(delta_t) * y_k1 + h_11(delta_t) * m_k1

    return p_x


# x time, y values
def cube_interpolate(y_km1, y_k, y_k1, y_k2, delta_t):
    # hermite basis
    h_00 = lambda x :  2*x**3 - 3*x**2 + 1
    h_10 = lambda x :    x**3 - 2*x**2 + x
    h_01 = lambda x : -2*x**3 + 3*x**2
    h_11 = lambda x :    x**3 -   x**2

    m_k  = (y_k1 - y_km1) / 2
    m_k1 = (y_k2 - y_k) / 2

    p_x = h_00(delta_t) * y_k + h_10(delta_t) * m_k + h_01(delta_t) * y_k1 + h_11(delta_t) * m_k1

    return p_x


# x time, y values
def quad_diff_interpolate(y_k1, y_k2, y_m1, y_m2, delta_t):
    # hermite basis
    h_00 = lambda x :  2*x**3 - 3*x**2 + 1
    h_10 = lambda x :    x**3 - 2*x**2 + x
    h_01 = lambda x : -2*x**3 + 3*x**2
    h_11 = lambda x :    x**3 -   x**2

    p_x = h_00(delta_t) * y_k1 + h_10(delta_t) * y_m1 + \
          h_01(delta_t) * y_k2 + h_11(delta_t) * y_m2 

    return p_x

