import numpy as np
import pdb
def matching_error(d, e):
    return e / d

def system_error(d, e):
    return 2 * e / (d - 2 * e)

def dispairity(fx, B, z):
    return fx * B / z

def pixel_per_meter(fx, f):
    return fx / f

def pixels_in_B(fx, f, B):
    return B * pixel_per_meter(fx, f)

f = 0.006
fx = 1093
B = 1.2
e = 0
me = 1
msg = "z = %d, dx = %.4f, pixel_per_car = %d, error_on_1 = %.4f, error_on_2 = %.4f, error_on_3 = %.4f, error_on_6 = %.4f"
for z in range(1, 6):
    z *= 10
    dx = dispairity(fx, B, z)
    ppm = fx / z * 2
    error = matching_error(dx, me)
#     s_error = system_error(dx, e)
#     error = 1 - (1 - r_error) * (1 - s_error)
    print(msg % (z, dx, ppm, matching_error(dx, 1), matching_error(dx, 2), matching_error(dx, 3), matching_error(dx,6)))
          
