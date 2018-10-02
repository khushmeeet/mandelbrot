import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from numba import jit

from utils import save_image


@jit
def calculate_zn(c, max_iters, horizon):
  z = c
  for i in range(max_iters):
    abs_z = abs(z)
    if abs_z > horizon:
      return i - np.log(np.log(abs_z)) / np.log(2) + (np.log(np.log(horizon)) / np.log(2))
    z = z ** 2 + c
  return 0


@jit
def mandelbrot_set(x_min, x_max, y_min, y_max, width, height, max_iters):
  horizon = 2**40
  x = np.linspace(x_min, x_max, width)
  y = np.linspace(y_min, y_max, height)
  pixels = np.empty((width, height))
  
  for i in range(width):
    for j in range(height):
      pixels[i, j] = calculate_zn(x[i] + 1j * y[j], max_iters, horizon)
  
  return (x, y, pixels)


@jit
def mandelbrot_image(x_min, x_max, y_min, y_max, width=10, height=10, max_iters=512, cmap='jet', gamma=0.3):
    dpi = 72
    img_width = dpi * width
    img_height = dpi * height
    x, y, z = mandelbrot_set(x_min, x_max, y_min, y_max,
                             img_width, img_height, max_iters)

    fig, ax = plt.subplots(figsize=(width, height), dpi=72)
    ticks = np.arange(0, img_width, 3 * dpi)
    x_ticks = x_min + (x_max - x_min) * ticks / img_width
    plt.xticks(ticks, x_ticks)
    y_ticks = y_min + (y_max - y_min) * ticks / img_width
    plt.yticks(ticks, y_ticks)
    norm = colors.PowerNorm(gamma)
    ax.imshow(z.T, origin='lower', cmap=cmap, norm=norm)
    save_image(fig)


mandelbrot_image(-2.0, 0.5, -1.25, 1.25)
