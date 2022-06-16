"""
Compare Decimation Methods
--------------------------

This example compares various decimation methods

"""

import time

import pyvista as pv
from pyvista import examples

import fast_simplification

# load the example bunny mesh
mesh = examples.download_louis_louvre()

# nice camera angle
cpos = [(6.264157141857314, -6.959267635766402, 11.71668951132694),
        (1.3291685457683413, 2.267162128740896, 12.263240938610595),
        (0.0023825740958850136, -0.05786378450796799, 0.9983216444528751)]


###############################################################################
# Compare decimation times
reduction = 0.9
print("Approach                        Time Elapsed")

tstart = time.time()
fas_sim = fast_simplification.simplify_mesh(mesh, target_reduction=reduction)
print(f"Fast Quadratic Simplification  {time.time() - tstart:8.4f} seconds")

tstart = time.time()
dec_std = mesh.decimate(reduction)
print(f"vtkQuadricDecimation           {time.time() - tstart:8.4f} seconds")

tstart = time.time()
dec_pro = mesh.decimate_pro(reduction)
print(f"vtkDecimatePro                 {time.time() - tstart:8.4f} seconds")


pv.set_plot_theme('document')
pl = pv.Plotter(shape=(2, 2), window_size=(1000, 1000))
pl.add_text('Original', 'upper_right', color='w')
pl.add_mesh(mesh, show_edges=True)
pl.camera_position = cpos
pl.subplot(0, 1)
pl.add_text(
    'Fast-Quadric-Mesh-Simplification', 'upper_right', color='w'
)
pl.add_mesh(fas_sim, show_edges=True)
pl.camera_position = cpos

pl.subplot(1, 0)
pl.add_mesh(dec_std, show_edges=True)
pl.add_text(
    'vtkQuadricDecimation', 'upper_right', color='w'
)
pl.camera_position = cpos

pl.subplot(1, 1)
pl.add_mesh(dec_pro, show_edges=True)
pl.add_text(
    'vtkDecimatePro\n11', 'upper_right', color='w'
)
pl.camera_position = cpos

pl.show()
