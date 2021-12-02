Python Fast-Quadric-Mesh-Simplification Wrapper
===============================================
This is a python wrapping of the `Fast-Quadric-Mesh-Simplification Library <https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification/>`_. Having arrived at the same problem as the original author, but needing a Python library, this project seeks to extend the work of the original library while adding integration with Python and the `PyVista <https://github.com/pyvista/pyvista>`_ project.

Usage
-----
The basic interface is quite straightforward and can work directly with arrays of points and triangles:

.. code:: python
   



import pyvista as pv
import fast_simplification
from pyvista import examples

# mesh = pv.Sphere(theta_resolution=300, phi_resolution=300)
mesh = examples.download_nefertiti()

out = fast_simplification.simplify_mesh(mesh, target_reduction=0.9)

cpos = [(183.12753009053094, -347.12194852107706, 53.064467139864554),
        (-48.71282243594153, -21.704785301053843, -13.311907764923113),
        (-0.1071583378997645, 0.12484949917192488, 0.98637198519376)]

out.plot(show_edges=True, cpos=cpos, window_size=(2000,2000))

"""
>>> timeit fast_simplification.simplify_mesh(mesh, 0.9)
3.14 s ± 136 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
"""

"""
Compare with built-in VTK/PyVista methods:

>>> timeit mesh.decimate_pro(0.9)
15.6 s ± 484 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> timeit mesh.decimate(0.9)
18.9 s ± 525 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

"""

fas_sim = fast_simplification.simplify_mesh(mesh, target_reduction=0.9)
dec_std = mesh.decimate(0.9)  # vtkQuadricDecimation
dec_pro = mesh.decimate_pro(0.9)  # vtkDecimatePro

pv.set_plot_theme('document')
pl = pv.Plotter(shape=(2, 2), window_size=(1000, 1000))
pl.add_text('Original', 'upper_right', color='w')
pl.add_mesh(mesh, show_edges=True)
pl.camera_position = cpos

pl.subplot(0, 1)
pl.add_text(
    'Fast-Quadric-Mesh-Simplification\n~3.1 seconds', 'upper_right', color='w'
)
pl.add_mesh(fast_sim, show_edges=True)
pl.camera_position = cpos

pl.subplot(1, 0)
pl.add_mesh(dec_std, show_edges=True)
pl.add_text(
    'vtkQuadricDecimation\n~16 seconds', 'upper_right', color='w'
)
pl.camera_position = cpos

pl.subplot(1, 1)
pl.add_mesh(dec_pro, show_edges=True)
pl.add_text(
    'vtkDecimatePro\n~19 seconds', 'upper_right', color='w'
)
pl.camera_position = cpos

pl.show()


###############################################################################
# With smooth shading
pl = pv.Plotter(shape=(2, 2), window_size=(1000, 1000))
pl.add_text('Original', 'upper_right', color='w')
pl.add_mesh(mesh, smooth_shading=True)
pl.camera_position = cpos

pl.subplot(0, 1)
pl.add_text(
    'Fast-Quadric-Mesh-Simplification\n~3.1 seconds', 'upper_right', color='w'
)
pl.add_mesh(fast_sim, smooth_shading=True)
pl.camera_position = cpos

pl.subplot(1, 0)
pl.add_mesh(dec_std, smooth_shading=True)
pl.add_text(
    'vtkQuadricDecimation\n~16 seconds', 'upper_right', color='w'
)
pl.camera_position = cpos

pl.subplot(1, 1)
pl.add_mesh(dec_pro, smooth_shading=True)
pl.add_text(
    'vtkDecimatePro\n~19 seconds', 'upper_right', color='w'
)
pl.camera_position = cpos

pl.show()

