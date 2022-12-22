Python Fast-Quadric-Mesh-Simplification Wrapper
===============================================
This is a python wrapping of the `Fast-Quadric-Mesh-Simplification Library
<https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification/>`_. Having
arrived at the same problem as the original author, but needing a Python
library, this project seeks to extend the work of the original library while
adding integration to Python and the `PyVista
<https://github.com/pyvista/pyvista>`_ project.

.. image:: https://github.com/pyvista/fast-simplification/raw/main/doc/images/simplify_demo.png
   

Basic Usage
-----------
The basic interface is quite straightforward and can work directly
with arrays of points and triangles:

.. code:: python   

    points = [[ 0.5, -0.5, 0.0],
              [ 0.0, -0.5, 0.0],
              [-0.5, -0.5, 0.0],
              [ 0.5,  0.0, 0.0],
              [ 0.0,  0.0, 0.0],
              [-0.5,  0.0, 0.0],
              [ 0.5,  0.5, 0.0],
              [ 0.0,  0.5, 0.0],
              [-0.5,  0.5, 0.0]]

    faces = [[0, 1, 3],
             [4, 3, 1],
             [1, 2, 4],
             [5, 4, 2],
             [3, 4, 6],
             [7, 6, 4],
             [4, 5, 7],
             [8, 7, 5]]

    points_out, faces_out = fast_simplification.simplify(points, faces, 0.5)
   

Advanced Usage
--------------
This library supports direct integration with VTK through PyVista to
provide a simplistic interface to the library. As this library
provides a 4-5x improvement to the VTK decimation algorithms.

.. code:: python

   >>> from pyvista import examples
   >>> mesh = examples.download_nefertiti()
   >>> out = fast_simplification.simplify_mesh(mesh, target_reduction=0.9)

   Compare with built-in VTK/PyVista methods:

   >>> fas_sim = fast_simplification.simplify_mesh(mesh, target_reduction=0.9)
   >>> dec_std = mesh.decimate(0.9)  # vtkQuadricDecimation
   >>> dec_pro = mesh.decimate_pro(0.9)  # vtkDecimatePro

   >>> pv.set_plot_theme('document')
   >>> pl = pv.Plotter(shape=(2, 2), window_size=(1000, 1000))
   >>> pl.add_text('Original', 'upper_right', color='w')
   >>> pl.add_mesh(mesh, show_edges=True)
   >>> pl.camera_position = cpos

   >>> pl.subplot(0, 1)
   >>> pl.add_text(
   ...    'Fast-Quadric-Mesh-Simplification\n~2.2 seconds', 'upper_right', color='w'
   ... )
   >>> pl.add_mesh(fas_sim, show_edges=True)
   >>> pl.camera_position = cpos

   >>> pl.subplot(1, 0)
   >>> pl.add_mesh(dec_std, show_edges=True)
   >>> pl.add_text(
   ...    'vtkQuadricDecimation\n~9.5 seconds', 'upper_right', color='w'
   ... )
   >>> pl.camera_position = cpos

   >>> pl.subplot(1, 1)
   >>> pl.add_mesh(dec_pro, show_edges=True)
   >>> pl.add_text(
   ...    'vtkDecimatePro\n11.4~ seconds', 'upper_right', color='w'
   ... )
   >>> pl.camera_position = cpos
   >>> pl.show()


Comparison to other libraries
-----------------------------
The `pyfqmr <https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction>`_
library wraps the same header file as this library and has similar capabilities.
In this library, the decision was made to write the Cython layer on top of an
additional C++ layer rather than directly interfacing with wrapper from Cython.
This results in a mild performance improvement.

Reusing the example above:

.. code:: python

   Set up a timing function.

   >>> import pyfqmr
   >>> vertices = mesh.points
   >>> faces = mesh.faces.reshape(-1, 4)[:, 1:]
   >>> def time_pyfqmr():
   ...     mesh_simplifier = pyfqmr.Simplify()
   ...     mesh_simplifier.setMesh(vertices, faces)
   ...     mesh_simplifier.simplify_mesh(
   ...         target_count=out.n_faces, aggressiveness=7, verbose=0
   ...     )
   ...     vertices_out, faces_out, normals_out = mesh_simplifier.getMesh()
   ...     return vertices_out, faces_out, normals_out

Now, time it and compare with the non-VTK API of this library:

.. code:: python

   >>> timeit time_pyfqmr()
   2.75 s ± 5.35 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

   >>> timeit vout, fout = fast_simplification.simplify(vertices, faces, 0.9)
   2.05 s ± 3.18 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

Additionally, the ``fast-simplification`` library has direct plugins
to the ``pyvista`` library, making it easy to read and write meshes:

.. code:: python

   >>> import pyvista
   >>> import fast_simplification
   >>> mesh = pyvista.read('my_mesh.stl')
   >>> simple = fast_simplification.simplify_mesh(mesh)
   >>> simple.save('my_simple_mesh.stl')

Since both libraries are based on the same core C++ code, feel free to
use whichever gives you the best performance and interoperability.
