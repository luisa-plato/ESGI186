# from dolfin import *
from dolfin import Mesh
import pytest

def test_load_meshes():
    # test that meshes can be loaded correctly
    mesh1 = Mesh("./meshes/disc.xml")
    mesh2 = Mesh("./meshes/half_disc.xml")
           
    # test if meshes are loaded correctly
    assert mesh1 is not None
    assert mesh2 is not None
    
    # test if meshes have vertices
    assert mesh1.num_vertices() > 0
    assert mesh2.num_vertices() > 0

