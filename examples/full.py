import rhinoscriptsyntax as rs
import compas_rhino as rhino

from compas_pattern.cad.rhino.utilities import surface_borders

from compas.datastructures.mesh import Mesh
from compas_pattern.datastructures.pseudo_quad_mesh import PseudoQuadMesh
from compas_pattern.datastructures.pseudo_quad_mesh import pqm_from_mesh
from compas_pattern.cad.rhino.utilities import draw_mesh

from compas_pattern.algorithms.templating import templating

from compas_pattern.algorithms.mapping import mapping

from compas_pattern.algorithms.triangulation import triangulation

from compas_pattern.algorithms.extraction import extraction

from compas_pattern.algorithms.decomposition import decomposition

from compas_pattern.algorithms.conforming import conforming

from compas_pattern.algorithms.remapping import remapping

from compas_pattern.algorithms.editing import editing

from compas_pattern.algorithms.densification import densification

from compas_pattern.algorithms.patterning import patterning

from compas.geometry.algorithms.smoothing import mesh_smooth_centroid
from compas.geometry.algorithms.smoothing import mesh_smooth_area
from compas_pattern.algorithms.smoothing import define_constraints
from compas_pattern.algorithms.smoothing import apply_constraints

def start():
    # -1. layer structure
    layers = ['shape_and_features', 'initial_coarse_quad_mesh', 'edited_coarse_quad_mesh', 'quad_mesh', 'pattern_topology', 'pattern_geometry']
    colours = [[255,0,0], [0,0,0], [0,0,0], [200,200,200], [100,100,100], [0,0,0]]
    for layer, colour in zip(layers, colours):
        rs.AddLayer(layer)
        rs.LayerColor(layer, colour)
        objects = rs.ObjectsByLayer(layer)
        rs.DeleteObjects(objects)
    
    # 0. input
    surface_guid = rs.GetObject('select surface', filter = 8)
    rs.ObjectColor(surface_guid, [255,0,0])
    surface_guid = rs.CopyObject(surface_guid)
    rs.ObjectLayer(surface_guid, 'shape_and_features')
    curve_features_guids = rs.GetObjects('select curve features', filter = 4)
    if curve_features_guids is None:
        curve_features_guids = []
    rs.ObjectColor(curve_features_guids, [255,0,0])
    curve_features_guids = rs.CopyObjects(curve_features_guids)
    rs.ObjectLayer(curve_features_guids, 'shape_and_features')
    point_features_guids = rs.GetObjects('select point features', filter = 1)
    if point_features_guids is None:
        point_features_guids = []
    rs.ObjectColor(point_features_guids, [255,0,0])
    point_features_guids = rs.CopyObjects(point_features_guids)
    rs.ObjectLayer(point_features_guids, 'shape_and_features')
    
    coarse_quad_mesh = None
    inner_borders = surface_borders(surface_guid, border_type = 2)
    if len(inner_borders) == 0 and len(curve_features_guids) == 0 and len(point_features_guids) == 0:
        if rs.GetString('use template?', defaultString = 'False', strings = ['True', 'False']) == 'True':
            vertices, faces = templating()
            coarse_quad_mesh = PseudoQuadMesh.from_vertices_and_faces(vertices, faces)
    rs.DeleteObjects(inner_borders)
    
    if coarse_quad_mesh is None:
        # 1. mapping
        discretisation = rs.GetReal('NURBS element discretisation', number = 1)
        rs.EnableRedraw(False)
        planar_boundary_polyline, planar_hole_polylines, planar_polyline_features, planar_point_features = mapping(discretisation, surface_guid, curve_features_guids = curve_features_guids, point_features_guids = point_features_guids)
        
        # 2. triangulation
        delaunay_mesh = triangulation(planar_boundary_polyline, holes = planar_hole_polylines, polyline_features = planar_polyline_features, point_features = planar_point_features)
        #draw_mesh(delaunay_mesh)
        
        # 3. decomposition
        medial_branches, boundary_polylines = decomposition(delaunay_mesh)
        
        # 4. conforming
        # 5. extraction
        vertices, faces = extraction(boundary_polylines, medial_branches)
        patch_decomposition = PseudoQuadMesh.from_vertices_and_faces(vertices, faces)
        coarse_quad_mesh = conforming(patch_decomposition, delaunay_mesh, medial_branches, boundary_polylines, planar_point_features, planar_polyline_features)
        
        # 6. remapping
        remapping(coarse_quad_mesh, surface_guid)
    
    coarse_quad_mesh_guid = draw_mesh(coarse_quad_mesh)
    rs.ObjectLayer(coarse_quad_mesh_guid, layer = 'initial_coarse_quad_mesh')
    rs.LayerVisible('initial_coarse_quad_mesh', visible = True)
    
    # 7. editing
    rs.EnableRedraw(False)
    rs.LayerVisible('initial_coarse_quad_mesh', visible = False)
    editing(coarse_quad_mesh)
    coarse_quad_mesh_guid = draw_mesh(coarse_quad_mesh.to_mesh())
    rs.ObjectLayer(coarse_quad_mesh_guid, layer = 'edited_coarse_quad_mesh')
    rs.LayerVisible('edited_coarse_quad_mesh', visible = True)
    
    # 8. densification
    rs.EnableRedraw(True)
    target_length = rs.GetReal('target length for densification', number = 1)
    quad_mesh = densification(coarse_quad_mesh, target_length)
    quad_mesh = quad_mesh.to_mesh()
    quad_mesh_guid = draw_mesh(quad_mesh)
    rs.ObjectLayer(quad_mesh_guid, layer = 'quad_mesh')
    rs.LayerVisible('edited_coarse_quad_mesh', visible = False)
    rs.LayerVisible('quad_mesh', visible = True)
    
    # 9. patterning
    rs.EnableRedraw(True)
    operators = ['dual', 'join', 'ambo', 'kis', 'needle', 'gyro']
    patterning_operator = rs.GetString('patterning operator', strings = operators)
    rs.EnableRedraw(False)    
    pattern_topology = patterning(quad_mesh, patterning_operator)
    pattern_topology_guid = draw_mesh(pattern_topology)
    rs.ObjectLayer(pattern_topology_guid, layer = 'pattern_topology')
    rs.LayerVisible('quad_mesh', visible = False)
    rs.LayerVisible('pattern_topology', visible = True)
    
    # 10. smoothing
    pattern_geometry = pattern_topology.copy()
    rs.EnableRedraw(True)
    smoothing_iterations = rs.GetInteger('number of iterations for smoothing', number = 20)
    damping_value = rs.GetReal('damping value for smoothing', number = .5)
    rs.EnableRedraw(False)
    constraints, surface_boundaries = define_constraints(pattern_geometry, surface_guid, curve_constraints = curve_features_guids, point_constraints = point_features_guids)
    fixed_vertices = [vkey for vkey, constraint in constraints.items() if constraint[0] == 'point']
    mesh_smooth_area(pattern_geometry, fixed = fixed_vertices, kmax = smoothing_iterations, damping = damping_value, callback = apply_constraints, callback_args = [pattern_geometry, constraints])
    rs.DeleteObjects(surface_boundaries)
    pattern_geometry_guid = draw_mesh(pattern_geometry)
    rs.ObjectLayer(pattern_geometry_guid, layer = 'pattern_geometry')
    rs.LayerVisible('pattern_topology', visible = False)
    rs.LayerVisible('pattern_geometry', visible = True)
    
    rs.EnableRedraw(True)

start()