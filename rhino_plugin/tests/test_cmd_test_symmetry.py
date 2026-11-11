"""CMD_test_symmetry without Rhino: stub the document, script the prompts, run main().

What is stubbed: ``rhinoscriptsyntax`` (prompts, layers, points, lines),
``scriptcontext`` (redraw), the bootstrap purge (switched off) and ``project.get_settings`` (fixed settings) and the
document helpers (``read_boundaries`` returns a domain; the ``bake_*`` functions
record what they were given). What is NOT stubbed: everything the command does
with the library -- detection, the unit, both expansions -- which is the point.

Run under both interpreters:
    singular312:      python test_cmd_test_symmetry.py
    Rhino 8 CPython:  PYTHONPATH=<site-envs/default-gdaObJ1D> python test_cmd_test_symmetry.py
"""
import os
import sys
import types

from _cases import FOUR_POLES, PLUS, SQUARE, check, finish, ngon, trefoil

COMMANDS = r"C:\Users\Casper\libraries\carbcomn\compas_singular\compas_singular\rhino_plugin\dev"
if COMMANDS not in sys.path:
    sys.path.insert(0, COMMANDS)

from compas.geometry import Polyline  # noqa: E402

from compas_singular.symmetry import SymmetryGroup  # noqa: E402
from compas_singular.symmetry.measure import mesh_invariance  # noqa: E402


class StubRs(object):

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []
        self.layers = set()
        self.objects = {}
        self._next = 0

    def GetString(self, message=None, defaultString=None, strings=None):
        self.prompts.append((message, strings))
        if not self.answers:
            raise AssertionError('unexpected prompt: {} {}'.format(message, strings))
        answer = self.answers.pop(0)
        if strings is not None and answer not in strings:
            raise AssertionError('answer {!r} not offered in {}'.format(answer, strings))
        return answer

    def IsLayer(self, name):
        return name in self.layers

    def AddLayer(self, name=None, color=None, visible=True, locked=False, parent=None):
        full = '{}::{}'.format(parent, name) if parent else name
        self.layers.add(full)
        return full

    def _add(self, kind, data):
        self._next += 1
        guid = '{}-{}'.format(kind, self._next)
        self.objects[guid] = [kind, None, data]
        return guid

    def AddPoint(self, point):
        return self._add('point', point)

    def AddLine(self, a, b):
        return self._add('line', (a, b))

    def AddTextDot(self, text, point):
        return self._add('dot', (text, point))

    def ObjectLayer(self, guid, layer=None):
        self.objects[guid][1] = layer


def install_stubs(answers, domain):
    rs = StubRs(answers)
    sys.modules['rhinoscriptsyntax'] = rs

    views = types.SimpleNamespace(Redraw=lambda: None)
    sys.modules['scriptcontext'] = types.SimpleNamespace(doc=types.SimpleNamespace(Views=views))

    sys.compas_singular_keep_modules = True
    import compas_singular.rhino.project as project
    project.get_settings = lambda: {'triangulation_spacing': 0.5, 'target_length': 0.5}

    baked = {}

    def clear_layer(layer, clean_sublayers=False):
        baked.pop(layer, None)
        for guid in [g for g, (_, lay, _) in rs.objects.items() if lay == layer]:
            del rs.objects[guid]

    def bake_mesh(mesh, layer, color=None, clear_existing=True):
        baked[layer] = mesh
        return 'mesh'

    def bake_polylines(polylines, layer, color=None, clear_existing=True):
        baked.setdefault(layer + ' (polylines)', []).extend(polylines)
        return ['p'] * len(polylines), 0

    def read_boundaries(spacing=None):
        outer = Polyline(domain['outer'] + [domain['outer'][0]])
        inners = [Polyline(h + [h[0]]) for h in domain.get('holes', [])]
        guides = [Polyline(g) for g in domain.get('guides', [])]
        return outer, inners, guides, domain.get('poles', [])

    helpers = types.ModuleType('compas_singular.rhino.helpers')
    helpers.clear_layer = clear_layer
    helpers.bake_mesh = bake_mesh
    helpers.bake_polylines = bake_polylines
    helpers.read_boundaries = read_boundaries
    sys.modules['compas_singular.rhino.helpers'] = helpers
    sys.modules.pop('CMD_test_symmetry', None)
    return rs, baked


def run(label, answers, domain, expect_group):
    print(label)
    rs, baked = install_stubs(answers, domain)
    import CMD_test_symmetry
    CMD_test_symmetry.main()
    root = CMD_test_symmetry.ROOT
    for name in ('CoarseUnit', 'CoarseLayout', 'QuadUnit', 'QuadMesh'):
        check('{}: baked {}'.format(label, name), root + '::' + name in baked, sorted(baked))
    if root + '::QuadMesh' not in baked:
        return
    mesh = baked[root + '::QuadMesh']
    unit = baked[root + '::CoarseUnit']
    group = SymmetryGroup.from_data(unit.symmetry['group'])
    inv = mesh_invariance(mesh, group)
    check('{}: enforced {}'.format(label, expect_group), group.name == expect_group, group.name)
    check('{}: baked quad mesh 100% symmetric'.format(label), inv['share'] == 1.0, inv)
    check('{}: axes and unit outline drawn'.format(label),
          any(lay == root + '::Symmetry' for _, lay, _ in rs.objects.values())
          and (root + '::UnitOutline (polylines)') in baked)
    check('{}: every prompt answered'.format(label), not rs.answers, rs.answers)


run('skeleton, square with 4 poles, keys M0 M90',
    ['Skeleton', 'M0', 'M90', 'Done', 'Continue'],
    dict(outer=SQUARE, poles=FOUR_POLES), 'D2')
run('field, plus plate, all',
    ['FrameField', 'All', 'Continue'],
    dict(outer=[list(p) for p in PLUS]), 'D4')
o3, i3 = trefoil(hole=2.0)
run('skeleton, trefoil, R120',
    ['Skeleton', 'R120', 'Done', 'Continue'],
    dict(outer=o3, holes=i3), 'C3')

print('exit paths')
rs, baked = install_stubs(['Exit'], dict(outer=SQUARE))
import CMD_test_symmetry  # noqa: E402
CMD_test_symmetry.main()
check('Exit at the route prompt bakes nothing', not baked, sorted(baked))
rs, baked = install_stubs(['Skeleton', 'All', 'Exit'], dict(outer=SQUARE))
sys.modules.pop('CMD_test_symmetry', None)
import CMD_test_symmetry  # noqa: E402,F811
CMD_test_symmetry.main()
check('Exit after the outline bakes no mesh', not any('Unit' in k and 'Outline' not in k for k in baked), sorted(baked))

finish()
