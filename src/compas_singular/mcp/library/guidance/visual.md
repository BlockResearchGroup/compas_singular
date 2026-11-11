# Looking at the mesh

`inspect(image=true)` draws the mesh with its inputs. The numbers cannot see any
of what follows, which is the only reason to spend the tokens on a picture.

Leave the image **off** while iterating — it costs roughly 850 tokens a call.
Turn it on when you need to judge one of the four checks below, and **always
once at the end**: `rhino_push` and `save_mesh` refuse until the mesh being
delivered has actually been drawn.

## Reading the picture

| | |
|---|---|
| orange | the input boundary curves — outer and holes |
| green | the input guide curves |
| magenta ring | an input point feature |
| magenta disc | a pole in the mesh |
| red disc | an irregular interior vertex (a singularity) |
| black | the mesh outline |
| grey | the mesh interior edges |

The walls and guides are drawn **underneath** the mesh, and the wall stroke is
deliberately narrower than the mesh outline. That is what makes check 1 work:
where the mesh sits on its boundary the orange is completely covered, so **any
orange you can see is a real deviation**, not an artefact of line width.

## The four checks

**1. Boundaries.** Orange showing along an edge means the mesh has left the wall
it was meant to follow. Say where, and how far in mesh terms — one row of
elements, or the whole side. A mesh that measures beautifully and has walked off
its boundary is a failed mesh.

**2. Point features.** Each input point feature is a ring. A respected one has a
disc inside it. **An empty ring is an input the mesh ignored** — the point
feature never became a pole. This is not recoverable by smoothing; it means the
layout was built without that feature, and it is worth saying so plainly.

**3. Guides.** Green curves the mesh was supposed to follow. Report whether the
mesh runs *along* a guide or *across* it. Note that version 1 has no tool that
attaches a mesh to a guide, so a guide the mesh ignores is a finding to report,
not something to fix here.

**4. Smoothness and polyedge continuity.** Follow the grey grid lines across the
mesh. In a good mesh they run as continuous, gently curving families all the way
through. Look for:

- **kinks** — a grid line that changes direction abruptly at a vertex
- **fanning and pinching** — a strip that opens or closes sharply rather than
  gradually
- **waviness** — a family of lines that should be straight or smoothly curved
  wandering back and forth
- **propagation** — distortion that spreads away from where it starts

Some distortion around a red disc is unavoidable and **is not a defect**: an
irregular vertex is where the grid has to change direction, and the elements
around it will never be square. What matters is whether it stays local. A
singularity whose distortion propagates halfway across the mesh is a layout
problem, and no amount of smoothing will fix it.

## Say what you see

Report the four checks in prose, including "nothing wrong" where nothing is
wrong. Do not restate the metrics — they are already in the same result. The
picture is there for what the metrics cannot say.
