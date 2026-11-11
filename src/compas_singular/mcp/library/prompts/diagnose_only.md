Diagnose the quad mesh in the Rhino document. Change nothing.

Call `rhino_status`, then `rhino_pull`, then `inspect` **with `image=true`** --
a diagnosis without looking at the mesh is worth very little. Read
`guidance://quality` for the bands and `guidance://visual` for what to look for
in the picture.

Then report, in plain prose:

- What the worst element is and where, using the handle from the report.
- Whether the problem is local or general -- `share_below` is the number that
  settles it.
- Which of the three smoothers you would reach for, and why.
- What the IMAGE shows that the numbers do not: whether the mesh has left its
  boundary (orange showing), whether every input point feature became a pole (a
  disc inside every ring), whether the guides are respected, and whether the
  polyedges run continuously or kink.
- Whether anything you can see is a layout problem rather than an element
  problem. Smoothing cannot move a singularity, so if the defect is around one,
  say that no amount of smoothing will fix it.

Do not call any smoothing tool, do not snapshot, and do not push. This is an
opinion, not an edit.
