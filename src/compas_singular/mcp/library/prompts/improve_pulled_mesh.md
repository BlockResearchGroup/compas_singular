Improve the quad mesh currently in the Rhino document.

Work in this order, and read before you act:

1. `rhino_status` -- confirm the link is attached. If it is not, say so and stop;
   the fix is for the person to run `CMD_mcp_link` in Rhino.
2. `rhino_pull` -- take the mesh and the boundary curves.
3. `inspect` -- read the `reading` and the `verdict`, not just the numbers. Note
   `share_below` in particular: it decides whether the problem is general or a
   single face.
4. `recall_examples` -- check whether a comparable mesh has been handled before.
5. Read `guidance://smoothing` if you have not already, and follow the order it
   sets out. `relax` first: it is gated and cannot make the mesh worse.
6. `snapshot` before any ungated pass, `inspect` after it, and `undo` anything
   that did not improve all three of min angle, max angle and aspect.
7. **Look at it.** `inspect` with `image=true`, and judge the four checks in
   `guidance://visual`: has the mesh left its boundary (any orange showing), did
   every point feature become a pole (a disc inside every ring), does it respect
   the guides, and do the polyedges run continuously or kink. This step is not
   optional -- `rhino_push` refuses until the current mesh has been drawn.
8. `remark` on why you chose what you chose, and on anything still wrong,
   including anything you saw in the image that the numbers do not show.
9. `rhino_push` -- only when the mesh is actually better than you found it. If it
   is not, say so and push nothing.

Two passes that change nothing mean the layout is the limit, not the smoothing.
Report that plainly rather than trying a fourth setting.
