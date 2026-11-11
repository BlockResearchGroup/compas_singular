# Writing remarks

A remark is prose attached to the session: why a setting was chosen, what is
still wrong, what was tried and rejected. `history` returns them alongside the
numbers, and `save_example` carries them into the corpus, so a remark written now
is what a later session reads back.

The numbers already record *what changed*. A remark should record what the
numbers cannot:

- **Why a step was taken.** "share_below was 0, so this is one face, not the
  mesh" is worth keeping. "Ran smooth_region" is not -- the history says that.
- **What was rejected and why.** A pass that was undone is evidence. Say what it
  did and that it was taken back.
- **What is still wrong at the end.** Especially when it cannot be fixed by
  smoothing. "The two singularities either side of the hole are 1.5 apart and
  the elements between them stay poor; this is a layout problem" is the single
  most useful thing to leave behind.
- **Anything surprising.** A mesh that got worse under a gentle pass, a boundary
  that would not slide, a region that creased.

Anchor a remark with `about` when it is about a place -- pass the vertex handle
from a report. An unanchored remark is about the session as a whole.

Keep them short and specific. A remark that restates the metrics is noise in
every session that reads it afterwards.

## Verdicts

`save_example` takes one of three:

`good`
: Worth copying. The approach was right and the numbers moved.
`acceptable`
: It worked, but there was a better route, or it took more steps than it should
  have. Say which in the remark.
`bad`
: Do not repeat this. **Save these** -- a bad example is as instructive as a good
  one, and the corpus returns them deliberately.
