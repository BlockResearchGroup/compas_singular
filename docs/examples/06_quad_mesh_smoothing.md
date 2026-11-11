# Smoothing the quad mesh

A dense mesh straight from `densify()` is made of patches that were each filled on their own, so the quads kink where patches meet. Smoothing moves the vertices to even out the quads, without changing the topology.

![Smoothing the quad mesh](../assets/images/examples/06_quad_mesh_smoothing.png)

```python
--8<-- "examples/06_quad_mesh_smoothing.py"
```
