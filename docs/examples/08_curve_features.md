# Curve features

A curve feature is a line the quad mesh must follow: a crease, a rib, a joint. The domain is cut open along the line before the skeleton is computed, so the line becomes a row of mesh edges.

![Curve features](../assets/images/examples/08_curve_features.png)

```python
--8<-- "examples/08_curve_features.py"
```
