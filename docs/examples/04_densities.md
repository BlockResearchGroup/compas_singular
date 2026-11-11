# Densities

The density of a strip is the number of quads across it. A strip runs through several patches, and the patches on either side of an edge share it, so setting densities per strip always gives a conforming mesh.

![Densities](../assets/images/examples/04_densities.png)

```python
--8<-- "examples/04_densities.py"
```
