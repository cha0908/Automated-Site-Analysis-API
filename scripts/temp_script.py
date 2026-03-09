import osmnx as ox
# One request for multiple natural values
gdf = ox.features_from_point(
    (22.294277, 114.174778),  # e.g. K11
    dist=600,
    tags={"natural": ["strait", "bay", "water", "coastline"]},
)
print(gdf.shape)
print(gdf["natural"].value_counts() if "natural" in gdf.columns else "no natural col")