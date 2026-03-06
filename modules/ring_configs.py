# Ring configurations for walking and driving isochrone maps (minutes 5–20).
# Walking and driving modules import WALK_RING_CONFIGS and DRIVE_RING_CONFIGS.

def _anchor_for(anchors, m):
    """Return the smallest anchor >= m (ceiling); if m > all anchors, return max anchor."""
    sorted_a = sorted(anchors)
    for a in sorted_a:
        if m <= a:
            return a
    return sorted_a[-1]


def _interp_rings(anchors, m):
    """Use the relevant anchor for m: 5→5, 6–10→10, 11–15→15, 16–20→20. No interpolation."""
    key = m if m in anchors else _anchor_for(anchors, m)
    return anchors[key].copy()


# ── Walking (5 km/h) ─────────────────────────────────────────────
_WALK_ANCHORS = {
    5: {
        "rings": [(83, "1 min\n0.08 km"), (250, "3 min\n0.25 km"), (400, "5 min\n0.40 km")],
        "shade_r": 400,
        "map_extent": 600,
        "walk_dist": 600,
    },
    10: {
        "rings": [(250, "3 min\n0.25 km"), (500, "6 min\n0.50 km"), (750, "10 min\n0.75 km")],
        "shade_r": 750,
        "map_extent": 875,
        "walk_dist": 900,
    },
    15: {
        "rings": [(375, "5 min\n0.375 km"), (750, "10 min\n0.75 km"), (1125, "15 min\n1.125 km")],
        "shade_r": 1125,
        "map_extent": 1300,
        "walk_dist": 1400,
    },
    20: {
        "rings": [(375, "5 min\n0.375 km"), (750, "10 min\n0.75 km"), (1500, "20 min\n1.5 km")],
        "shade_r": 1500,
        "map_extent": 1750,
        "walk_dist": 2000,
    },
}

WALK_RING_CONFIGS = {m: _interp_rings(_WALK_ANCHORS, m) for m in range(5, 21)}


# ── Driving (35 km/h) ────────────────────────────────────────────
_DRIVE_ANCHORS = {
    5: {
        "rings": [(83, "1.5 MINS\n0.083 KM"), (250, "3 MINS\n0.25 KM"), (400, "5 MINS\n0.40 KM")],
        "max_radius": 400,
        "map_extent": 600,
        "graph_dist": 800,
    },
    10: {
        "rings": [(250, "3 MINS\n0.25 KM"), (500, "6 MINS\n0.50 KM"), (750, "10 MINS\n0.75 KM")],
        "max_radius": 750,
        "map_extent": 875,
        "graph_dist": 1500,
    },
    15: {
        "rings": [(375, "1.5 MINS\n0.375 KM"), (750, "3 MINS\n0.75 KM"), (1125, "4.5 MINS\n1.125 KM")],
        "max_radius": 1300,
        "map_extent": 1400,
        "graph_dist": 2500,
    },
    20: {
        "rings": [(500, "2 MINS\n0.5 KM"), (1000, "4 MINS\n1 KM"), (1500, "20 MINS\n1.5 KM")],
        "max_radius": 1500,
        "map_extent": 1750,
        "graph_dist": 3000,
    },
}

DRIVE_RING_CONFIGS = {m: _interp_rings(_DRIVE_ANCHORS, m) for m in range(5, 21)}
