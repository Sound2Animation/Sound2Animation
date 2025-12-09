"""Material properties for RealImpact objects (density only, physics params in realimpact_sim.py)"""

MATERIALS = {
    "ceramic": {"density": 2400},
    "iron": {"density": 7870},
    "wood": {"density": 600},
    "plastic": {"density": 1200},
    "glass": {"density": 2500},
    "steel": {"density": 7850},
    "aluminum": {"density": 2700},
    "shell": {"density": 2700},
    "default": {"density": 1000},
}


def parse_material_from_name(object_name: str) -> tuple[str, dict]:
    """Parse material type from RealImpact object name based on dataset naming"""
    name_lower = object_name.lower()

    # Check explicit material keywords first
    if "plastic" in name_lower:
        return "plastic", MATERIALS["plastic"]
    elif "iron" in name_lower:
        return "iron", MATERIALS["iron"]
    elif "metal" in name_lower:
        return "iron", MATERIALS["iron"]
    elif "wood" in name_lower:
        return "wood", MATERIALS["wood"]
    elif "glass" in name_lower:
        return "glass", MATERIALS["glass"]
    elif "shell" in name_lower:
        return "shell", MATERIALS["shell"]
    elif "ceramic" in name_lower:
        return "ceramic", MATERIALS["ceramic"]
    elif "steel" in name_lower:
        return "steel", MATERIALS["steel"]
    elif "aluminum" in name_lower:
        return "aluminum", MATERIALS["aluminum"]
    # Generic shapes without material - default to ceramic
    elif any(s in name_lower for s in ["bowl", "pot", "planter", "cup", "mug", "pitcher", "swan", "flowerpot"]):
        return "ceramic", MATERIALS["ceramic"]
    elif any(s in name_lower for s in ["pan", "skillet", "plate", "mortar", "spoon", "spatula", "ladle"]):
        return "iron", MATERIALS["iron"]
    elif "goblet" in name_lower:
        return "glass", MATERIALS["glass"]
    elif any(s in name_lower for s in ["frisbee", "scoop", "bin"]):
        return "plastic", MATERIALS["plastic"]
    else:
        return "default", MATERIALS["default"]


def compute_mass(volume: float, density: float) -> float:
    """Compute mass from volume and density"""
    return volume * density
