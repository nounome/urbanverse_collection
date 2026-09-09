"""Explicit semantic aliases across CraftBench naming generations.

No substring car/building guessing. Unknown and large facility roots stay
protected. Callers must also verify inventory identity and needed map overlap.
"""
import re

ALIASES = {
    'sedan': 'static_vehicle', 'coupe_car': 'static_vehicle', 'sports_car': 'static_vehicle',
    'convertible_car': 'static_vehicle', 'suv': 'static_vehicle', 'pickup': 'static_vehicle',
    'hatchback': 'static_vehicle', 'police_car': 'static_vehicle', 'electric_cars': 'static_vehicle',
    'bus': 'static_vehicle', 'bicycle': 'static_two_wheeler', 'scooter': 'static_two_wheeler',
    'motorcycle': 'static_two_wheeler', 'bollard': 'bollard', 'shrub': 'shrub',
}


def cleanup_category(root):
    if '/' in root:return None
    value = root
    for prefix in ('vehicle_private_vehicle_', 'vehicle_emergency_vehicle_', 'vehicle_public_vehicle_',
                   'nature_vegetation_', 'amenity_transportation_amenity_', 'barrier_access_control_barrier_'):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    match = re.fullmatch(r'(.+)_([0-9a-f]{20,40})(?:_\d+)?', value)
    return ALIASES.get(match.group(1)) if match else None
