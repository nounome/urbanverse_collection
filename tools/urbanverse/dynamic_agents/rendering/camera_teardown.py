"""Detach outputs while scene objects and their deletion callbacks are alive."""


def detach_camera_outputs(scene):
    detached = 0
    for sensor in list(scene.sensors.values()):
        registry = getattr(sensor, '_rep_registry', None)
        if registry is None:
            continue
        for name, annotators in list(registry.items()):
            for annotator, path in zip(annotators, sensor._render_product_paths):
                annotator.detach([path])
                detached += 1
            # Camera.__del__ must not detach these outputs a second time after
            # ManagerBasedEnv.close deletes scene/robot views and managers.
            del registry[name]
    return dict(detached_annotator_count=detached,
                scope='before env.close; no runtime state changes')
