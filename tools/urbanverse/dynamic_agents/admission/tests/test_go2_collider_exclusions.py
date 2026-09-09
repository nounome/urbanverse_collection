import numpy as np
import pytest
from urbanverse.dynamic_agents.navigation.mesh_go2_route import conservative_collider_mask


class Grid:
    h=w=20
    obstacles=[dict(instance_root='tree',path='/tree/mesh',bounds=[[3,4,0],[6,8,9]])]

    def pixels(self,xy):return np.asarray(xy)


def test_exclusion_includes_full_overhead_bounds():
    mask=conservative_collider_mask(Grid(),['tree'])
    assert mask[4:9,3:7].all()
    assert not mask[0,0]
    assert mask.sum()==20


def test_empty_preserves_default():
    assert not conservative_collider_mask(Grid(),[]).any()


def test_unknown_instance_fails_closed():
    with pytest.raises(ValueError,match='Unknown'):
        conservative_collider_mask(Grid(),['missing'])
