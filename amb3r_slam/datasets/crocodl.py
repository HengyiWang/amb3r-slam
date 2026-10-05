"""CroCoDL, prepared by VidMap like LaMAR but with ground truth per session."""

import os

from amb3r_slam.datasets.common import register
from amb3r_slam.datasets.lamar import Lamar, posed_sequence


@register
class Crocodl(Lamar):
    """CroCoDL ARCHE locations: short phone captures at 5 Hz."""
    name = 'crocodl'
    scenes = ('ARCHE_B3', 'ARCHE_B5', 'ARCHE_D2', 'ARCHE_GRANDE')
    fps = 5

    def testset(self, scenes):
        import pycolmap
        import yaml
        for scene in scenes:
            testset = os.path.join(self.root, 'testsets', f'ios-{scene}', 'complete.yaml')
            if not os.path.isfile(testset):
                print(f'[skip] {scene}: no complete.yaml (not prepared yet)', flush=True)
                continue
            sel = yaml.safe_load(open(testset))
            recs = {}
            for name in sorted(sel):
                session = name.rsplit('-', 1)[0]
                sess_dir = os.path.join(self.root, 'datasets', scene, session)
                if not os.path.isdir(sess_dir):
                    print(f'[skip] {scene}/{name}: session not prepared', flush=True)
                    continue
                if session not in recs:
                    recs[session] = pycolmap.Reconstruction(os.path.join(sess_dir, 'rec'))
                try:
                    got = posed_sequence(recs[session], sel[name],
                                         os.path.join(sess_dir, 'images'))
                except KeyError as e:
                    print(f'[skip] {scene}/{name}: image id {e} missing from rec', flush=True)
                    continue
                if got is None:
                    print(f'[skip] {scene}/{name}: fewer than 4 posed frames', flush=True)
                    continue
                yield (scene, name) + got
