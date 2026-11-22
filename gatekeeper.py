"""Mandatory image-type filter using the supplied trained fundus checkpoint."""
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import tensorflow as tf


class FundusRejected(ValueError):
    def __init__(self, result):
        super().__init__('The fundus gatekeeper did not accept this image. Upload a clear retinal fundus photograph.')
        self.result = result


class GatekeeperUnavailable(RuntimeError):
    """A missing or broken gate must never silently permit disease analysis."""


def prepare_gate_input(image, mode):
    array = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.NEAREST), dtype=np.float32)
    if mode == 'rgb_0_1':
        array = array / 255.0
    elif mode == 'mobilenet_v2':
        array = array / 127.5 - 1.0
    elif mode != 'rgb_0_255':
        raise ValueError('Unsupported gatekeeper preprocessing.')
    return array[None, ...]


class FundusGatekeeper:
    def __init__(self, root=None):
        self.root = Path(root) if root else Path(__file__).resolve().parent
        self.model = None
        self.config = None
        self.lock = threading.RLock()

    def _load(self):
        if self.config is None:
            spec = json.loads((self.root / 'gatekeeper_config.json').read_text())
            if spec.get('fundus_class_index') not in {0, 1}:
                raise ValueError('Invalid fundus class index.')
            threshold = float(spec['accept_threshold'])
            if not np.isfinite(threshold) or not 0.5 <= threshold < 1:
                raise ValueError('Gate acceptance threshold must be at least 0.5 and less than 1.')
            if spec.get('preprocessing') not in {'rgb_0_1', 'rgb_0_255', 'mobilenet_v2'}:
                raise ValueError('Invalid gate preprocessing.')
            self.config = spec
        if self.model is None:
            model = tf.keras.models.load_model(self.root / self.config['file'], compile=False, safe_mode=True)
            if model.input_shape != (None, 224, 224, 3) or model.output_shape != (None, 1):
                raise ValueError('Unexpected gatekeeper input/output shape.')
            self.model = model

    def check(self, image):
        try:
            with self.lock:
                self._load()
                batch = prepare_gate_input(image, self.config['preprocessing'])
                values = np.asarray(self.model(batch, training=False))
                if values.shape != (1, 1):
                    raise ValueError('Gatekeeper must return one sigmoid score.')
                raw = float(values[0, 0])
                if not np.isfinite(raw) or not 0 <= raw <= 1:
                    raise ValueError('Gatekeeper returned an invalid score.')
                fundus = 1.0 - raw if self.config['fundus_class_index'] == 0 else raw
                threshold = float(self.config['accept_threshold'])
                return {
                    'accepted': fundus > threshold,  # Exact ties are not accepted.
                    'fundus_score': fundus, 'raw_output': raw, 'accept_threshold': threshold,
                    'preprocessing': self.config['preprocessing'],
                    'contract_status': self.config['contract_status'],
                    'note': 'Image-type screening only; passing does not establish image quality, disease status or clinical validity.'
                }
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception('Fundus gatekeeper failed: %s', exc)
            raise GatekeeperUnavailable('The fundus image check is unavailable. No disease analysis was performed. Check the server log and model files.') from exc

    def require_fundus(self, image):
        result = self.check(image)
        if not result['accepted']:
            raise FundusRejected(result)
        return result
