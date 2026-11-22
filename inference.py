"""Explicit checkpoint-specific inference; no guessed thresholds or label flipping."""
import json
from pathlib import Path
import threading

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from vessel_pipeline import segment_vessels, measure_width_proxy, retinal_field_mask
from gatekeeper import FundusGatekeeper
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
Image.MAX_IMAGE_PIXELS = 16_000_000


def load_rgb(source):
    """Decode real image contents and normalize EXIF orientation, not pixel scale."""
    try:
        with Image.open(source) as image:
            if image.format not in {"JPEG", "PNG", "TIFF", "BMP", "WEBP"}:
                raise ValueError("Please upload a JPEG, PNG, TIFF, BMP or WebP image.")
            if image.width * image.height > Image.MAX_IMAGE_PIXELS:
                raise ValueError("Image is too large (maximum 16 megapixels).")
            image.load()
            rgb = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc
    if min(rgb.size) < 64:
        raise ValueError("Image must be at least 64 by 64 pixels.")
    if float(np.asarray(rgb).astype(np.float32).std(axis=(0, 1)).max()) < 1.0:
        raise ValueError("The image is blank or has insufficient variation.")
    return rgb


def prepare_image(image, model, preprocessing):
    """Nearest-neighbour RGB resize matches the supplied Keras generator defaults."""
    height, width = model.input_shape[1:3]
    if height is None or width is None or model.input_shape[-1] != 3:
        raise ValueError("Expected a fixed-size RGB image classifier.")
    array = np.asarray(image.resize((width, height), Image.Resampling.NEAREST), dtype=np.float32)
    if preprocessing == "rgb_0_1":
        array = array / 255.0
    elif preprocessing != "rgb_0_255":
        raise ValueError(f"Unsupported preprocessing mode: {preprocessing}")
    return array[np.newaxis, ...]


def scalar_score(prediction):
    values = np.asarray(prediction)
    if values.shape != (1, 1):
        raise ValueError(f"Expected one sigmoid output; got {values.shape}.")
    value = float(values[0, 0])
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Model returned an invalid score.")
    return value


class GradCAM:
    """Explain the positive-class logit without mutating the model's sigmoid.

    Supports the supplied flat Functional and nested Sequential classifiers.
    Explicit inference mode prevents Dropout/BatchNorm changing explanations.
    """
    def __init__(self, model):
        self.model = model
        self.final = model.layers[-1]
        if not isinstance(self.final, tf.keras.layers.Dense) or self.final.units != 1:
            raise ValueError("Grad-CAM requires a single-output Dense classifier.")
        self.inner = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
        base = self.inner if self.inner is not None else model
        # Use the final spatial activation, not the pre-BatchNorm top convolution.
        # All three supplied EfficientNet checkpoints expose this same layer.
        conv = next((l for l in reversed(base.layers) if l.name == 'top_activation'), None)
        if conv is None:
            conv = next((l for l in reversed(base.layers)
                         if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))), None)
        if conv is None:
            raise ValueError("No spatial convolution layer found.")
        self.layer_name = conv.name
        self.view = tf.keras.Model(
            base.inputs,
            [conv.output, base.outputs[0] if self.inner is not None else self.final.input],
        )

    def __call__(self, array, target="positive"):
        if target not in {"positive", "negative"}:
            raise ValueError("Grad-CAM target must be positive or negative.")
        batch = tf.convert_to_tensor(array, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(batch)
            if self.inner is None:
                conv, features = self.view(batch, training=False)
            else:
                features = batch
                conv = None
                for layer in self.model.layers:
                    if isinstance(layer, tf.keras.layers.InputLayer):
                        continue
                    if layer is self.final:
                        break
                    if layer is self.inner:
                        conv, features = self.view(features, training=False)
                    else:
                        features = layer(features, training=False)
            # Use the logit to avoid vanishing sigmoid gradients at extreme scores.
            logits = tf.matmul(features, self.final.kernel)
            if self.final.use_bias:
                logits = logits + self.final.bias
            target_channel = logits[:, 0] if target == "positive" else -logits[:, 0]
        gradients = tape.gradient(target_channel, conv)
        if gradients is None:
            raise ValueError("No gradient path from classifier to spatial features.")
        weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv[0] * weights, axis=-1)
        heatmap = tf.maximum(heatmap, 0).numpy()
        if not np.isfinite(heatmap).all():
            raise ValueError("Grad-CAM produced non-finite values.")
        peak = float(heatmap.max())
        return heatmap / peak if peak > 1e-12 else np.zeros_like(heatmap)


def save_overlay(image, heatmap, destination):
    """Only positive attribution adds colour; zero-activation pixels stay unchanged."""
    if not np.isfinite(heatmap).all() or heatmap.max() <= 0:
        return False
    preview = image.copy()
    preview.thumbnail((1600, 1600))
    base = cv2.cvtColor(np.asarray(preview), cv2.COLOR_RGB2BGR)
    resized = np.clip(cv2.resize(heatmap, preview.size, interpolation=cv2.INTER_LINEAR), 0, 1)
    colour = cv2.applyColorMap(np.uint8(resized * 255), cv2.COLORMAP_TURBO)
    alpha = (.65 * resized)[..., None]
    overlay = np.uint8(np.clip(base * (1 - alpha) + colour * alpha, 0, 255))
    if not cv2.imwrite(str(destination), overlay):
        raise OSError("Could not save heatmap.")
    return True


class Predictor:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.config = json.loads((self.root / "model_config.json").read_text())
        self.models = {}
        self.explainers = {}
        self.unet = None
        self.gatekeeper = FundusGatekeeper(self.root)
        self.lock = threading.RLock()

    def get_model(self, key):
        with self.lock:
            if key not in self.models:
                path = self.root / self.config[key]["file"]
                # Optimizer/metric restoration is unnecessary for inference.
                model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
                if model.output_shape[-1] != 1:
                    raise ValueError(f"{key}: expected one classifier output.")
                self.models[key] = model
            return self.models[key]

    def analyze(self, image, output_dir, request_id, explain=True, cam_target="positive"):
        if cam_target not in {"positive", "negative"}:
            raise ValueError("Unknown explanation target.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = {}
        # Serialise heavy inference so overlapping uploads do not exhaust CPU/RAM.
        with self.lock:
            for key, spec in self.config.items():
                model = self.get_model(key)
                batch = prepare_image(image, model, spec["preprocessing"])
                score = scalar_score(model(batch, training=False))
                result = {"score": score, "display_score": f"{score * 100:.1f}%",
                          "positive_label": spec["positive_label"], "map_filename": None,
                          "explanation_status": "not_requested", "cam_target": cam_target,
                          "cam_layer": None, "heatmap_filename": None,
                          "cam_resolution": None, "cam_outside_field_percent": None}
                if explain:
                    try:
                        if key not in self.explainers:
                            self.explainers[key] = GradCAM(model)
                        heatmap = self.explainers[key](batch, target=cam_target)
                        result["cam_layer"] = self.explainers[key].layer_name
                        result["cam_resolution"] = list(heatmap.shape)
                        name = f"{request_id}_{key}.png"
                        if save_overlay(image, heatmap, Path(output_dir) / name):
                            preview = image.copy()
                            preview.thumbnail((768, 768))
                            field = retinal_field_mask(np.asarray(preview))
                            activation = cv2.resize(heatmap, preview.size, interpolation=cv2.INTER_LINEAR)
                            mass = float(activation.sum())
                            if field.any() and mass > 0:
                                result["cam_outside_field_percent"] = 100.0 * float(activation[~field].sum()) / mass
                            result["map_filename"] = name
                            result["explanation_status"] = "available"
                            raw_name = f"{request_id}_{key}_activation.png"
                            Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(
                                (300, 300), Image.Resampling.NEAREST).save(Path(output_dir) / raw_name)
                            result["heatmap_filename"] = raw_name
                        else:
                            result["explanation_status"] = "no_target_activation"
                    except (ValueError, RuntimeError, OSError, tf.errors.OpError) as exc:
                        # A failed explanation must not discard a valid prediction.
                        import logging
                        logging.getLogger(__name__).exception("%s heatmap failed: %s", key, exc)
                        result["explanation_status"] = "unavailable"
                results[key] = result
        return results


    def analyze_vessels(self, image, output_dir, request_id):
        """Run the supplied U-Net on bounded-size overlapping RGB patches."""
        with self.lock:
            if self.unet is None:
                self.unet = tf.keras.models.load_model(
                    self.root / "unet_vessel_segmentation.keras", compile=False, safe_mode=True)
            working = image.copy()
            working.thumbnail((768, 768), Image.Resampling.LANCZOS)
            rgb = np.asarray(working)
            probability = segment_vessels(self.unet, rgb)
            field = retinal_field_mask(rgb)
            field_count = int(field.sum())
            if field_count == 0:
                return {"status": "unavailable", "reason": "No usable image field was found."}
            binary = (probability >= .5) & field
            vessel_count = int(binary.sum())
            mask_name = f"{request_id}_vessels.png"
            overlay_name = f"{request_id}_vessels_overlay.png"
            probability_name = f"{request_id}_vessel_probability.png"
            Image.fromarray(np.uint8(binary) * 255).save(Path(output_dir) / mask_name)
            Image.fromarray(np.uint8(np.clip(probability, 0, 1) * field * 255)).save(
                Path(output_dir) / probability_name)
            overlay = rgb.copy()
            overlay[binary] = np.uint8(.3 * rgb[binary] + .7 * np.array([30, 230, 183]))
            Image.fromarray(overlay).save(Path(output_dir) / overlay_name)
            skeleton = cv2.ximgproc.thinning(np.uint8(binary) * 255)
            # Retain the existing exploratory width routine without calling it clinical AVR.
            width = measure_width_proxy(rgb, probability * field, 'kmeans')
            return {
                "status": "available" if vessel_count else "no_vessels_detected",
                "mask_filename": mask_name, "overlay_filename": overlay_name,
                "probability_filename": probability_name,
                "processed_size": list(working.size), "original_size": list(image.size),
                "field_pixels": field_count, "vessel_pixels": vessel_count,
                "vessel_area_percent": 100.0 * vessel_count / field_count,
                "skeleton_pixels": int(np.count_nonzero(skeleton)),
                "segmentation_threshold": .5, "width_proxy": width,
                "note": "Area is measured inside a brightness-based field estimate at the displayed resolution. "
                        "It has no validated normal/abnormal cutoff; the mask does not distinguish arteries from veins."
            }

    def analyze_complete(self, image, output_dir, request_id, cam_target="positive"):
        """Both existing streams, with independent failure reporting."""
        with self.lock:
            # Mandatory gate is first. Rejected/unavailable gates never reach either stream.
            gate = self.gatekeeper.require_fundus(image)
            scores = self.analyze(image, output_dir, request_id, cam_target=cam_target)
            try:
                vessels = self.analyze_vessels(image, output_dir, request_id)
            except (ValueError, RuntimeError, OSError, tf.errors.OpError) as exc:
                import logging
                logging.getLogger(__name__).exception("U-Net analysis failed: %s", exc)
                for path in Path(output_dir).glob(f"{request_id}_vessel*.png"):
                    path.unlink(missing_ok=True)
                vessels = {"status": "unavailable", "reason": "Vessel segmentation could not be completed."}
            return {"scores": scores, "vessels": vessels, "gatekeeper": gate}
