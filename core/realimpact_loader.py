"""
RealImpact Dataset Loader
"""

import numpy as np
import trimesh
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class RealImpactObject:
    name: str
    mesh: trimesh.Trimesh
    vertices: np.ndarray
    faces: np.ndarray
    hit_positions: np.ndarray
    hit_vertex_ids: np.ndarray
    listener_positions: np.ndarray
    angles: np.ndarray
    distances: np.ndarray
    mic_ids: np.ndarray
    audio_data: np.ndarray = None
    sample_rate: int = 48000
    _index: dict = field(default_factory=dict, repr=False)

    def build_index(self):
        """Build lookup index: (vertex_id, angle, distance) -> sample_indices"""
        for i in range(len(self.hit_vertex_ids)):
            key = (int(self.hit_vertex_ids[i]), int(self.angles[i]), int(self.distances[i]))
            if key not in self._index:
                self._index[key] = []
            self._index[key].append(i)

    def get_unique_hit_info(self) -> tuple[np.ndarray, np.ndarray]:
        """Get unique hit vertex IDs and their 3D positions"""
        unique_ids = np.unique(self.hit_vertex_ids)
        unique_positions = []
        for vid in unique_ids:
            mask = self.hit_vertex_ids == vid
            pos = self.hit_positions[mask][0]
            unique_positions.append(pos)
        return unique_ids, np.array(unique_positions, dtype=np.float32)

    def find_nearest_hit_vertex(self, position: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find k nearest hit vertices to a 3D position.
        Returns (vertex_ids, distances, weights) for interpolation.
        """
        unique_ids, unique_positions = self.get_unique_hit_info()
        dists = np.linalg.norm(unique_positions - position, axis=1)
        indices = np.argsort(dists)[:k]

        nearest_ids = unique_ids[indices]
        nearest_dists = dists[indices]

        if k == 1:
            weights = np.array([1.0])
        else:
            inv_dists = 1.0 / (nearest_dists + 1e-6)
            weights = inv_dists / inv_dists.sum()

        return nearest_ids, nearest_dists, weights

    def query_sound(self, hit_position: np.ndarray, listener_position: np.ndarray,
                    interpolate: bool = False) -> tuple[np.ndarray, float]:
        """
        Query sound sample by hit position and listener position.
        Returns (audio_samples, total_weight)
        """
        if not self._index:
            self.build_index()

        rel_pos = listener_position - hit_position
        distance = np.linalg.norm(rel_pos) * 1000  # convert to mm
        angle = np.degrees(np.arctan2(np.sqrt(rel_pos[0]**2 + rel_pos[2]**2), rel_pos[1]))
        angle = np.clip(angle, 0, 180)

        available_angles = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180])
        closest_angle = available_angles[np.argmin(np.abs(available_angles - angle))]

        available_dists = np.array([0, 333, 666, 1000])
        closest_dist = available_dists[np.argmin(np.abs(available_dists - distance))]

        k = 3 if interpolate else 1
        nearest_ids, _, weights = self.find_nearest_hit_vertex(hit_position, k=k)

        if self.audio_data is None:
            return None, 0.0

        mixed_audio = None
        for i, vid in enumerate(nearest_ids):
            key = (int(vid), int(closest_angle), int(closest_dist))
            if key in self._index:
                idx = self._index[key][0]
                audio = self.audio_data[idx].astype(np.float32)
                if mixed_audio is None:
                    mixed_audio = audio * weights[i]
                else:
                    min_len = min(len(mixed_audio), len(audio))
                    mixed_audio[:min_len] += audio[:min_len] * weights[i]

        return mixed_audio, weights.sum() if mixed_audio is not None else 0.0


class RealImpactLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def list_objects(self) -> list[str]:
        objects = []
        for item in sorted(self.dataset_path.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                preprocessed = item / 'preprocessed'
                if preprocessed.exists() and (preprocessed / 'transformed.obj').exists():
                    objects.append(item.name)
        return objects

    def load_object(self, object_name: str, load_audio: bool = True) -> RealImpactObject:
        obj_path = self.dataset_path / object_name / 'preprocessed'

        mesh = trimesh.load(obj_path / 'transformed.obj', process=False, maintain_order=True)
        hit_positions = np.load(obj_path / 'vertexXYZ.npy').astype(np.float32)
        hit_vertex_ids = np.load(obj_path / 'vertexID.npy').astype(np.int32)
        listener_positions = np.load(obj_path / 'listenerXYZ.npy').astype(np.float32)
        angles = np.load(obj_path / 'angle.npy')
        distances = np.load(obj_path / 'distance.npy')
        mic_ids = np.load(obj_path / 'micID.npy')

        audio_data = None
        if load_audio and (obj_path / 'deconvolved_0db.npy').exists():
            print(f"Loading audio data for {object_name}...")
            audio_data = np.load(obj_path / 'deconvolved_0db.npy')
            print(f"  Audio shape: {audio_data.shape}")

        obj = RealImpactObject(
            name=object_name,
            mesh=mesh,
            vertices=mesh.vertices.astype(np.float32),
            faces=mesh.faces.astype(np.int32),
            hit_positions=hit_positions,
            hit_vertex_ids=hit_vertex_ids,
            listener_positions=listener_positions,
            angles=angles,
            distances=distances,
            mic_ids=mic_ids,
            audio_data=audio_data
        )
        obj.build_index()
        return obj
