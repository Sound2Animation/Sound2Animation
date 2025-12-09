"""
RealImpact Object Simulation with Newton Rigid Body Physics
"""

from __future__ import annotations

import numpy as np
import argparse
import time
import json
import re
from pathlib import Path

import warp as wp
import newton
import newton.examples

from core.realimpact_loader import RealImpactLoader, RealImpactObject
from core.sound_synthesizer import SoundSynthesizer
from core.material_properties import parse_material_from_name, MATERIALS

wp.init()


# Physics parameters for mesh collision (based on Newton examples)
# ke: contact stiffness, kd: contact damping, kf: friction stiffness
# mu: friction coefficient, restitution: bounciness (0-1)
# ke: contact stiffness, kd: contact damping
MATERIAL_PHYSICS = {
    "ceramic": {"ke": 1e4, "kd": 5.0, "kf": 0.0, "mu": 0.3, "restitution": 0.7},
    "iron": {"ke": 1e4, "kd": 5.0, "kf": 0.0, "mu": 0.4, "restitution": 0.8},
    "wood": {"ke": 1e4, "kd": 10.0, "kf": 0.0, "mu": 0.5, "restitution": 0.4},
    "plastic": {"ke": 1e4, "kd": 5.0, "kf": 0.0, "mu": 0.3, "restitution": 0.6},
    "glass": {"ke": 1e4, "kd": 5.0, "kf": 0.0, "mu": 0.3, "restitution": 0.75},
    "default": {"ke": 1e4, "kd": 5.0, "kf": 0.0, "mu": 0.3, "restitution": 0.6},
}

OBJECT_TYPE_PATTERNS = [
    (r'bowl', 'bowl'), (r'plate', 'plate'), (r'cup', 'cup'), (r'mug', 'mug'),
    (r'pan', 'pan'), (r'skillet', 'skillet'), (r'pot', 'pot'), (r'lid', 'lid'),
    (r'spoon', 'spoon'), (r'fork', 'fork'), (r'knife', 'knife'), (r'scoop', 'scoop'),
    (r'goblet', 'goblet'), (r'vase', 'vase'), (r'jar', 'jar'), (r'bottle', 'bottle'),
    (r'slab', 'slab'), (r'block', 'block'), (r'frisbee', 'frisbee'), (r'tray', 'tray'),
]

def parse_object_type(name: str) -> str:
    name_lower = name.lower()
    for pattern, obj_type in OBJECT_TYPE_PATTERNS:
        if re.search(pattern, name_lower):
            return obj_type
    return "object"


class NewtonRigidBodySim:
    def __init__(
        self,
        obj: RealImpactObject,
        drop_height: float = 0.5,
        device: str = "cuda:0",
        viewer=None,
        use_convex_hull: bool = False,
        target_collision_faces: int | None = 2000,
        max_contacts_per_pair: int = 2048,
        collision_margin: float = 0.05,  # Larger margin for better contact detection
    ):
        self.obj = obj
        self.device = device

        self.material_name, mat_props = parse_material_from_name(obj.name)
        self.density = mat_props["density"]
        phys = MATERIAL_PHYSICS.get(self.material_name, MATERIAL_PHYSICS["default"])

        print(f"  Material: {self.material_name}")
        print(f"  Density: {self.density} kg/m³")

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = phys["ke"]
        builder.default_shape_cfg.kd = phys["kd"]
        builder.default_shape_cfg.kf = phys["kf"]
        builder.default_shape_cfg.mu = phys["mu"]
        builder.default_shape_cfg.restitution = phys["restitution"]
        builder.default_shape_cfg.density = self.density

        print(f"  Restitution: {phys['restitution']}")

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            ke=phys["ke"], kd=phys["kd"], kf=phys["kf"], mu=phys["mu"],
            restitution=0.8  # Moderate bounce
        )
        builder.add_ground_plane(cfg=ground_cfg)

        # Use the original mesh for collisions (matches Newton examples) to keep concavities.
        if use_convex_hull:
            print("  Using convex hull for collision mesh (requested)")
            collision_mesh = obj.mesh.convex_hull
        else:
            collision_mesh = obj.mesh.copy()

        if target_collision_faces and len(collision_mesh.faces) > target_collision_faces:
            try:
                # trimesh expects reduction ratio (0-1), not face count
                reduction_ratio = 1.0 - (target_collision_faces / len(collision_mesh.faces))
                collision_mesh = collision_mesh.simplify_quadric_decimation(reduction_ratio)
            except BaseException as exc:
                print(f"  Warning: mesh simplification failed ({exc}), using original mesh")

        # Original mesh is already Z-up (same as Newton), no conversion needed
        vertices = collision_mesh.vertices.astype(np.float32)
        faces = collision_mesh.faces.flatten().astype(np.int32)
        mesh = newton.Mesh(vertices, faces)
        print(f"  Collision mesh: {len(collision_mesh.vertices)} vertices, {len(collision_mesh.faces)} faces")

        # Newton uses Z-up coordinate system
        init_pos = (0.0, 0.0, drop_height)
        # Random rotation: uniformly sample quaternion
        u1, u2, u3 = np.random.random(3)
        init_rot = wp.quat(
            np.sqrt(1-u1) * np.sin(2*np.pi*u2),
            np.sqrt(1-u1) * np.cos(2*np.pi*u2),
            np.sqrt(u1) * np.sin(2*np.pi*u3),
            np.sqrt(u1) * np.cos(2*np.pi*u3),
        )
        xform = wp.transform(init_pos, init_rot)

        # Store initial transform for export (w,x,y,z format)
        self.initial_position = np.array(init_pos, dtype=np.float32)
        self.initial_rotation = np.array([float(init_rot[3]), float(init_rot[0]), float(init_rot[1]), float(init_rot[2])], dtype=np.float32)
        self.drop_height = drop_height
        self.object_type = parse_object_type(obj.name)

        self.body_id = builder.add_body(xform=xform)
        builder.add_joint_free(self.body_id)
        builder.add_shape_mesh(body=self.body_id, mesh=mesh)
        print(f"  Mesh vertices: {len(obj.vertices)}, faces: {len(obj.faces)}")

        self.model = builder.finalize(device=device)
        shape_body = self.model.shape_body.numpy()
        self.object_shape_ids = {i for i, b in enumerate(shape_body) if b == self.body_id}

        # Note: mass is calculated from collision mesh volume, may differ from original

        # Set model-level soft contact parameters (from Newton official examples)
        self.model.soft_contact_ke = phys["ke"]
        self.model.soft_contact_kd = phys["kd"]
        self.model.soft_contact_kf = phys["kf"]
        self.model.soft_contact_mu = phys["mu"]
        self.model.soft_contact_restitution = phys["restitution"]

        # Use XPBD solver (like Newton official examples) for stable rigid body simulation
        # enable_restitution=True allows bouncing with proper energy decay
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=10,
            enable_restitution=True,
            rigid_contact_relaxation=0.8,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create collision pipeline (from Newton basic_shapes example)
        self.rigid_contact_margin = collision_margin
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            collision_pipeline_type="standard",
            rigid_contact_max_per_pair=max_contacts_per_pair,
            rigid_contact_margin=self.rigid_contact_margin,
        )

        # Match Newton official example: 100fps with 10 substeps = 1ms timestep
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 12  # 120fps / 12 = 0.69ms timestep (close to official 1ms)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.animation_frames = []
        self.collision_events = []
        self.prev_velocity = np.zeros(3)
        self.ground_level = 0.0
        self.last_collision_time = -1.0
        self.min_collision_interval = 0.05  # ~50ms minimum between collisions (20Hz max)

        bbox = obj.mesh.bounds
        self.obj_bottom = bbox[0][1]

        # Scale mass by material (collision mesh volume is underestimated)
        mass_scale = {
            "iron": 5.0, "steel": 5.0,
            "ceramic": 3.0, "glass": 3.0,
            "plastic": 2.0, "wood": 2.0, "rubber": 2.0,
        }.get(self.material_name, 2.0)
        raw_mass = self.model.body_mass.numpy()[0]
        self.mass = raw_mass * mass_scale
        self.model.body_mass.assign([self.mass])
        print(f"  Mass: {self.mass*1000:.1f} g (raw: {raw_mass*1000:.1f}g, scale: {mass_scale}x)")

        self.viewer = viewer
        if viewer:
            viewer.set_model(self.model)

        # Store contacts for rendering
        self.contacts = None

    def render(self):
        if self.viewer:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            if self.contacts:
                self.viewer.log_contacts(self.contacts, self.state_0)
            self.viewer.end_frame()

    def step(self) -> dict:
        collision_info = None
        last_contacts = None

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            contacts = self.model.collide(
                self.state_0,
                collision_pipeline=self.collision_pipeline,
            )
            self.solver.step(self.state_0, self.state_1, self.control, contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            last_contacts = contacts

        # Read contact data after all substeps complete
        self.contacts = last_contacts
        total_contact_count = int(last_contacts.rigid_contact_count.numpy()[0]) if last_contacts else 0

        body_q = self.state_0.body_q.numpy()[0]
        body_qd = self.state_0.body_qd.numpy()[0]

        pos = np.array([body_q[0], body_q[1], body_q[2]])
        rot = np.array([body_q[6], body_q[3], body_q[4], body_q[5]])  # (w, x, y, z)
        vel = np.array([body_qd[0], body_qd[1], body_qd[2]])

        self.animation_frames.append({
            'time': self.sim_time,
            'position': pos.copy(),
            'quaternion': rot.copy(),
            'velocity': vel.copy(),
            'contact': total_contact_count > 0
        })

        # Impulse from velocity change: J = m × |Δv|
        delta_v = vel - self.prev_velocity
        impulse = self.mass * np.linalg.norm(delta_v)

        # Trigger sound when in contact and impulse exceeds threshold
        time_since_last = self.sim_time - self.last_collision_time
        cooldown_ok = time_since_last >= self.min_collision_interval
        impulse_threshold = 0.05  # Minimum impulse to trigger sound (N·s)

        if total_contact_count > 0 and impulse > impulse_threshold and cooldown_ok:
            self.last_collision_time = self.sim_time
            contact_points = self._extract_contact_points(last_contacts)
            hit_point = contact_points.mean(axis=0) if len(contact_points) > 0 else pos.copy()
            collision_info = {
                'time': self.sim_time,
                'position': pos.copy(),
                'contact_point': hit_point.copy(),
                'impulse': impulse,
                'contact_count': total_contact_count,
            }
            self.collision_events.append(collision_info)

        self.prev_velocity = vel.copy()
        self.sim_time += self.frame_dt
        return collision_info

    def _extract_object_impulse(self, contacts) -> np.ndarray:
        """Extract total contact force acting on our object"""
        if contacts is None:
            return np.zeros(3, dtype=np.float32)

        count = int(contacts.rigid_contact_count.numpy()[0])
        if count == 0:
            return np.zeros(3, dtype=np.float32)

        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        forces = contacts.rigid_contact_force.numpy()[:count]

        total_force = np.zeros(3, dtype=np.float32)
        for i in range(count):
            if shape0[i] in self.object_shape_ids:
                total_force += forces[i]
            elif shape1[i] in self.object_shape_ids:
                total_force -= forces[i]  # Opposite direction for shape1

        return total_force

    def _extract_contact_points(self, contacts) -> np.ndarray:
        if contacts is None:
            return np.zeros((0, 3), dtype=np.float32)

        count = int(contacts.rigid_contact_count.numpy()[0])
        if count == 0:
            return np.zeros((0, 3), dtype=np.float32)

        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        pts0 = contacts.rigid_contact_point0.numpy()[:count]
        pts1 = contacts.rigid_contact_point1.numpy()[:count]

        contact_points = []
        for i in range(count):
            if shape0[i] in self.object_shape_ids:
                contact_points.append(pts0[i])
            elif shape1[i] in self.object_shape_ids:
                contact_points.append(pts1[i])

        if not contact_points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array(contact_points, dtype=np.float32)

    def export_animation(self, filepath: str):
        """Export animation in absolute format (position and rotation per frame)"""
        with open(filepath, 'w') as f:
            f.write("# RealImpact Animation (absolute coordinates)\n")
            f.write("# format: time tx ty tz qw qx qy qz\n")
            for frame in self.animation_frames:
                t = frame['time']
                p = frame['position']
                q = frame['quaternion']
                f.write(f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
        print(f"Exported animation: {filepath} ({len(self.animation_frames)} frames)")

    def export_metadata(self, filepath: str, duration: float):
        """Export metadata JSON with description and initial transform"""
        mass_g = float(self.mass * 1000)
        desc = f"A {self.material_name} {self.object_type} ({mass_g:.0f}g) drops from {self.drop_height:.1f}m height"

        meta = {
            "description": desc,
            "object_name": self.obj.name,
            "object_type": self.object_type,
            "material": self.material_name,
            "mass_g": round(mass_g, 1),
            "density_kg_m3": int(self.density),
            "initial_transform": {
                "position": [float(x) for x in self.initial_position],
                "rotation_quat": [float(x) for x in self.initial_rotation],
            },
            "drop_height_m": float(self.drop_height),
            "ground_level": float(self.ground_level),
            "duration_s": float(duration),
            "fps": int(self.fps),
            "num_frames": len(self.animation_frames),
            "collisions": len(self.collision_events),
        }

        with open(filepath, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Exported metadata: {filepath}")

    def synthesize_audio(self, duration: float, camera_pos: np.ndarray = None, interpolate: bool = False) -> np.ndarray:
        """Synthesize audio from collision events"""
        if camera_pos is None:
            camera_pos = np.array([0.5, 0.25, 0.5], dtype=np.float32)

        synth = SoundSynthesizer(sample_rate=self.obj.sample_rate)
        synth.init_recording(duration + 1.0)

        for event in self.collision_events:
            hit_pos = event.get('contact_point', event['position'])
            audio, _ = self.obj.query_sound(hit_pos, camera_pos, interpolate=interpolate)
            if audio is not None:
                synth.trigger_sound(audio, event['impulse'], time_offset=event['time'])

        return synth.get_recording()[:int(self.obj.sample_rate * duration)]

    def export_all(self, output_dir: str, name: str, duration: float, generate_blend: bool = True):
        """Export animation, audio, metadata and optionally Blender file"""
        import subprocess
        from scipy.io import wavfile

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        anim_path = output_path / f"{name}_anim.txt"
        self.export_animation(str(anim_path))

        meta_path = output_path / f"{name}_meta.json"
        self.export_metadata(str(meta_path), duration)

        audio = self.synthesize_audio(duration)
        audio_path = output_path / f"{name}_audio.wav"
        wavfile.write(str(audio_path), self.obj.sample_rate, audio)

        if generate_blend:
            subprocess.run([
                "blender", "--background", "--python", "blender_render.py",
                "--", "--object", self.obj.name, "--dataset", str(Path(self.obj.mesh_path).parent.parent.parent),
                "--animation", str(anim_path), "--output", str(output_path), "--preview"
            ], capture_output=True)

        return {"anim": str(anim_path), "audio": str(audio_path), "meta": str(meta_path)}


class BatchRigidBodySim:
    """Batch simulation of multiple objects in a single Newton scene"""

    def __init__(
        self,
        objects: list[RealImpactObject],
        drop_heights: list[float],
        spacing: float = 3.0,
        device: str = "cuda:0",
        target_collision_faces: int = 2000,
    ):
        self.objects = objects
        self.device = device
        self.n_objects = len(objects)
        grid_size = int(np.ceil(np.sqrt(self.n_objects)))

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        self.body_ids = []
        self.offsets = []
        self.masses = []
        self.object_shape_ids = []

        for i, obj in enumerate(objects):
            material_name, mat_props = parse_material_from_name(obj.name)
            density = mat_props["density"]
            phys = MATERIAL_PHYSICS.get(material_name, MATERIAL_PHYSICS["default"])

            builder.default_shape_cfg.ke = phys["ke"]
            builder.default_shape_cfg.kd = phys["kd"]
            builder.default_shape_cfg.mu = phys["mu"]
            builder.default_shape_cfg.restitution = phys["restitution"]
            builder.default_shape_cfg.density = density

            collision_mesh = obj.mesh.copy()
            if target_collision_faces and len(collision_mesh.faces) > target_collision_faces:
                try:
                    ratio = 1.0 - (target_collision_faces / len(collision_mesh.faces))
                    collision_mesh = collision_mesh.simplify_quadric_decimation(ratio)
                except:
                    pass

            vertices = collision_mesh.vertices.astype(np.float32)
            faces = collision_mesh.faces.flatten().astype(np.int32)
            mesh = newton.Mesh(vertices, faces)

            row, col = i // grid_size, i % grid_size
            offset = np.array([col * spacing, row * spacing, 0.0])
            self.offsets.append(offset)

            pos = (offset[0], offset[1], drop_heights[i])
            # Random rotation: uniformly sample quaternion
            u1, u2, u3 = np.random.random(3)
            init_rot = wp.quat(
                np.sqrt(1-u1) * np.sin(2*np.pi*u2),
                np.sqrt(1-u1) * np.cos(2*np.pi*u2),
                np.sqrt(u1) * np.sin(2*np.pi*u3),
                np.sqrt(u1) * np.cos(2*np.pi*u3),
            )
            xform = wp.transform(pos, init_rot)

            body_id = builder.add_body(xform=xform)
            builder.add_joint_free(body_id)
            builder.add_shape_mesh(body=body_id, mesh=mesh)
            self.body_ids.append(body_id)

        self.model = builder.finalize(device=device)

        shape_body = self.model.shape_body.numpy()
        for body_id in self.body_ids:
            shape_ids = {i for i, b in enumerate(shape_body) if b == body_id}
            self.object_shape_ids.append(shape_ids)

        for i, body_id in enumerate(self.body_ids):
            material_name, _ = parse_material_from_name(objects[i].name)
            mass_scale = {"iron": 5.0, "ceramic": 3.0, "glass": 3.0, "plastic": 2.0, "wood": 2.0}.get(material_name, 2.0)
            raw_mass = self.model.body_mass.numpy()[body_id]
            self.masses.append(raw_mass * mass_scale)

        all_masses = self.model.body_mass.numpy()
        for i, body_id in enumerate(self.body_ids):
            all_masses[body_id] = self.masses[i]
        self.model.body_mass.assign(all_masses)

        # Use XPBD solver (like Newton official examples) for stable rigid body simulation
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=10,
            enable_restitution=True,
            rigid_contact_relaxation=0.8,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model, collision_pipeline_type="standard",
            rigid_contact_max_per_pair=2048, rigid_contact_margin=0.05,
        )

        # Match Newton official example: ~1ms timestep
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 12  # 120fps / 12 = 0.69ms timestep
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.trajectories = [[] for _ in range(self.n_objects)]
        self.collision_events = [[] for _ in range(self.n_objects)]
        self.prev_velocities = [np.zeros(3) for _ in range(self.n_objects)]
        self.last_collision_times = [-1.0] * self.n_objects

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        all_q = self.state_0.body_q.numpy()
        all_qd = self.state_0.body_qd.numpy()

        contact_count = int(contacts.rigid_contact_count.numpy()[0]) if contacts else 0
        shape0 = contacts.rigid_contact_shape0.numpy()[:contact_count] if contact_count > 0 else []
        shape1 = contacts.rigid_contact_shape1.numpy()[:contact_count] if contact_count > 0 else []
        pts0 = contacts.rigid_contact_point0.numpy()[:contact_count] if contact_count > 0 else []
        pts1 = contacts.rigid_contact_point1.numpy()[:contact_count] if contact_count > 0 else []

        for i, body_id in enumerate(self.body_ids):
            body_q = all_q[body_id]
            body_qd = all_qd[body_id]

            # Normalize position by subtracting grid offset
            pos = np.array([body_q[0], body_q[1], body_q[2]]) - self.offsets[i]
            rot = np.array([body_q[6], body_q[3], body_q[4], body_q[5]])
            vel = np.array([body_qd[0], body_qd[1], body_qd[2]])

            # Extract contact points for this object
            contact_points = []
            for j in range(contact_count):
                if shape0[j] in self.object_shape_ids[i]:
                    contact_points.append(pts0[j] - self.offsets[i])
                elif shape1[j] in self.object_shape_ids[i]:
                    contact_points.append(pts1[j] - self.offsets[i])

            self.trajectories[i].append({
                'time': self.sim_time, 'position': pos.copy(), 'quaternion': rot.copy(),
                'velocity': vel.copy(), 'contact': len(contact_points) > 0
            })

            delta_v = vel - self.prev_velocities[i]
            impulse = self.masses[i] * np.linalg.norm(delta_v)
            time_since_last = self.sim_time - self.last_collision_times[i]

            if len(contact_points) > 0 and impulse > 0.05 and time_since_last >= 0.05:
                self.last_collision_times[i] = self.sim_time
                hit_point = np.mean(contact_points, axis=0) if contact_points else pos.copy()
                self.collision_events[i].append({
                    'time': self.sim_time, 'position': pos.copy(),
                    'contact_point': hit_point, 'impulse': impulse,
                })

            self.prev_velocities[i] = vel.copy()

        self.sim_time += self.frame_dt

    def get_results(self, obj_idx: int):
        """Get trajectory and collision events for object at index"""
        return self.trajectories[obj_idx], self.collision_events[obj_idx]

    def export_animation(self, obj_idx: int, filepath: str):
        """Export animation for object at index"""
        frames = self.trajectories[obj_idx]
        with open(filepath, 'w') as f:
            f.write("# RealImpact Animation (absolute coordinates)\n")
            f.write("# format: time tx ty tz qw qx qy qz\n")
            for frame in frames:
                t = frame['time']
                p = frame['position']
                q = frame['quaternion']
                f.write(f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    def synthesize_audio(self, obj_idx: int, duration: float, camera_pos: np.ndarray = None) -> np.ndarray:
        """Synthesize audio for object at index"""
        obj = self.objects[obj_idx]
        events = self.collision_events[obj_idx]
        if camera_pos is None:
            camera_pos = np.array([0.5, 0.25, 0.5], dtype=np.float32)

        synth = SoundSynthesizer(sample_rate=obj.sample_rate)
        synth.init_recording(duration + 1.0)

        for event in events:
            audio, _ = obj.query_sound(event['contact_point'], camera_pos)
            if audio is not None:
                synth.trigger_sound(audio, event['impulse'], time_offset=event['time'])

        return synth.get_recording()[:int(obj.sample_rate * duration)]

    def export_all(self, obj_idx: int, output_dir: str, duration: float, dataset: str = None, generate_blend: bool = True):
        """Export animation, audio and optionally Blender file for object at index"""
        import subprocess
        from scipy.io import wavfile

        obj = self.objects[obj_idx]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        anim_path = output_path / f"{obj.name}_anim.txt"
        self.export_animation(obj_idx, str(anim_path))

        audio = self.synthesize_audio(obj_idx, duration)
        audio_path = output_path / f"{obj.name}_audio.wav"
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.8
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(str(audio_path), obj.sample_rate, audio_int16)

        if generate_blend and dataset:
            subprocess.run([
                "blender", "--background", "--python", "blender_render.py",
                "--", "--object", obj.name, "--dataset", dataset,
                "--animation", str(anim_path), "--output", str(output_path), "--preview"
            ], capture_output=True)

        return {"anim": str(anim_path), "audio": str(audio_path)}


def main():
    import newton.examples

    parser = newton.examples.create_parser()
    parser.add_argument('--dataset', type=str, default='/run/media/jim_z/SRC/dev/RealImpact/dataset')
    parser.add_argument('--object', type=str, default='3_CeramicKoiBowl')
    parser.add_argument('--drop-height', type=float, default=0.5)
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--interpolate', action='store_true')
    parser.add_argument('--use-convex-hull', action='store_true', help='Use convex hull for collision mesh')
    parser.add_argument('--collision-faces', type=int, default=2000, help='Target faces for collision mesh decimation')
    parser.add_argument('--max-contacts-per-pair', type=int, default=2048, help='Rigid contact points per shape pair')
    parser.add_argument('--collision-margin', type=float, default=0.02, help='Rigid contact margin (meters)')

    viewer, args = newton.examples.init(parser)

    print("=" * 60)
    print("RealImpact Simulation (Newton Rigid Body Physics)")
    print("=" * 60)
    print(f"Device: {wp.get_device()}")

    loader = RealImpactLoader(args.dataset)
    print(f"\nLoading object: {args.object}")
    obj = loader.load_object(args.object, load_audio=True)
    print(f"  Vertices: {len(obj.vertices)}")
    print(f"  Faces: {len(obj.faces)}")

    hit_ids, hit_positions = obj.get_unique_hit_info()
    print(f"  Hit vertices: {len(hit_ids)}")

    sim = NewtonRigidBodySim(
        obj,
        drop_height=args.drop_height,
        viewer=viewer,
        use_convex_hull=args.use_convex_hull,
        target_collision_faces=args.collision_faces,
        max_contacts_per_pair=args.max_contacts_per_pair,
        collision_margin=args.collision_margin,
    )

    synth = SoundSynthesizer(sample_rate=obj.sample_rate)
    synth.init_recording(args.duration + 3.0)

    camera_position = np.array([0.5, 0.25, 0.5], dtype=np.float32)

    print(f"\nStarting simulation (duration: {args.duration}s, {int(args.duration * sim.fps)} frames)...")
    start_time = time.time()

    num_frames = int(args.duration * sim.fps)
    for frame in range(num_frames):
        collision = sim.step()
        sim.render()

        if collision:
            hit_pos = collision.get('contact_point', collision['position'])
            impulse = collision['impulse']
            audio, weight = obj.query_sound(hit_pos, camera_position, interpolate=args.interpolate)
            if audio is not None:
                synth.trigger_sound(audio, impulse, time_offset=collision['time'])
                print(f"  Collision at t={collision['time']:.3f}s, impulse={impulse:.4f}N·s, contacts={collision['contact_count']}")

        if frame % 60 == 0:
            print(f"  Frame {frame}/{num_frames} ({100*frame/num_frames:.0f}%)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = output_dir / f"{args.object}_audio.wav"
    synth.save_recording(str(audio_path))

    anim_path = output_dir / f"{args.object}_anim.txt"
    sim.export_animation(str(anim_path))

    meta_path = output_dir / f"{args.object}_meta.json"
    sim.export_metadata(str(meta_path), args.duration)

    print(f"\nSimulation complete!")
    print(f"  Duration: {time.time() - start_time:.2f}s real time")
    print(f"  Collisions: {len(sim.collision_events)}")
    print(f"  Output: {output_dir}")

    # Auto-generate Blender file
    import subprocess
    blend_path = output_dir / f"{args.object}.blend"
    subprocess.run([
        "blender", "--background", "--python", "blender_render.py",
        "--", "--object", args.object, "--dataset", args.dataset,
        "--animation", str(anim_path), "--output", str(output_dir), "--preview"
    ])
    print(f"Blender file: {blend_path}")


if __name__ == "__main__":
    main()
