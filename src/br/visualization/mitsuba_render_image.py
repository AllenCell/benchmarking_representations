import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import trimesh

mi.set_variant("scalar_rgb")
from mitsuba import ScalarTransform4f as T
from trimesh.transformations import rotation_matrix


def plot(this_mesh_path, save_path, angle, angle2=None, angle3=None, name="mesh"):
    myMesh = trimesh.load(this_mesh_path)

    # Scale the mesh to approximately one unit based on the height
    sf = 1.0
    myMesh.apply_scale(sf / myMesh.extents.max())

    myMesh = myMesh.apply_transform(rotation_matrix(np.deg2rad(angle), [0, 0, -1]))
    if angle2:
        myMesh = myMesh.apply_transform(rotation_matrix(np.deg2rad(angle2), [0, 1, 0]))

    if angle3:
        myMesh = myMesh.apply_transform(rotation_matrix(np.deg2rad(angle3), [1, 0, 0]))
    # myMesh = myMesh.apply_transform(rotation_matrix(np.deg2rad(0), [1,0,0]))

    # Translate the mesh so that it's centroid is at the origin and rests on the ground plane
    myMesh.apply_translation(
        [
            -myMesh.bounds[0, 0] - myMesh.extents[0] / 2.0,
            -myMesh.bounds[0, 1] - myMesh.extents[1] / 2.0,
            -myMesh.bounds[0, 2],
        ]
    )

    # Fix the mesh normals for the mesh
    myMesh.fix_normals()

    # Write the mesh to an external file (Wavefront .obj)
    with open("mesh.obj", "w") as f:
        f.write(trimesh.exchange.export.export_obj(myMesh, include_normals=True))

    # Create a sensor that is used for rendering the scene
    def load_sensor(r, phi, theta):
        # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
        origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])

        return mi.load_dict(
            {
                "type": "perspective",
                "fov": 15.0,
                "to_world": T.look_at(
                    origin=origin, target=[0, 0, myMesh.extents[2] / 2], up=[0, 0, 1]
                ),
                "sampler": {"type": "independent", "sample_count": 16},
                "film": {
                    "type": "hdrfilm",
                    "width": 1024,
                    "height": 768,
                    "rfilter": {
                        "type": "tent",
                    },
                    "pixel_format": "rgb",
                },
            }
        )

    # Scene parameters
    relativeLightHeight = 8

    # A scene dictionary contains the description of the rendering scene.
    scene2 = mi.load_dict(
        {
            "type": "scene",
            # The keys below correspond to object IDs and can be chosen arbitrarily
            "integrator": {"type": "path"},
            "mesh": {
                "type": "obj",
                "filename": "mesh.obj",
                "face_normals": True,  # This prevents smoothing of sharp-corners by discarding surface-normals. Useful for engineering CAD.
                "bsdf": {
                    "type": "pplastic",
                    "diffuse_reflectance": {"type": "rgb", "value": [0.05, 0.03, 0.1]},
                    "alpha": 0.02,
                },
            },
            # A general emitter is used for illuminating the entire scene (renders the background white)
            "light": {"type": "constant", "radiance": 1.0},
            "areaLight": {
                "type": "rectangle",
                # The height of the light can be adjusted below
                "to_world": T.translate([0, 0.0, myMesh.bounds[1, 2] + relativeLightHeight])
                .scale(1.0)
                .rotate([1, 0, 0], 5.0),
                "flip_normals": True,
                "emitter": {
                    "type": "area",
                    "radiance": {
                        "type": "spectrum",
                        "value": 30.0,
                    },
                },
            },
            "floor": {
                "type": "disk",
                "to_world": T.scale(3).translate([0.0, 0.0, 0.0]),
                "material": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": 0.75},
                },
            },
        }
    )

    radius = 7
    phis = [70.0]
    theta = 60.0

    sensors = [load_sensor(radius, phi, theta) for phi in phis]

    """
    Render the Scene
    The render samples are specified in spp
    """
    image = mi.render(scene2, sensor=sensors[0], spp=256)

    # Write the output
    mi.util.write_bitmap(save_path + f"/{name}.png", image)

    # Display the output in an Image
    plt.imshow(image ** (1.0 / 2.2))
    plt.axis("off")


if __name__ == "__main__":
    mesh_path = MESH_PATH
    name = NAME
    this_mesh_path = mesh_path + f"{name}.ply"
    plot(this_mesh_path, -120)
