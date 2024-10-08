import os
import sys

import mitsuba as mi
import numpy as np
from plyfile import PlyData


class XMLTemplates:
    HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1920"/>
            <integer name="height" value="1080"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>

    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
"""
    BALL_SEGMENT = """
    <shape type="sphere">
        <float name="radius" value="0.015"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""
    TAIL = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.2"/>
        </transform>
    </shape>

    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""


class PointCloudRenderer:
    POINTS_PER_OBJECT = 2025
    XML_HEAD = XMLTemplates.HEAD
    XML_BALL_SEGMENT = XMLTemplates.BALL_SEGMENT
    XML_TAIL = XMLTemplates.TAIL

    def __init__(self, file_path):
        self.file_path = file_path
        self.folder, full_filename = os.path.split(file_path)
        self.folder = self.folder or "."
        self.filename, _ = os.path.splitext(full_filename)

    @staticmethod
    def compute_color(x, y, z):
        vec = np.clip(np.array([x, y, z]), 0.001, 1.0)
        vec /= np.linalg.norm(vec)
        return vec

    @staticmethod
    def standardize_point_cloud(pcl, points_per_object=POINTS_PER_OBJECT):
        # pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
        # pcl = pcl[pt_indices]
        mins = np.amin(pcl, axis=0)
        maxs = np.amax(pcl, axis=0)
        center = (mins + maxs) / 2.0
        scale = np.amax(maxs - mins)
        rescale = ((pcl - center) / scale).astype(np.float32)
        unscale = (pcl).astype(np.float32)
        return unscale, center

    def load_point_cloud(self):
        file_extension = os.path.splitext(self.file_path)[1]
        if file_extension == ".npy":
            return np.load(self.file_path, allow_pickle=True)
        elif file_extension == ".npz":
            return np.load(self.file_path)["pred"]
        elif file_extension == ".ply":
            ply_data = PlyData.read(self.file_path)
            return np.column_stack(ply_data["vertex"][t] for t in ("x", "y", "z"))
        else:
            raise ValueError("Unsupported file format.")

    def generate_xml_content(self, pcl, colors, location_as_color=True):
        xml_segments = [self.XML_HEAD]
        for j, point in enumerate(pcl):
            if location_as_color:
                color = self.compute_color(point[0] + 0.5, point[1] + 0.5, point[2] + 0.5 - 0.0125)
            else:
                color = (colors[j][0], colors[j][1], colors[j][2])
            xml_segments.append(self.XML_BALL_SEGMENT.format(point[0], point[1], point[2], *color))
        xml_segments.append(self.XML_TAIL)
        return "".join(xml_segments)

    @staticmethod
    def save_xml_content_to_file(output_file_path, xml_content):
        xml_file_path = f"{output_file_path}.xml"
        with open(xml_file_path, "w") as f:
            f.write(xml_content)
        return xml_file_path

    @staticmethod
    def render_scene(xml_file_path):
        mi.set_variant("scalar_rgb")
        scene = mi.load_file(xml_file_path)
        img = mi.render(scene)
        return img

    @staticmethod
    def rotate_canon_pc(pcl):
        pcl = pcl[:, [2, 0, 1]]

        theta = np.pi * 3 / 4
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        pcl[:, [0, 1]] = pcl[:, [0, 1]].dot(rotation_matrix)  # random rotation (x,z)

        theta = -np.pi * 1 / 4
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        pcl[:, [0, 2]] = pcl[:, [0, 2]].dot(rotation_matrix)  # random rotation (x,z)
        # pcl[:, 0] *= -1
        # pcl[:, 2] += 0.0125
        return pcl

    @staticmethod
    def save_scene(output_file_path, rendered_scene):
        mi.util.write_bitmap(f"{output_file_path}.png", rendered_scene)

    def process(self):
        this_pc_bin = self.file_path.split("/")[-1].split(".")[0].split("_")[-1]
        pcl_data = self.load_point_cloud()
        if len(pcl_data.shape) < 3:
            pcl_data = pcl_data[np.newaxis, :, :]

        for index, pcl in enumerate(pcl_data):
            points = pcl[:, :3] * 0.06
            colors = pcl[:, 3:]
            pcl, center = self.standardize_point_cloud(points)
            center = round(abs(center).max(), 2)

            pcl = pcl[:, [2, 0, 1]]

            # output_filename = f'{self.filename}_{index:02d}_{center}'
            output_filename = f"{self.filename}_{index:02d}"
            output_file_path = f"{self.folder}/{output_filename}"
            print(f"Processing {output_filename}...")
            xml_content = self.generate_xml_content(pcl, colors, location_as_color=True)
            xml_file_path = self.save_xml_content_to_file(output_file_path, xml_content)
            rendered_scene = self.render_scene(xml_file_path)
            self.save_scene(output_file_path, rendered_scene)
            print(f"Finished processing {output_filename}.")


def main(argv):
    if len(argv) < 2:
        print("Filename not provided as argument.")
        return
    if argv[1].split(".")[-1] != "npy":
        items = os.listdir(argv[1])
        items = [argv[1] + i for i in items]
        items = [i for i in items if i.split(".")[-1] == "npy"]
        # items = [i for i in items if "CETN2" in i]
        for i in items:
            renderer = PointCloudRenderer(i)
            renderer.process()
    else:
        renderer = PointCloudRenderer(argv[1])
        renderer.process()


if __name__ == "__main__":
    main(sys.argv)
