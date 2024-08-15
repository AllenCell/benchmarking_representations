# %%
import quilt3
import os

# %%
b = quilt3.Bucket("s3://allencell")

b.ls(
    "aics/morphology_appropriate_representation_learning/"
    "cellPACK_single_cell_punctate_structure/reference_nuclear_shapes/"
)

# %%
datadir = os.path.join(
    os.path.dirname(__file__), "..", "data", "cellpack", "reference_nuclear_shapes"
)
os.makedirs(datadir, exist_ok=True)
# %%
b.fetch(
    "aics/morphology_appropriate_representation_learning/"
    "cellPACK_single_cell_punctate_structure/reference_nuclear_shapes/",
    f"{datadir}/",
)
