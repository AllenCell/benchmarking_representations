from upath import UPath as Path

class Report:
    cell_id_col: str

    def __init__(self, name=None, output_dir=None, verbose=False, cell_id_col="",
                 fov_id_col="", structure_name_col="", **kwargs):

        self.verbose = verbose
        self.cell_id_col = cell_id_col
        self.fov_id_col = fov_id_col
        self.structure_name_col = structure_name_col

        if output_dir is None:
            raise ValueError("Must specify output dir")
        self.output_dir = output_dir

        if name is None:
            raise ValueError("Must specify report name")
        self.name = name

    def build(self, manifest):
        raise NotImplementedError

