class ModelParameters:
    """Subset of Java ModelParameters for Python runtime parity."""

    valid = False

    sar2sar_edge_threshold = 0.0
    sar2normal_edge_threshold = 0.0
    normal2sar_edge_threshold = 0.0
    normal2normal_edge_threshold = 0.0

    @classmethod
    def is_valid(cls):
        return cls.valid

    @classmethod
    def should_add_edge(cls, orig, bene):
        if not cls.valid:
            return True

        num_neighbors = len(orig.get_bene_list())
        prop_sar_bene = orig.get_prop_sar_bene()

        if orig.is_sar_account():
            if bene.is_sar_account():
                return prop_sar_bene >= cls.sar2sar_edge_threshold
            return prop_sar_bene >= cls.sar2normal_edge_threshold

        if bene.is_sar_account():
            if cls.normal2sar_edge_threshold <= 0.0:
                return True
            return num_neighbors > int(1.0 / cls.normal2sar_edge_threshold) and prop_sar_bene >= cls.normal2sar_edge_threshold
        return prop_sar_bene >= cls.normal2normal_edge_threshold
