from flamapy.core.discover import DiscoverMetamodels
from flamapy.core.models import VariabilityModel

from flamapy.metamodels.configuration_metamodel.models.configuration import Configuration


class AAFMService:
    dm = DiscoverMetamodels()

    def get_feature_model(self, path: str):
        """
        Serialize a .uvl file into a python object

        :param path: path to the .uvl file
        :return:
        """
        fm = self.dm.use_transformation_t2m(path, 'fm')
        if not self.is_fm_satisfiable(fm):
            raise ValueError("Feature Model not satisfiable")
        return fm

    def get_sat_model(self, feature_model: VariabilityModel):
        """

        :param feature_model: feature model to transform
        :return:
        """
        return self.dm.use_transformation_m2m(feature_model, "pysat")

    def get_bdd_model(self, feature_model: VariabilityModel):
        """

        :param feature_model: feature model to transform
        :return:
        """
        return self.dm.use_transformation_m2m(feature_model, "bdd")

    def is_fm_satisfiable(self, fm: VariabilityModel) -> bool:
        """
        Return true if [fm] has at least one valid configuration

        :param fm: feature model to check
        :return:
        """
        sat_model = self.get_sat_model(fm)
        operation = self.dm.get_operation(sat_model, 'PySATSatisfiable')
        operation.execute(sat_model)
        return operation.get_result()

    def is_config_satisfiable(self, fm: VariabilityModel, config: Configuration) -> bool:
        """
        Return true if partial configuration [config] has at least one valid full configuration

        :param fm: feature model to check
        :param config: partial configuration to check
        :return:
        """
        sat_model = self.get_sat_model(fm)
        operation = self.dm.get_operation(sat_model, 'PySATSatisfiableConfiguration')
        config.is_full = False
        operation.set_configuration(config)
        operation.execute(sat_model)
        return operation.get_result()

    def get_leaf_features(self, fm: VariabilityModel):
        return self.dm.use_operation(fm, 'FMLeafFeatures').get_result()

    def filter(self, fm: VariabilityModel, config: Configuration) -> list[Configuration]:
        if not self.is_config_satisfiable(fm, config):
            raise ValueError("Configuration not satisfiable")
        sat = self.get_sat_model(fm)
        operation = self.dm.get_operation(sat, 'PySATFilter')
        config.is_full = False
        operation.set_configuration(config)
        operation.execute(sat)

        configs = []
        for c in operation.get_result():
            configs.append(self.get_config_from_select_feature(fm, c))
        return configs

    def get_config_from_select_feature(self, fm: VariabilityModel, selected: list[str]) -> Configuration:
        elements = {}
        features = self.get_features(fm.root, None)
        for f in features:
            elements[f.name] = f.name in selected
        return Configuration(elements)

    def sample(self, fm: VariabilityModel, size: int) -> list[Configuration]:
        """
        Uniform random sampling of [size] configuration from [fm]

        :param size: size of the sample
        :param fm: feature model to sample
        :return:
        """

        bdd_model = self.get_bdd_model(fm)
        operation = self.dm.get_operation(bdd_model, 'BDDSampling')
        operation.set_sample_size(size)
        operation.execute(bdd_model)
        return operation.get_result()

    def get_all_config(self, fm: VariabilityModel) -> list[Configuration]:
        bdd_model = self.get_bdd_model(fm)
        operation = self.dm.get_operation(bdd_model, 'BDDConfigurations')
        operation.execute(bdd_model)
        return operation.get_result()

    def get_config_repartition(self, fm : VariabilityModel):
        bdd_model = self.get_bdd_model(fm)
        operation = self.dm.get_operation(bdd_model, 'BDDProductDistribution')
        operation.execute(bdd_model)
        return operation.get_result()

    def get_ref_config(self, fm: VariabilityModel) -> Configuration:
        elements = {}
        features = self.get_features(fm.root, 'reference')
        for feat in features:
            elements[feat.name] = True
        configs = self.filter(fm, Configuration(elements))
        if len(configs) > 1:
            raise ValueError("More than 1 reference configuration found")
        return configs[0]


    def get_features(self, feature, attr: str|None):
        features = []
        if not attr:
            features.append(feature)
        else:
            for attribute in feature.attributes:
                if attribute.name == attr:
                    features.append(feature)

        for relation in feature.relations:
            for child in relation.children:
                features = features + self.get_features(child, attr)

        return features


if __name__ == '__main__':
    srv = AAFMService()
    conf = srv.get_ref_config(srv.get_feature_model("/home/ymerel/fmri-feature-model/model/uvl/preprocessing_pipeline.uvl"))
    print(conf)
