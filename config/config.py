"""
Main config file of video and camera parameters.
"""
import yaml


class ParametersConfig:
    """
    Clase en la que se almacenan los parametros del registration
    """
    def __init__(self, yaml_file='config/parameters.yml'):
        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)

            self.numEpochsCoarseLoc = config.get('numEpochs').get('coarseLoc')
            self.numEpochsFineLoc = config.get('numEpochs').get('fineLoc')
            self.numEpochsGlobalLoc = config.get('numEpochs').get('globalLoc')

            self.epochLengthCoarseLoc = config.get('epochLength').get('coarseLoc')
            self.epochLengthFineLoc = config.get('epochLength').get('fineLoc')
            self.epochLengthGlobalLoc = config.get('epochLength').get('globalLoc')

            self.rPosFineLoc = config.get('thresholds').get('fineLoc').get('rPos')
            self.rNegFineLoc = config.get('thresholds').get('fineLoc').get('rNeg')
            self.rPosGlobalLoc = config.get('thresholds').get('globalLoc').get('rPos')
            self.rNegGlobalLoc = config.get('thresholds').get('globalLoc').get('rNeg')

            self.losses = config.get('losses').get('losses')
            self.lossesCoarseLocTraining = config.get('losses').get('coarseLoc')
            self.lossesFineLocTraining = config.get('losses').get('fineLoc')
            self.lossesGlobalLocTraining = config.get('losses').get('globalLoc')

            self.lossAbreviations = config.get('losses').get('abreviations')

            # optimize this part
            self.marginsCoarseLoc = config.get('margins').get('coarseLoc')
            self.marginsFineLoc = config.get('margins').get('fineLoc')
            self.marginsGlobalLoc = config.get('margins').get('globalLoc')

            self.showLoss = config.get('showLoss')
            self.doValidation = config.get('doValidation')

            self.kFineLoc = config.get('k').get('fineLoc')
            self.kGlobalLoc = config.get('k').get('globalLoc')


PARAMETERS = ParametersConfig()
