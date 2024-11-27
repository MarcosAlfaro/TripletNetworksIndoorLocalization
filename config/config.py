import yaml
import os

current_directory = os.path.dirname(os.path.realpath(__file__))


class Config():
    def __init__(self, yaml_file=os.path.join(current_directory, 'parameters.yaml')):
        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)

            self.device = config.get('device')

            self.csvDir = config.get('csvDir')
            self.figuresDir = config.get('figuresDir')
            self.datasetDir = config.get('datasetDir')
            self.modelsDir = config.get('modelsDir')

            self.lenList = config.get('lenList')
            self.testEnv = config.get('testEnv')
            self.envs = config.get('envs')

            self.epochLength_coarseLoc = config.get('epochLength_coarseLoc')
            self.epochLength_fineLoc = config.get('epochLength_fineLoc')
            self.epochLength_globalLoc = config.get('epochLength_globalLoc')

            self.batchSize = config.get('batchSize')
            self.numModelsSaved = config.get('numModelsSaved')


            self.sameP = config.get('sameProbability')

            self.losses = config.get('losses')
            self.selectedLosses = config.get('selectedLosses')
            self.lossAbreviations = config.get('lossAbreviations')

            self.marginsCoarseLoc = config.get('marginsCoarseLoc')
            self.marginsFineLoc = config.get('marginsFineLoc')
            self.marginsGlobalLoc = config.get('marginsGlobalLoc')

            self.numEpochs = config.get('numEpochs')
            self.numIterations = config.get('numIterations')


            self.rPos = config.get('rPos')
            self.rNeg = config.get('rNeg')

            self.kMax = config.get('kMax')

            self.occlusionValues = config.get('occlusionValues')
            self.noiseValues = config.get('noiseValues')
            self.blurValues = config.get('blurValues')

            self.lossExp2 = config.get('lossExp2')
            self.marginExp2 = config.get('marginExp2')


PARAMS = Config()
