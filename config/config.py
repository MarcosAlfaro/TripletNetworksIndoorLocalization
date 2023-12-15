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

            self.dataset = config.get('dataset')
            self.numExp = 2

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

            self.trainedNetsCoarseLoc = config.get('trainedNets').get('coarseLoc')

            self.losses = config.get('losses').get('losses')
            self.lossesCoarseLocTraining = config.get('losses').get('coarseLoc').get('training')
            self.lossesCoarseLocTest = config.get('losses').get('coarseLoc').get('test')
            self.lossesFineLocTraining = config.get('losses').get('fineLoc').get('training')
            self.lossesFineLocTest = config.get('losses').get('fineLoc').get('test')
            self.lossesGlobalLocTraining = config.get('losses').get('globalLoc').get('training')
            self.lossesGlobalLocTest = config.get('losses').get('globalLoc').get('test')

            self.lossAbreviations = config.get('losses').get('abreviations')

            # optimize this part
            self.marginsCoarseLoc = config.get('margins').get('coarseLoc')
            self.marginsFineLoc = config.get('margins').get('fineLoc')
            self.marginsGlobalLoc = config.get('margins').get('globalLoc')

            self.showLoss = config.get('showLoss')
            self.doValidation = config.get('doValidation')

            self.kFineLoc = config.get('k').get('fineLoc')
            self.kGlobalLoc = config.get('k').get('globalLoc')

            # self.base_dir = config.get('base_dir')
            #
            # self.training_dir = config.get('rutas_imgs').get('training_dir')
            # self.validation_dir = config.get('rutas_imgs').get('validation_dir')
            # self.centro_geom_dir = config.get('rutas_imgs').get('centro_geom_dir')
            # self.test_cloudy_dir = config.get('rutas_imgs').get('test_cloudy_dir')
            # self.test_sunny_dir = config.get('rutas_imgs').get('test_sunny_dir')
            # self.test_night_dir = config.get('rutas_imgs').get('test_night_dir')
            #
            # self.train_csv_dir = config.get('rutas_csv').get('train_csv_dir')
            # self.val_csv_dir = config.get('rutas_csv').get('val_csv_dir')
            # self.data_csv_dir = config.get('rutas_csv').get('data_csv_dir')
            # self.test_csv_dir = config.get('rutas_csv').get('test_csv_dir')
            # self.cloudy_csv_dir = config.get('rutas_csv').get('test').get('cloudy_csv_dir')
            # self.sunny_csv_dir = config.get('rutas_csv').get('test').get('sunny_csv_dir')
            # self.night_csv_dir = config.get('rutas_csv').get('test').get('night_csv_dir')
            #
            # self.train_batch_size = config.get('batch_size').get('train_batch_size')
            # self.val_batch_size = config.get('batch_size').get('val_batch_size')
            # self.test_batch_size = config.get('batch_size').get('test_batch_size')
            #
            # self.train_number_epochs = config.get('train_number_epochs')
            #
            # self.do_dataparallel = config.get('do_dataparallel')
            #
            # self.shuffle_train = config.get('shuffle_train')
            #
            # self.do_validation = config.get('frequency').get('do_validation')
            # self.show_loss = config.get('frequency').get('show_loss')
            #
            # self.exp = config.get('exp')
            #
            # self.test_netLg = config.get('test_net').get('Lg')
            # self.test_netLf = config.get('test_net').get('Lf')
            # self.test_netLf_rad = config.get('test_net').get('Lf_rad')
            # self.test_netLG = config.get('test_net').get('LG')
            #
            # self.loss = config.get('loss')
            # self.margin = config.get('margin')
            # self.alpha = config.get('alpha')
            #
            # self.num_iteraciones = config.get('num_iteraciones')
            #
            # self.max_error_accuracy = config.get('max_error_accuracy')
            # self.min_val_accuracy = config.get('min_val_accuracy')
            # self.accuracy_end_train = config.get('accuracy_end_train')
            # self.error_end_train = config.get('error_end_train')


PARAMETERS = ParametersConfig()
