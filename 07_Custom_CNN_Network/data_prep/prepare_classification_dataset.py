import shutil
from os import listdir
from os.path import isdir, isfile, join

from .utils import create_path


class DatasetPreparator:
    def __init__(self, datapath, trainpath):
        self.data_path = datapath
        self.output_path = trainpath

    def dataset_creator(self):
        print("Accessing Dataset....")
        create_path(self.output_path)
        create_path(join(self.output_path, 'train'))
        classes = [d for d in listdir(self.data_path) if isdir(join(self.data_path, d))]
        csv_path = join(self.output_path, 'train.csv')
        with open(csv_path, 'w+') as writer:
            writer.write('ids,cls\n')
            for cls in classes:
                anno_files = [f for f in listdir(join(self.data_path, cls)) if (
                    isfile(join(self.data_path, cls, f)) and f.endswith('.jpg'))]

                for file in anno_files:
                    writer.write('{},{}\n'.format(file, cls))
                    shutil.copyfile(join(self.data_path, cls, file), join(self.output_path, 'train', file))
        print('Created Train.csv File')
        return csv_path
