### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from pix2pixHD.options.test_options import TestOptions
from pix2pixHD.data.data_loader import CreateDataLoader
from pix2pixHD.models.models import create_model
import util.utility as util
from util.visualizer import Visualizer
from util import html
import torch

opt = TestOptions().parse(save=False)
#opt.nThreads = 1   # test code only supports nThreads = 1
#opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

if __name__=='__main__':
    # test
    if not opt.engine and not opt.onnx:
        model = create_model(opt)
        if opt.data_type == 16:
            model.half()
        elif opt.data_type == 8:
            model.type(torch.uint8)

        if opt.verbose:
            print(model)

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        if opt.data_type == 16:
            data['label'] = data['label'].half()
            data['inst']  = data['inst'].half()
        elif opt.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst']  = data['inst'].uint8()
        if opt.export_onnx:
            print ("Exporting to ONNX: ", opt.export_onnx)
            assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
            torch.onnx.export(model, [data['label'], data['inst']],
                              opt.export_onnx, verbose=True)
            exit(0)

        #generated,generated_pre = model.inference(data['label'], data['label_pre'],data['bg'])
        generated, generated_pre = model((data['label'], data['label_pre'], data['bg']))
        # visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
        #                        ('synthesized_image', util.tensor2im(generated.data[0]))])
        for j in range(opt.batchSize):
            visuals = OrderedDict([('synthesized_image', util.tensor2im(generated.data[j]))])
            img_path = data['path']
            print('process image... %s' % img_path[j])
            visualizer.save_images(webpage, visuals, [img_path[j]])

    webpage.save()