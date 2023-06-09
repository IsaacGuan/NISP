import os
import sys
import argparse
import lottery_ticket_pruner
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.framework.ops import disable_eager_execution
from feature_sampler import FeatureSampler
from color_mapper import ColorMapper
from uv_mapper import UVMapper
from decomposed_uv_mapper import DecomposedUVMapper
from ray_tracer import *
from utils import *

disable_eager_execution()

tf.keras.utils.get_custom_objects().update({
    'sin': tf.math.sin
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
RESULTS_COLOR_DIR = os.path.join(RESULTS_DIR, 'color-mapper')
RESULTS_UV_DIR = os.path.join(RESULTS_DIR, 'uv-mapper')
RESULTS_DECOMPOSED_UV_DIR = os.path.join(RESULTS_DIR, 'decomposed-uv-mapper')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='mario', help='name of the model')
    parser.add_argument('--train', action='store_true', help='train or test')
    parser.add_argument('--texture_model_type', default='uv_decomposed', help='type of the texture model (color/uv/uv_decomposed)')
    parser.add_argument('--fourier_max_freq', type=int, default=0, help='number of fourier features as input to the network')
    parser.add_argument('--use_siren', action='store_true', help='use siren layers')
    parser.add_argument('--use_sdf', action='store_true', help='use sdf to train the network')
    parser.add_argument('--sample_mode', default='Importance', help='mode of sampling points and features')
    parser.add_argument('--sample_num', type=int, default=64*64*64*4, help='number of sample points')
    parser.add_argument('--epoch_num', type=int, default=2000, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='number of training samples per batch')
    parser.add_argument('--validate', action='store_true', help='validate during training or not')
    parser.add_argument('--prune', action='store_true', help='pune the network or not')
    parser.add_argument('--use_normal_map', action='store_true', help='use the normal map for rendering or not')
    args = parser.parse_args()
    model_name = args.model_name
    is_train = args.train
    texture_model_type = args.texture_model_type
    fourier_max_freq = args.fourier_max_freq
    use_siren = args.use_siren
    use_sdf = args.use_sdf
    sample_mode = args.sample_mode
    sample_num = args.sample_num
    epoch_num = args.epoch_num
    batch_size = args.batch_size
    validate = args.validate
    prune = args.prune
    use_normal_map = args.use_normal_map

    if is_train:
        feature_sampler = FeatureSampler()
        feature_sampler.load_object(model_name)
        feature_sampler.sample(sample_mode=sample_mode, sample_num=sample_num)

        if validate:
            np.save(os.path.join(RESULTS_DIR, model_name + '_surface_points.npy'), feature_sampler.surface_points)

        surface_component_gt, _, surface_uv_gt, surface_color_gt = feature_sampler.get_component_distance_uv_color(feature_sampler.surface_points)

        if texture_model_type == 'color' or texture_model_type == 'all':
            color_mapper = ColorMapper(
                model_name = model_name,
                batch_size = batch_size,
                fourier_max_freq = fourier_max_freq,
                use_siren = use_siren,
                use_sdf = use_sdf)
            color_mapper.create_model()

            loss_color_list, loss_distance_list, ae_color_list, mae_color_list = color_mapper.train(
                epoch_num = epoch_num,
                point_train = feature_sampler.point_samples,
                color_train = feature_sampler.color_samples,
                distance_train = feature_sampler.distance_samples,
                point_validate = feature_sampler.surface_points,
                color_gt = surface_color_gt,
                validate = validate)

            if loss_color_list:
                plt.figure()
                plt.plot(loss_color_list)
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.savefig(os.path.join(RESULTS_COLOR_DIR, model_name + '_color_loss.png'))
                plt.savefig(os.path.join(RESULTS_COLOR_DIR, model_name + '_color_loss.pdf'))

                f = open(os.path.join(RESULTS_COLOR_DIR, model_name + '_color_loss.txt'), 'w')
                for row in loss_color_list:
                    f.write(str(row) + '\n')
                f.close()

            if loss_distance_list:
                plt.figure()
                plt.plot(loss_distance_list)
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.savefig(os.path.join(RESULTS_COLOR_DIR, model_name + '_distance_loss.png'))
                plt.savefig(os.path.join(RESULTS_COLOR_DIR, model_name + '_distance_loss.pdf'))

                f = open(os.path.join(RESULTS_COLOR_DIR, model_name + '_distance_loss.txt'), 'w')
                for row in loss_distance_list:
                    f.write(str(row) + '\n')
                f.close()

            if mae_color_list:
                plt.figure()
                plt.plot(mae_color_list)
                plt.ylabel('MAE')
                plt.xlabel('Epoch')
                plt.savefig(os.path.join(RESULTS_COLOR_DIR, model_name + '_color_mae.png'))
                plt.savefig(os.path.join(RESULTS_COLOR_DIR, model_name + '_color_mae.pdf'))

                f = open(os.path.join(RESULTS_COLOR_DIR, model_name + '_color_mae.txt'), 'w')
                for row in mae_color_list:
                    f.write(str(row) + '\n')
                f.close()

            if ae_color_list:
                ae_color = np.array(ae_color_list)
                np.save(os.path.join(RESULTS_COLOR_DIR, model_name + '_color_ae.npy'), ae_color)

            with open(os.path.join(RESULTS_COLOR_DIR, model_name + '.json'), 'w') as json_file:
                json_file.write(color_mapper.model.to_json())

            color_mapper.model.save_weights(os.path.join(RESULTS_COLOR_DIR, model_name + '.h5'))

            fout = open(os.path.join(RESULTS_COLOR_DIR, model_name + '.out'), 'w')
            sys.stdout = fout

            color_predicted = color_mapper.model.predict(feature_sampler.point_samples)
            color_gt = feature_sampler.color_samples
            print('Color MAE: %f' % np.absolute(np.subtract(color_gt, color_predicted)).mean())

            surface_color_predicted = color_mapper.model.predict(feature_sampler.surface_points)
            print('Surface color MAE: %f' % np.absolute(np.subtract(surface_color_gt, surface_color_predicted)).mean())

            fout.close()
            sys.stdout = sys.__stdout__

            write_ply(os.path.join(RESULTS_COLOR_DIR, model_name + '_pred.ply'), feature_sampler.surface_points, surface_color_predicted * 255)
            write_ply(os.path.join(RESULTS_COLOR_DIR, model_name + '_gt.ply'), feature_sampler.surface_points, surface_color_gt * 255)

        if texture_model_type == 'uv' or texture_model_type == 'all':
            uv_mapper = UVMapper(
                model_name = model_name,
                batch_size = batch_size,
                fourier_max_freq = fourier_max_freq,
                use_siren = use_siren,
                use_sdf = use_sdf)
            uv_mapper.create_model()

            loss_uv_list, loss_distance_list, ae_uv_list, mae_uv_list, ae_color_list, mae_color_list = uv_mapper.train(
                epoch_num = epoch_num,
                point_train = feature_sampler.point_samples,
                uv_train = feature_sampler.uv_samples,
                distance_train = feature_sampler.distance_samples,
                point_validate = feature_sampler.surface_points,
                uv_gt = surface_uv_gt,
                color_gt = surface_color_gt,
                validate = validate,
                tex = feature_sampler.tex)

            if loss_uv_list:
                plt.figure()
                plt.plot(loss_uv_list)
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.savefig(os.path.join(RESULTS_UV_DIR, model_name + '_uv_loss.png'))
                plt.savefig(os.path.join(RESULTS_UV_DIR, model_name + '_uv_loss.pdf'))

                f = open(os.path.join(RESULTS_UV_DIR, model_name + '_uv_loss.txt'), 'w')
                for row in loss_uv_list:
                    f.write(str(row) + '\n')
                f.close()

            if loss_distance_list:
                plt.figure()
                plt.plot(loss_distance_list)
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.savefig(os.path.join(RESULTS_UV_DIR, model_name + '_distance_loss.png'))
                plt.savefig(os.path.join(RESULTS_UV_DIR, model_name + '_distance_loss.pdf'))

                f = open(os.path.join(RESULTS_UV_DIR, model_name + '_distance_loss.txt'), 'w')
                for row in loss_distance_list:
                    f.write(str(row) + '\n')
                f.close()

            if mae_uv_list:
                plt.figure()
                plt.plot(mae_uv_list)
                plt.ylabel('MAE')
                plt.xlabel('Epoch')
                plt.savefig(os.path.join(RESULTS_UV_DIR, model_name + '_uv_mae.png'))
                plt.savefig(os.path.join(RESULTS_UV_DIR, model_name + '_uv_mae.pdf'))

                f = open(os.path.join(RESULTS_UV_DIR, model_name + '_uv_mae.txt'), 'w')
                for row in mae_uv_list:
                    f.write(str(row) + '\n')
                f.close()
                f.close()

            if mae_color_list:
                plt.figure()
                plt.plot(mae_color_list)
                plt.ylabel('MAE')
                plt.xlabel('Epoch')
                plt.savefig(os.path.join(RESULTS_UV_DIR, model_name + '_color_mae.png'))
                plt.savefig(os.path.join(RESULTS_UV_DIR, model_name + '_color_mae.pdf'))

                f = open(os.path.join(RESULTS_UV_DIR, model_name + '_color_mae.txt'), 'w')
                for row in mae_color_list:
                    f.write(str(row) + '\n')
                f.close()

            if ae_uv_list:
                ae_uv = np.array(ae_uv_list)
                np.save(os.path.join(RESULTS_UV_DIR, model_name + '_uv_ae.npy'), ae_uv)

            if ae_color_list:
                ae_color = np.array(ae_color_list)
                np.save(os.path.join(RESULTS_UV_DIR, model_name + '_color_ae.npy'), ae_color)

            with open(os.path.join(RESULTS_UV_DIR, model_name + '.json'), 'w') as json_file:
                json_file.write(uv_mapper.model.to_json())

            uv_mapper.model.save_weights(os.path.join(RESULTS_UV_DIR, model_name + '.h5'))

            fout = open(os.path.join(RESULTS_UV_DIR, model_name + '.out'), 'w')
            sys.stdout = fout

            uv_predicted = uv_mapper.model.predict(feature_sampler.point_samples)
            uv_gt = feature_sampler.uv_samples
            color_predicted = uv_to_color(uv_predicted, feature_sampler.tex)
            color_gt = feature_sampler.color_samples
            print('UV MAE: %f' % np.absolute(np.subtract(uv_gt, uv_predicted)).mean())
            print('Color MAE: %f' % np.absolute(np.subtract(color_gt, color_predicted)).mean())

            surface_uv_predicted = uv_mapper.model.predict(feature_sampler.surface_points)
            surface_color_predicted = uv_to_color(surface_uv_predicted, feature_sampler.tex)
            print('Surface UV MAE: %f' % np.absolute(np.subtract(surface_uv_gt, surface_uv_predicted)).mean())
            print('Surface color MAE: %f' % np.absolute(np.subtract(surface_color_gt, surface_color_predicted)).mean())

            fout.close()
            sys.stdout = sys.__stdout__

            write_ply(os.path.join(RESULTS_UV_DIR, model_name + '_pred.ply'), feature_sampler.surface_points, surface_color_predicted * 255)
            write_ply(os.path.join(RESULTS_UV_DIR, model_name + '_gt.ply'), feature_sampler.surface_points, surface_color_gt * 255)

            plt.figure()
            plt.axis('off')
            plt.scatter(surface_uv_predicted[:,0], surface_uv_predicted[:,1], s=1)
            plt.savefig(os.path.join(RESULTS_UV_DIR, model_name + '_uv_layout_pred.png'))

            plt.figure()
            plt.axis('off')
            plt.scatter(surface_uv_gt[:,0], surface_uv_gt[:,1], s=1)
            plt.savefig(os.path.join(RESULTS_UV_DIR, model_name + '_uv_layout_gt.png'))

        if texture_model_type == 'uv_decomposed' or texture_model_type == 'all':
            component_samples_onehot = np.zeros((feature_sampler.component_samples.size, feature_sampler.component_samples.max()+1))
            component_samples_onehot[np.arange(feature_sampler.component_samples.size),feature_sampler.component_samples] = 1

            components = trimesh.graph.connected_component_labels(feature_sampler.mesh.face_adjacency)

            decomposed_uv_mapper = DecomposedUVMapper(
                model_name = model_name,
                components_num = len(np.unique(components)),
                batch_size = batch_size,
                fourier_max_freq = fourier_max_freq,
                use_siren = use_siren,
                use_sdf = use_sdf)
            decomposed_uv_mapper.create_model()

            if prune:
                initial_weights_point2component = decomposed_uv_mapper.point2component.get_weights()
                initial_weights_point2UV = decomposed_uv_mapper.point2UV.get_weights()
                pruner_point2component = lottery_ticket_pruner.LotteryTicketPruner(decomposed_uv_mapper.point2component)
                pruner_point2UV = lottery_ticket_pruner.LotteryTicketPruner(decomposed_uv_mapper.point2UV)

            loss_component_list, loss_uv_list, loss_distance_list, accuracy_component_list, precision_component_list, ae_uv_list, mae_uv_list, ae_color_list, mae_color_list = decomposed_uv_mapper.train(
                epoch_num = epoch_num,
                point_train = feature_sampler.point_samples,
                component_train = feature_sampler.component_samples,
                component_onehot_train = component_samples_onehot,
                uv_train = feature_sampler.uv_samples,
                distance_train = feature_sampler.distance_samples,
                point_validate = feature_sampler.surface_points,
                component_gt = surface_component_gt,
                uv_gt = surface_uv_gt,
                color_gt = surface_color_gt,
                validate = validate,
                tex = feature_sampler.tex)

            if loss_component_list:
                plt.figure()
                plt.plot(loss_component_list)
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_component_loss.png'))
                plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_component_loss.pdf'))

                f = open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_component_loss.txt'), 'w')
                for row in loss_component_list:
                    f.write(str(row) + '\n')
                f.close()

            if loss_uv_list:
                plt.figure()
                plt.plot(loss_uv_list)
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_uv_loss.png'))
                plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_uv_loss.pdf'))

                f = open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_uv_loss.txt'), 'w')
                for row in loss_uv_list:
                    f.write(str(row) + '\n')
                f.close()

            if loss_distance_list:
                plt.figure()
                plt.plot(loss_distance_list)
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_distance_loss.png'))
                plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_distance_loss.pdf'))

                f = open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_distance_loss.txt'), 'w')
                for row in loss_distance_list:
                    f.write(str(row) + '\n')
                f.close()

            if accuracy_component_list:
                plt.figure()
                plt.plot(accuracy_component_list)
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_component_accuracy.png'))
                plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_component_accuracy.pdf'))

                f = open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_component_accuracy.txt'), 'w')
                for row in accuracy_component_list:
                    f.write(str(row) + '\n')
                f.close()

            if precision_component_list:
                plt.figure()
                plt.plot(precision_component_list)
                plt.ylabel('Precision')
                plt.xlabel('Epoch')
                plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_component_precision.png'))
                plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_component_precision.pdf'))

                f = open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_component_precision.txt'), 'w')
                for row in precision_component_list:
                    f.write(str(row) + '\n')
                f.close()

            if mae_uv_list:
                plt.figure()
                plt.plot(mae_uv_list)
                plt.ylabel('MAE')
                plt.xlabel('Epoch')
                plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_uv_mae.png'))
                plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_uv_mae.pdf'))

                f = open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_uv_mae.txt'), 'w')
                for row in mae_uv_list:
                    f.write(str(row) + '\n')
                f.close()
                f.close()

            if mae_color_list:
                plt.figure()
                plt.plot(mae_color_list)
                plt.ylabel('MAE')
                plt.xlabel('Epoch')
                plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_color_mae.png'))
                plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_color_mae.pdf'))

                f = open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_color_mae.txt'), 'w')
                for row in mae_color_list:
                    f.write(str(row) + '\n')
                f.close()

            if ae_uv_list:
                ae_uv = np.array(ae_uv_list)
                np.save(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_uv_ae.npy'), ae_uv)

            if ae_color_list:
                ae_color = np.array(ae_color_list)
                np.save(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_color_ae.npy'), ae_color)

            with open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_point2component.json'), 'w') as json_file:
                json_file.write(decomposed_uv_mapper.point2component.to_json())

            decomposed_uv_mapper.point2component.save_weights(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_point2component.h5'))

            with open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_point2UV.json'), 'w') as json_file:
                json_file.write(decomposed_uv_mapper.point2UV.to_json())

            decomposed_uv_mapper.point2UV.save_weights(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_point2UV.h5'))

            with open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '.json'), 'w') as json_file:
                json_file.write(decomposed_uv_mapper.model.to_json())

            decomposed_uv_mapper.model.save_weights(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '.h5'))

            fout = open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '.out'), 'w')
            sys.stdout = fout

            component_predicted = decomposed_uv_mapper.point2component.predict(feature_sampler.point_samples)
            component_predicted = np.argmax(component_predicted, axis=1)
            accuracy, precision = compute_accuracy_precision(component_predicted, feature_sampler.component_samples)
            print('Component accuracy: %f' % accuracy)
            print('Component precision: %f' % precision)

            uv_predicted = decomposed_uv_mapper.point2UV.predict([feature_sampler.point_samples, feature_sampler.component_samples])
            uv_gt = feature_sampler.uv_samples
            color_predicted = uv_to_color(uv_predicted, feature_sampler.tex)
            color_gt = feature_sampler.color_samples
            print('UV MAE (GT component): %f' % np.absolute(np.subtract(uv_gt, uv_predicted)).mean())
            print('Color MAE (GT component): %f' % np.absolute(np.subtract(color_gt, color_predicted)).mean())

            uv_predicted = decomposed_uv_mapper.model.predict(feature_sampler.point_samples)
            uv_gt = feature_sampler.uv_samples
            color_predicted = uv_to_color(uv_predicted, feature_sampler.tex)
            color_gt = feature_sampler.color_samples
            print('UV MAE (predicted component): %f' % np.absolute(np.subtract(uv_gt, uv_predicted)).mean())
            print('Color MAE (predicted component): %f' % np.absolute(np.subtract(color_gt, color_predicted)).mean())

            surface_component_predicted = decomposed_uv_mapper.point2component.predict(feature_sampler.surface_points)
            surface_component_predicted = np.argmax(surface_component_predicted, axis=1)
            accuracy, precision = compute_accuracy_precision(surface_component_predicted, surface_component_gt)
            print('Surface component accuracy: %f' % accuracy)
            print('Surface component precision: %f' % precision)

            surface_uv_predicted = decomposed_uv_mapper.point2UV.predict([feature_sampler.surface_points, surface_component_gt])
            surface_color_predicted = uv_to_color(surface_uv_predicted, feature_sampler.tex)
            print('Surface UV MAE (GT component): %f' % np.absolute(np.subtract(surface_uv_gt, surface_uv_predicted)).mean())
            print('Surface color MAE (GT component): %f' % np.absolute(np.subtract(surface_color_gt, surface_color_predicted)).mean())

            surface_uv_predicted = decomposed_uv_mapper.model.predict(feature_sampler.surface_points)
            surface_color_predicted = uv_to_color(surface_uv_predicted, feature_sampler.tex)
            print('Surface UV MAE (predicted component): %f' % np.absolute(np.subtract(surface_uv_gt, surface_uv_predicted)).mean())
            print('Surface color MAE (predicted component): %f' % np.absolute(np.subtract(surface_color_gt, surface_color_predicted)).mean())

            fout.close()
            sys.stdout = sys.__stdout__

            write_ply(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_pred.ply'), feature_sampler.surface_points, surface_color_predicted * 255)
            write_ply(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_gt.ply'), feature_sampler.surface_points, surface_color_gt * 255)

            plt.figure()
            plt.axis('off')
            plt.scatter(surface_uv_predicted[:,0], surface_uv_predicted[:,1], s=1)
            plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_uv_layout_pred.png'))

            plt.figure()
            plt.axis('off')
            plt.scatter(surface_uv_gt[:,0], surface_uv_gt[:,1], s=1)
            plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_uv_layout_gt.png'))

            if prune:
                trained_weights_point2component = decomposed_uv_mapper.point2component.get_weights()
                trained_weights_point2UV = decomposed_uv_mapper.point2UV.get_weights()
                pruner_point2component.set_pretrained_weights(decomposed_uv_mapper.point2component)
                pruner_point2UV.set_pretrained_weights(decomposed_uv_mapper.point2UV)

                prune_strategy = 'smallest_weights'

                accuracy_dic = {}
                precision_dic = {}
                mae_gt_component_dic = {}
                mae_pred_component_dic = {}

                num_prune_rounds = 10
                prune_rate = 0.2
                overall_prune_rate = 0.0

                for i in range(num_prune_rounds):
                    prune_rate = pow(prune_rate, 1.0 / (i/30 + 1))
                    overall_prune_rate = overall_prune_rate + prune_rate*(1.0 - overall_prune_rate)

                    pruner_point2component.calc_prune_mask(decomposed_uv_mapper.point2component, prune_rate, prune_strategy)
                    pruner_point2UV.calc_prune_mask(decomposed_uv_mapper.point2UV, prune_rate, prune_strategy)

                    experiment = 'pruned@{:.4f}'.format(overall_prune_rate)

                    decomposed_uv_mapper.point2component.set_weights(initial_weights_point2component)
                    decomposed_uv_mapper.point2UV.set_weights(initial_weights_point2UV)

                    decomposed_uv_mapper.point2component.fit(
                        x = feature_sampler.point_samples,
                        y = component_samples_onehot,
                        batch_size = batch_size,
                        epochs = epoch_num,
                        callbacks = [lottery_ticket_pruner.PrunerCallback(pruner_point2component)])

                    decomposed_uv_mapper.point2UV.fit(
                        x = [feature_sampler.point_samples, feature_sampler.component_samples],
                        y = feature_sampler.uv_samples,
                        batch_size = batch_size,
                        epochs = epoch_num,
                        callbacks = [lottery_ticket_pruner.PrunerCallback(pruner_point2UV)])

                    decomposed_uv_mapper.point2component.save_weights(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_point2component_' + experiment + '.h5'))
                    decomposed_uv_mapper.point2UV.save_weights(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_point2UV_' + experiment + '.h5'))

                    surface_component_predicted = decomposed_uv_mapper.point2component.predict(feature_sampler.surface_points)
                    surface_component_predicted = np.argmax(surface_component_predicted, axis=1)
                    accuracy, precision = compute_accuracy_precision(surface_component_predicted, surface_component_gt)
                    accuracy_dic[experiment] = accuracy
                    precision_dic[experiment] = precision

                    surface_uv_predicted = decomposed_uv_mapper.point2UV.predict([feature_sampler.surface_points, surface_component_gt])
                    mae = np.absolute(np.subtract(surface_uv_gt, surface_uv_predicted)).mean()
                    mae_gt_component_dic[experiment] = mae

                    surface_uv_predicted = decomposed_uv_mapper.point2UV.predict([feature_sampler.surface_points, surface_component_predicted])
                    mae = np.absolute(np.subtract(surface_uv_gt, surface_uv_predicted)).mean()
                    mae_pred_component_dic[experiment] = mae

                f = open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_accuracy_dic.csv'), 'w')
                for key, value in accuracy_dic.items():
                    f.write(str(key) + ',' + str(accuracy_dic[key]) + '\n');
                f.close()

                f = open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_precision_dic.csv'),'w')
                for key, value in precision_dic.items():
                    f.write(str(key) + ',' + str(precision_dic[key]) + '\n');
                f.close()

                f = open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_mae_gt_component_dic.csv'),'w')
                for key, value in mae_gt_component_dic.items():
                    f.write(str(key) + ',' + str(mae_gt_component_dic[key]) + '\n');
                f.close()

                f = open(os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + '_mae_pred_component_dic.csv'),'w')
                for key, value in mae_pred_component_dic.items():
                    f.write(str(key) + ',' + str(mae_pred_component_dic[key]) + '\n');
                f.close()

    else:
        scene = [
            SDF(model_name, vec3(0., 0., 0.), 1., 0., rgb(1., .572, .184), texture_model_type, use_normal_map),
        ]

        (w, h) = (1024, 1024)
        r = float(w) / h

        L_list = [vec3(5., 5., 10.)]
        E_list = [vec3(0., 0., 2.)]

        for i in range(len(E_list)):
            L = L_list[i]
            E = E_list[i]

            if texture_model_type == 'color':
                image_file = os.path.join(RESULTS_COLOR_DIR, model_name + str(i) + '.png')
            elif texture_model_type == 'uv':
                image_file = os.path.join(RESULTS_UV_DIR, model_name + str(i) + '.png')
            elif texture_model_type == 'uv_decomposed':
                image_file = os.path.join(RESULTS_DECOMPOSED_UV_DIR, model_name + str(i) + '.png')
            else:
                raise('Texture model type is not supported...')

            render(w, h, -1.2, 1.2 / r, 1.2, -1.2 / r, L, E, scene, image_file)
